"""Model zoo for the benchmark.

Every model takes the 12 tensors produced by ``rarepred.data.collate`` (code indices, code mask, visit padding
mask and lengths for diagnoses, medications and procedures) and returns logits over the label vocabulary:
``(B, N)`` for single-exit models, ``(B, D, N)`` for models read out after each of their D block applications.

Families
--------
* ``LabelPrior``      : training label frequencies (no input), a floor for every metric.
* ``BagOfCodes``      : counts of every code in the history, MLP.
* ``VisitGRU``        : sum-pooled visit vectors, GRU over visits (Doctor AI style).
* ``RETAIN``          : reverse-time attention over visits (Choi et al., 2016).
* ``MultiModalBackbone`` (models.py): the MedCo backbone and its fusion / pooling / knowledge variants.
* ``GRAMEmbedding``   : ontology attention over ancestors (Choi et al., 2017) for diagnoses (CCS) and
                        medications (ATC), plugged into the backbone.
* ``DepthStack``      : per-modality stack of PyHealth Transformer blocks applied along a *schedule* of block
                        indices. Schedules express plain deep stacks ``(0, 1, 2, 3)``, a looped single block
                        ``(0, 0, 0, 0)``, a **layer loop** ``(0, 0, 1, 1)`` (each layer repeated before the next)
                        and a **model loop** ``(0, 1, 0, 1)`` (the whole stack repeated). With
                        ``supervision != 'final'`` the model is read out after every application and trained with a
                        fixed prior over depths or a PonderNet gate, which enables adaptive depth at test time.

All stacks with more than one block application use the fix described in Notebook 3: PyHealth's block ends
with ``Dropout(x)`` on the whole residual stream; inside a stack this dropout is removed from the recurrent path
and applied once on the readout branch, for deep, looped and adaptive models alike, so that they differ only in
their schedule.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import random_embedding

try:
    from pyhealth.models.transformer import TransformerBlock
except Exception:  # pragma: no cover
    TransformerBlock = None


def _visits(emb_layer, idx, cmask):
    return (emb_layer(idx) * cmask.unsqueeze(-1)).sum(dim=2)


def _split(batch12):
    b = batch12
    return (b[0], b[1], b[2]), (b[4], b[5], b[6]), (b[8], b[9], b[10])


# ---------------------------------------------------------------------------
# Simple baselines
# ---------------------------------------------------------------------------
class LabelPrior(nn.Module):
    """Predicts the training frequency of each label for every sample."""

    def __init__(self, label_freq):
        super().__init__()
        p = torch.as_tensor(label_freq, dtype=torch.float).clamp(1e-6, 1 - 1e-6)
        self.register_buffer("bias", torch.log(p / (1 - p)))

    def forward(self, *b):
        return self.bias.unsqueeze(0).expand(b[0].shape[0], -1)


class BagOfCodes(nn.Module):
    def __init__(self, C_dx, C_rx, C_px, D_out, hidden=512, dropout=0.5):
        super().__init__()
        self.C = (C_dx, C_rx, C_px)
        self.net = nn.Sequential(nn.Linear(sum(self.C), hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, D_out))

    @staticmethod
    def _counts(idx, cmask, C):
        B = idx.shape[0]
        out = torch.zeros(B, C, device=idx.device)
        out.scatter_add_(1, idx.reshape(B, -1), cmask.reshape(B, -1))
        return torch.log1p(out)

    def forward(self, *b):
        (dx, dm, _), (rx, rm, _), (px, pm, _) = _split(b)
        x = torch.cat([self._counts(dx, dm, self.C[0]), self._counts(rx, rm, self.C[1]),
                       self._counts(px, pm, self.C[2])], dim=1)
        return self.net(x)


class _VisitEncoder(nn.Module):
    """Three embedding matrices, sum pooling, concatenation of modalities per visit."""

    def __init__(self, C_dx, C_rx, C_px, d=256):
        super().__init__()
        self.emb_dx, self.emb_rx, self.emb_px = random_embedding(C_dx, d), random_embedding(C_rx, d), random_embedding(C_px, d)

    def forward(self, b):
        (dx, dm, dp), (rx, rm, _), (px, pm, _) = _split(b)
        v = torch.cat([_visits(self.emb_dx, dx, dm), _visits(self.emb_rx, rx, rm), _visits(self.emb_px, px, pm)], -1)
        return v, ~dp                                           # (B, T, 3d), valid mask


class VisitGRU(nn.Module):
    def __init__(self, C_dx, C_rx, C_px, D_out, d=256, dropout=0.5):
        super().__init__()
        self.enc = _VisitEncoder(C_dx, C_rx, C_px, d)
        self.gru = nn.GRU(3 * d, d, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(d, D_out)

    def forward(self, *b):
        v, valid = self.enc(b)
        lengths = valid.sum(1).clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(v, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)
        return self.fc(self.drop(h[-1]))


class RETAIN(nn.Module):
    def __init__(self, C_dx, C_rx, C_px, D_out, d=256, dropout=0.5):
        super().__init__()
        self.enc = _VisitEncoder(C_dx, C_rx, C_px, d)
        self.proj = nn.Linear(3 * d, d)
        self.gru_a, self.gru_b = nn.GRU(d, d, batch_first=True), nn.GRU(d, d, batch_first=True)
        self.w_a, self.w_b = nn.Linear(d, 1), nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(d, D_out)

    def forward(self, *b):
        v, valid = self.enc(b)
        x = self.drop(self.proj(v))
        lengths = valid.sum(1).clamp(min=1)
        # reverse each sequence within its length
        T = x.shape[1]
        ar = torch.arange(T, device=x.device).unsqueeze(0)
        rev_idx = (lengths.unsqueeze(1) - 1 - ar).clamp(min=0)
        xr = x.gather(1, rev_idx.unsqueeze(-1).expand_as(x))
        ga, _ = self.gru_a(xr)
        gb, _ = self.gru_b(xr)
        mask = ar < lengths.unsqueeze(1)
        alpha = self.w_a(ga).squeeze(-1).masked_fill(~mask, -1e9).softmax(1)
        beta = torch.tanh(self.w_b(gb))
        c = (alpha.unsqueeze(-1) * beta * xr * mask.unsqueeze(-1)).sum(1)
        return self.fc(self.drop(c))


# ---------------------------------------------------------------------------
# GRAM: ontology attention embeddings
# ---------------------------------------------------------------------------
def prefix_ancestors(code: str, system: str):
    """Ancestors from the code itself: CCS '7.2.11.1' -> 7, 7.2, 7.2.11; ATC 'A11C' -> A, A11."""
    if system == "ccs":
        parts = code.split(".")
        return [".".join(parts[:i]) for i in range(1, len(parts))]
    if system == "atc":
        return [a for a in (code[:1], code[:3]) if a and a != code]
    return []


class GRAMEmbedding(nn.Module):
    """g_i = sum_j alpha_ij e_j over the code and its ancestors (GRAM)."""

    def __init__(self, code2idx: dict, d: int, system: str, att_dim: int = 128):
        super().__init__()
        nodes = {}
        anc_lists = {}
        for c, i in code2idx.items():
            chain = [c] + prefix_ancestors(c, system)
            anc_lists[i] = chain
            for a in chain:
                nodes.setdefault(a, len(nodes) + 1)
        n_codes = max(code2idx.values()) + 1
        A = max(len(v) for v in anc_lists.values()) if anc_lists else 1
        idx = torch.zeros(n_codes, A, dtype=torch.long)
        msk = torch.zeros(n_codes, A, dtype=torch.bool)
        for i, chain in anc_lists.items():
            for j, a in enumerate(chain):
                idx[i, j], msk[i, j] = nodes[a], True
        self.register_buffer("anc_idx", idx)
        self.register_buffer("anc_mask", msk)
        self.basic = random_embedding(len(nodes) + 1, d)
        self.att = nn.Sequential(nn.Linear(2 * d, att_dim), nn.Tanh(), nn.Linear(att_dim, 1, bias=False))

    @property
    def weight(self):
        e = self.basic(self.anc_idx)                              # (n, A, d)
        q = e[:, :1].expand_as(e)
        s = self.att(torch.cat([q, e], -1)).squeeze(-1).masked_fill(~self.anc_mask, -1e9)
        g = (s.softmax(-1).unsqueeze(-1) * e).sum(1)
        return g * self.anc_mask[:, :1].float()                  # PAD row stays zero

    def forward(self, idx):
        return F.embedding(idx, self.weight)


# ---------------------------------------------------------------------------
# Depth stacks: deep, looped (layer loop / model loop), adaptive depth
# ---------------------------------------------------------------------------
class DepthStack(nn.Module):
    def __init__(self, C_dx, C_rx, C_px, D_out, d=256, n_heads=1, dropout=0.5, n_blocks=1, schedule=(0,),
                 supervision="final", lam=0.3, gate=False, positional=False, readout_position="first",
                 max_visits=512):
        super().__init__()
        assert TransformerBlock is not None, "pyhealth is required"
        self.schedule = tuple(schedule)
        self.D = len(self.schedule)
        self.supervision, self.lam, self.positional, self.readout_position = supervision, lam, positional, readout_position
        self.multi_exit = supervision != "final"
        self.emb = nn.ModuleList([random_embedding(C, d) for C in (C_dx, C_rx, C_px)])
        self.blocks = nn.ModuleList()
        for _ in range(3):
            blocks = nn.ModuleList([TransformerBlock(d, n_heads, dropout) for _ in range(n_blocks)])
            for blk in blocks:
                blk.dropout = nn.Identity()          # no dropout on the recurrent / residual path
            self.blocks.append(blocks)
        if positional:
            self.pos = nn.Embedding(max_visits, d)
            nn.init.normal_(self.pos.weight, std=0.01)
        self.readout_drop = nn.Dropout(dropout)       # the block-output dropout, applied once before readout
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(3 * d, D_out)
        self.gate = nn.Linear(3 * d, 1) if gate else None

    def _read(self, hs, last):
        if self.readout_position == "first":
            return torch.cat([h[:, 0] for h in hs], 1)
        return torch.cat([h[torch.arange(h.shape[0]), last] for h in hs], 1)

    def forward(self, *b, return_gate=False):
        mods = _split(b)
        hs, masks = [], []
        last = None
        for m, (idx, cm, pad) in enumerate(mods):
            v = _visits(self.emb[m], idx, cm)
            valid = (~pad).long()
            if self.positional:      # distance to the most recent visit: 0 = last visit
                L = valid.sum(1, keepdim=True)
                rel = (L - 1 - torch.arange(v.shape[1], device=v.device).unsqueeze(0)).clamp(min=0)
                v = v + self.pos(rel) * valid.unsqueeze(-1)
            hs.append(v)
            masks.append(torch.einsum("ab,ac->abc", valid, valid))
            last = (valid.sum(1) - 1).clamp(min=0)
        logits, gates = [], []
        for t, k in enumerate(self.schedule):
            hs = [self.blocks[m][k](hs[m], masks[m]) for m in range(3)]
            if self.multi_exit or t == self.D - 1:
                rep = self._read(hs, last)
                logits.append(self.fc(self.dropout(self.readout_drop(rep))))
                if self.gate is not None:
                    gates.append(self.gate(rep).squeeze(-1))
        if not self.multi_exit:
            return logits[-1]
        out = torch.stack(logits, 1)                              # (B, D, N)
        if return_gate and self.gate is not None:
            return out, torch.stack(gates, 1)
        return out

    # -- objective -----------------------------------------------------------
    def depth_prior(self, device):
        D = self.D
        if self.supervision == "uniform":
            p = torch.full((D,), 1.0 / D)
        else:                                                     # truncated geometric
            p = torch.tensor([self.lam * (1 - self.lam) ** t for t in range(D)])
            p = p / p.sum()
        return p.to(device)

    def loss(self, b, y):
        if not self.multi_exit:
            return F.binary_cross_entropy_with_logits(self(*b), y)
        if self.gate is None:
            out = self(*b)
            per = F.binary_cross_entropy_with_logits(out, y.unsqueeze(1).expand_as(out), reduction="none").mean(-1)
            return (per * self.depth_prior(y.device)).sum(1).mean()
        out, g = self(*b, return_gate=True)                       # PonderNet
        per = F.binary_cross_entropy_with_logits(out, y.unsqueeze(1).expand_as(out), reduction="none").mean(-1)
        logh = F.logsigmoid(g)
        log1mh = F.logsigmoid(-g)
        cum = torch.cumsum(torch.cat([torch.zeros_like(log1mh[:, :1]), log1mh[:, :-1]], 1), 1)
        logq = logh + cum
        logq = torch.cat([logq[:, :-1], cum[:, -1:]], 1)          # last step takes the remainder
        q = logq.exp()
        prior = self.depth_prior(y.device).log()
        kl = (q * (logq - prior)).sum(1)
        return ((q * per).sum(1) + 1e-3 * kl).mean()


# ---------------------------------------------------------------------------
# Logit adjustment (Menon et al., 2021)
# ---------------------------------------------------------------------------
class LogitAdjusted(nn.Module):
    """Wraps a model: adds tau * log(prior) to the logits during training only."""

    def __init__(self, base, label_freq, tau=1.0):
        super().__init__()
        self.base = base
        p = torch.as_tensor(label_freq, dtype=torch.float).clamp(1e-6, 1 - 1e-6)
        self.register_buffer("adj", tau * torch.log(p / (1 - p)))

    def forward(self, *b):
        return self.base(*b)

    def loss(self, b, y):
        return F.binary_cross_entropy_with_logits(self.base(*b) + self.adj, y)
