"""Backbone variants built around the Notebook 2 architecture.

``MultiModalBackbone`` with its default arguments (``fusion='concat'``,
``pooling='sum'``, random embeddings, linear readout) is *exactly* the
Notebook 2 ``SharedBackbone``: same parameter names, same initialisation, same
forward pass. This is checked in the notebooks by copying a ``SharedBackbone``
state dict into it and comparing logits.

Variants (one factor at a time):

* ``pooling='set'``   : attention pooling over the codes of a visit (a
  Set-Transformer block with one pooling seed) instead of a sum.
* ``fusion='sum'``    : h = h_dx + h_rx + h_px instead of concatenation.
* ``fusion='cross'``  : one extra attention layer over the visit states of the
  three modalities, so that a medication can attend to the diagnoses of the
  same patient before the readout.
* ``KnowledgeEmbedding``: code embeddings initialised from frozen text
  embeddings of code descriptions (LLM / RAG), z_i = a W e_i + delta_i, with an
  optional frequency-dependent penalty on delta_i.
* ``readout='knowledge'``: logit_r = w_r . h + (U e_r) . h, where e_r is a frozen
  text embedding of label r and U is shared by all labels, so that rare labels
  borrow a classifier direction learned on frequent ones.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

try:
    from pyhealth.models import TransformerLayer
except Exception:  # pragma: no cover - pyhealth is required for real runs
    TransformerLayer = None


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
def random_embedding(n, d):
    emb = nn.Embedding(n, d, padding_idx=0)
    nn.init.normal_(emb.weight, mean=0.0, std=0.01)  # Notebook 2 / MedCo scale
    emb.weight.data[0].fill_(0)
    return emb


class KnowledgeEmbedding(nn.Module):
    """z_i = scale * W e_i + delta_i  (row 0 = PAD stays exactly zero).

    E      : (n, e) frozen text embeddings, row 0 must correspond to PAD.
    counts : (n,) training frequency of each code, for the shrinkage penalty
             lambda_i = 1 / (1 + n_i) (multiplied by lambda_0 in the loss).
    """

    def __init__(self, E: torch.Tensor, d: int, scale: float = 0.01, counts=None):
        super().__init__()
        E = E.float().clone()
        mu, sd = E[1:].mean(0, keepdim=True), E[1:].std(0, keepdim=True) + 1e-6
        E[1:] = (E[1:] - mu) / sd                 # standardise each dimension
        E[0] = 0.0
        self.register_buffer("E", E)
        self.proj = nn.Linear(E.shape[1], d, bias=False)
        nn.init.normal_(self.proj.weight, std=1.0 / math.sqrt(E.shape[1]))
        self.scale = scale
        self.delta = nn.Embedding(E.shape[0], d, padding_idx=0)
        nn.init.zeros_(self.delta.weight)
        if counts is None:
            counts = torch.zeros(E.shape[0])
        self.register_buffer("shrink", 1.0 / (1.0 + counts.float()))

    @property
    def weight(self):  # full embedding matrix, for analyses
        return self.scale * self.proj(self.E) + self.delta.weight

    def forward(self, idx):
        return self.scale * self.proj(self.E[idx]) + self.delta(idx)

    def penalty(self):
        return (self.shrink[1:, None] * self.delta.weight[1:] ** 2).sum(1).mean()


# ---------------------------------------------------------------------------
# Visit pooling
# ---------------------------------------------------------------------------
class SetPool(nn.Module):
    """One self-attention block over the codes of a visit, then attention
    pooling with a learned seed (Set Transformer: SAB + PMA, Lee et al. 2019)."""

    def __init__(self, d, heads=4, dropout=0.1):
        super().__init__()
        self.sab = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.ln2 = nn.LayerNorm(d)
        self.seed = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.pma = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)

    def forward(self, emb, code_mask):
        B, T, L, d = emb.shape
        x = emb.reshape(B * T, L, d)
        m = code_mask.reshape(B * T, L) > 0
        has = m.any(1)
        kpm = ~m
        kpm[~has, 0] = False                      # empty visit: attend to PAD (zero) slot
        h = self.ln1(x + self.sab(x, x, x, key_padding_mask=kpm, need_weights=False)[0])
        h = self.ln2(h + self.ff(h))
        q = self.seed.expand(B * T, 1, d)
        out = self.pma(q, h, h, key_padding_mask=kpm, need_weights=False)[0].squeeze(1)
        out = out * has.unsqueeze(-1).float()
        # keep the scale of Notebook 2's summed visit vectors (sum of ~n codes)
        n = m.sum(1, keepdim=True).clamp(min=1).float()
        return (out * n).reshape(B, T, d)


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------
class MultiModalBackbone(nn.Module):
    def __init__(self, C_dx, C_rx, C_px, D_out, d=256, n_heads=1, n_layers=1, dropout=0.5,
                 fusion="concat", pooling="sum", emb_dx=None, emb_rx=None, emb_px=None,
                 readout="linear", label_E=None):
        super().__init__()
        assert TransformerLayer is not None, "pyhealth is required"
        assert fusion in ("concat", "sum", "cross") and pooling in ("sum", "set")
        self.fusion, self.pooling, self.readout_kind = fusion, pooling, readout

        self.emb_dx = emb_dx if emb_dx is not None else random_embedding(C_dx, d)
        self.emb_rx = emb_rx if emb_rx is not None else random_embedding(C_rx, d)
        self.emb_px = emb_px if emb_px is not None else random_embedding(C_px, d)

        self.transformer_dx = TransformerLayer(d, heads=n_heads, dropout=dropout, num_layers=n_layers)
        self.transformer_rx = TransformerLayer(d, heads=n_heads, dropout=dropout, num_layers=n_layers)
        self.transformer_px = TransformerLayer(d, heads=n_heads, dropout=dropout, num_layers=n_layers)

        if pooling == "set":
            self.pool_dx, self.pool_rx, self.pool_px = SetPool(d), SetPool(d), SetPool(d)
        if fusion == "cross":
            self.modality = nn.Parameter(torch.zeros(3, d))
            self.cross = nn.TransformerEncoderLayer(d, nhead=max(1, n_heads), dim_feedforward=4 * d,
                                                    dropout=0.1, batch_first=True, norm_first=True)

        h_dim = d if fusion == "sum" else 3 * d
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(h_dim, D_out)

        if readout == "knowledge":
            assert label_E is not None
            E = label_E.float()
            E = (E - E.mean(0, keepdim=True)) / (E.std(0, keepdim=True) + 1e-6)
            E = E / math.sqrt(E.shape[1])           # unit-scale rows: keeps U's Adam steps comparable to W's
            self.register_buffer("label_E", E)
            self.label_proj = nn.Linear(E.shape[1], h_dim, bias=False)
            nn.init.zeros_(self.label_proj.weight)   # starts as the linear readout

    # -- encoders -----------------------------------------------------------
    def _visits(self, emb_layer, pool, idx, cmask):
        emb = emb_layer(idx)                                    # (B, T, L, d)
        if self.pooling == "sum":
            return (emb * cmask.unsqueeze(-1)).sum(dim=2)       # Notebook 2
        return pool(emb, cmask)

    def _encode(self, emb_layer, transformer, pool, idx, cmask, pad):
        v = self._visits(emb_layer, pool, idx, cmask)
        seq, cls = transformer(v, (~pad).long())                # PyHealth: cls = position 0
        return seq, cls

    def patient_representation(self, dx_idx, dx_mask, dx_pad, dx_len, rx_idx, rx_mask, rx_pad, rx_len,
                               px_idx, px_mask, px_pad, px_len):
        pools = (getattr(self, "pool_dx", None), getattr(self, "pool_rx", None), getattr(self, "pool_px", None))
        s_dx, c_dx = self._encode(self.emb_dx, self.transformer_dx, pools[0], dx_idx, dx_mask, dx_pad)
        s_rx, c_rx = self._encode(self.emb_rx, self.transformer_rx, pools[1], rx_idx, rx_mask, rx_pad)
        s_px, c_px = self._encode(self.emb_px, self.transformer_px, pools[2], px_idx, px_mask, px_pad)
        if self.fusion == "concat":
            return torch.cat([c_dx, c_rx, c_px], dim=1)
        if self.fusion == "sum":
            return c_dx + c_rx + c_px
        # cross-modality attention over all visit states
        toks = torch.cat([s_dx + self.modality[0], s_rx + self.modality[1], s_px + self.modality[2]], dim=1)
        kpm = torch.cat([dx_pad, rx_pad, px_pad], dim=1)
        out = self.cross(toks, src_key_padding_mask=kpm)
        T1, T2 = s_dx.shape[1], s_rx.shape[1]
        return torch.cat([out[:, 0], out[:, T1], out[:, T1 + T2]], dim=1)

    def forward(self, *batch12):
        h = self.dropout(self.patient_representation(*batch12))
        logits = self.fc(h)
        if self.readout_kind == "knowledge":
            logits = logits + h @ self.label_proj(self.label_E).T
        return logits

    def penalty(self):
        p = 0.0
        for e in (self.emb_dx, self.emb_rx, self.emb_px):
            if isinstance(e, KnowledgeEmbedding):
                p = p + e.penalty()
        return p


class SharedBackbone(nn.Module):
    """Verbatim copy of the Notebook 2 model (reference for equivalence checks)."""

    def __init__(self, C_dx, C_rx, C_px, D_out, d=256, n_heads=1, n_layers=1, dropout=0.5):
        super().__init__()
        self.emb_dx = nn.Embedding(C_dx, d, padding_idx=0)
        self.emb_rx = nn.Embedding(C_rx, d, padding_idx=0)
        self.emb_px = nn.Embedding(C_px, d, padding_idx=0)
        for emb in [self.emb_dx, self.emb_rx, self.emb_px]:
            nn.init.normal_(emb.weight, mean=0.0, std=0.01)
            emb.weight.data[0].fill_(0)
        self.transformer_dx = TransformerLayer(d, heads=n_heads, dropout=dropout, num_layers=n_layers)
        self.transformer_rx = TransformerLayer(d, heads=n_heads, dropout=dropout, num_layers=n_layers)
        self.transformer_px = TransformerLayer(d, heads=n_heads, dropout=dropout, num_layers=n_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(3 * d, D_out)

    def _encode_modality(self, emb_layer, transformer, code_idx, code_mask, pad_mask, lengths):
        emb = emb_layer(code_idx)
        v = (emb * code_mask.unsqueeze(-1)).sum(dim=2)
        _, x = transformer(v, (~pad_mask).long())
        return x

    def forward(self, dx_idx, dx_mask, dx_pad, dx_len, rx_idx, rx_mask, rx_pad, rx_len,
                px_idx, px_mask, px_pad, px_len):
        x = torch.cat([self._encode_modality(self.emb_dx, self.transformer_dx, dx_idx, dx_mask, dx_pad, dx_len),
                       self._encode_modality(self.emb_rx, self.transformer_rx, rx_idx, rx_mask, rx_pad, rx_len),
                       self._encode_modality(self.emb_px, self.transformer_px, px_idx, px_mask, px_pad, px_len)],
                      dim=1)
        return self.fc(self.dropout(x))
