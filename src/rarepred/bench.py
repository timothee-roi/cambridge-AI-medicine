"""Benchmark: registry of models, training, evaluation (including adaptive depth).

Usage (one model, one seed; see scripts/benchmark.py and scripts/benchmark.sbatch):
    python scripts/benchmark.py --model modelloop_L2T2_adaptive --seed 42
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import (build_input_vocabs, label_frequency_groups, load_artifacts, make_loader, precompute,
                   to_device)
from .metrics import evaluate, sample_auprc
from .models import KnowledgeEmbedding, MultiModalBackbone
from .train import set_seed
from .zoo import (BagOfCodes, DepthStack, GRAMEmbedding, LabelPrior, LogitAdjusted, RETAIN, VisitGRU)

# name -> (family, short description, block applications per modality)
MODELS = {
    "label_prior":              ("baseline", "training label frequencies", 0),
    "bag_of_codes":             ("baseline", "code counts of the history + MLP", 0),
    "gru":                      ("recurrent", "GRU over sum-pooled visits", 0),
    "retain":                   ("recurrent", "RETAIN", 0),
    "transformer":              ("transformer", "MedCo backbone (1 layer per modality, concat)", 1),
    "transformer_pos_last":     ("transformer", "+ visit positions, last-visit readout", 1),
    "transformer_setpool":      ("transformer", "+ attention pooling within visits", 1),
    "transformer_cross":        ("transformer", "+ cross-modality attention", 1),
    "transformer_logitadj":     ("transformer", "+ logit-adjusted loss", 1),
    "gram":                     ("knowledge", "GRAM ontology embeddings (CCS, ATC)", 1),
    "knowledge_rag_readout":    ("knowledge", "RAG-initialised embeddings + knowledge readout", 1),
    "deep_L2":                  ("deep", "2 distinct layers", 2),
    "deep_L4":                  ("deep", "4 distinct layers", 4),
    "loop_T4":                  ("looped", "1 layer applied 4 times", 4),
    "layerloop_L2T2":           ("looped", "layer loop: L1 L1 L2 L2", 4),
    "modelloop_L2T2":           ("looped", "model loop: L1 L2 L1 L2", 4),
    "deep_L4_adaptive":         ("adaptive", "4 distinct layers, exit after any layer", 4),
    "loop_T4_adaptive":         ("adaptive", "loop T4, geometric depth prior, post-hoc exit", 4),
    "loop_T4_gate":             ("adaptive", "loop T4, PonderNet gate", 4),
    "layerloop_L2T2_adaptive":  ("adaptive", "layer loop, geometric depth prior, post-hoc exit", 4),
    "modelloop_L2T2_adaptive":  ("adaptive", "model loop, geometric depth prior, post-hoc exit", 4),
}

_STACKS = {
    "transformer_pos_last": dict(n_blocks=1, schedule=(0,), positional=True, readout_position="last"),
    "deep_L2": dict(n_blocks=2, schedule=(0, 1)),
    "deep_L4": dict(n_blocks=4, schedule=(0, 1, 2, 3)),
    "loop_T4": dict(n_blocks=1, schedule=(0, 0, 0, 0)),
    "layerloop_L2T2": dict(n_blocks=2, schedule=(0, 0, 1, 1)),
    "modelloop_L2T2": dict(n_blocks=2, schedule=(0, 1, 0, 1)),
    "deep_L4_adaptive": dict(n_blocks=4, schedule=(0, 1, 2, 3), supervision="geometric"),
    "loop_T4_adaptive": dict(n_blocks=1, schedule=(0, 0, 0, 0), supervision="geometric"),
    "loop_T4_gate": dict(n_blocks=1, schedule=(0, 0, 0, 0), supervision="geometric", gate=True),
    "layerloop_L2T2_adaptive": dict(n_blocks=2, schedule=(0, 0, 1, 1), supervision="geometric"),
    "modelloop_L2T2_adaptive": dict(n_blocks=2, schedule=(0, 1, 0, 1), supervision="geometric"),
}


class Data:
    def __init__(self, processed, batch_size=64, max_train=None, knowledge_npz=None):
        meta, tr, va, te, seqs = load_artifacts(processed)
        if max_train:
            tr = tr[:max_train]
        self.meta, self.train_samples, self.test_samples, self.seqs = meta, tr, te, seqs
        self.vocab, self.code2idx, self.D_out = meta["vocab"], meta["code2idx"], meta["N"]
        self.dx2idx, self.rx2idx, self.px2idx = build_input_vocabs(seqs)
        self.C = (len(self.dx2idx) + 1, len(self.rx2idx) + 1, len(self.px2idx) + 1)
        a = (seqs, self.code2idx, self.dx2idx, self.rx2idx, self.px2idx, self.D_out)
        self.train_loader = make_loader(precompute(tr, *a), batch_size, shuffle=True)
        self.val_loader = make_loader(precompute(va, *a), batch_size)
        self.test_loader = make_loader(precompute(te, *a), batch_size)
        self.groups, self.freq, self.bounds = label_frequency_groups(tr, self.vocab, "sample")
        self.groups_mass, _, _ = label_frequency_groups(tr, self.vocab, "mass")
        self.groups_patient = {"rare": meta["rare_idx"], "medium": meta["medium_idx"], "common": meta["common_idx"]}
        self.prior = np.clip(self.freq / max(1, len(tr)), 1e-6, 1 - 1e-6)
        self.knowledge_npz = knowledge_npz


def build(name, data: Data):
    C_dx, C_rx, C_px = data.C
    N = data.D_out
    if name == "label_prior":
        return LabelPrior(data.prior)
    if name == "bag_of_codes":
        return BagOfCodes(C_dx, C_rx, C_px, N)
    if name == "gru":
        return VisitGRU(C_dx, C_rx, C_px, N)
    if name == "retain":
        return RETAIN(C_dx, C_rx, C_px, N)
    if name == "transformer":
        return MultiModalBackbone(C_dx, C_rx, C_px, N)
    if name == "transformer_setpool":
        return MultiModalBackbone(C_dx, C_rx, C_px, N, pooling="set")
    if name == "transformer_cross":
        return MultiModalBackbone(C_dx, C_rx, C_px, N, fusion="cross")
    if name == "transformer_logitadj":
        return LogitAdjusted(MultiModalBackbone(C_dx, C_rx, C_px, N), data.prior)
    if name == "gram":
        return MultiModalBackbone(C_dx, C_rx, C_px, N,
                                  emb_dx=GRAMEmbedding(data.dx2idx, 256, "ccs"),
                                  emb_rx=GRAMEmbedding(data.rx2idx, 256, "atc"))
    if name == "knowledge_rag_readout":
        k = dict(np.load(data.knowledge_npz, allow_pickle=True))
        embs = {}
        for m, vocab in (("dx", data.dx2idx), ("rx", data.rx2idx), ("px", data.px2idx)):
            src = k[f"{m}_rag"] if f"{m}_rag" in k else k[f"{m}_desc"]
            look = dict(zip(list(k[f"{m}_codes"]), src))
            E = np.zeros((len(vocab) + 1, src.shape[1]), np.float32)
            for c, i in vocab.items():
                if c in look:
                    E[i] = look[c]
            embs[f"emb_{m}"] = KnowledgeEmbedding(torch.tensor(E), 256)
        look = dict(zip(list(k["dx_codes"]), k["dx_rag"]))
        label_E = torch.tensor(np.stack([look.get(c, np.zeros(k["dx_rag"].shape[1], np.float32)) for c in data.vocab]))
        return MultiModalBackbone(C_dx, C_rx, C_px, N, readout="knowledge", label_E=label_E, **embs)
    if name in _STACKS:
        return DepthStack(C_dx, C_rx, C_px, N, **_STACKS[name])
    raise KeyError(name)


def _loss(model, batch):
    b, y = batch[:12], batch[12]
    if hasattr(model, "loss"):
        return model.loss(b, y)
    return F.binary_cross_entropy_with_logits(model(*b), y)


@torch.no_grad()
def predict_all(model, loader, device):
    """Probabilities: (n, N) or (n, D, N) for multi-exit models."""
    model.eval()
    P, Y = [], []
    for batch in loader:
        batch = to_device(batch, device)
        P.append(torch.sigmoid(model(*batch[:12])).float().cpu())
        Y.append(batch[12].cpu())
    return torch.cat(P).numpy(), torch.cat(Y).numpy()


def _score(probs, ys):
    if probs.ndim == 3:
        per = [sample_auprc(probs[:, t], ys) for t in range(probs.shape[1])]
        return float(np.nanmax(per)), per
    return sample_auprc(probs, ys), None


def train(model, data: Data, device, ckpt: Path, max_epochs=50, patience=5, lr=1e-3, log=print, tag=""):
    if ckpt.exists():
        st = torch.load(ckpt, map_location=device)
        model.load_state_dict(st["model"])
        if st.get("complete"):
            log(f"[{tag}] loaded complete checkpoint")
            return model, st["history"]
        log(f"[{tag}] resuming from epoch {len(st['history'])}")
        history, best, best_state = st["history"], st["best"], st["model"]
        wait = st.get("wait", 0)
    else:
        history, best, best_state, wait = [], -1.0, None, 0
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:                                   # nothing to train (label prior is fixed)
        return model, history
    opt = torch.optim.Adam(params, lr=lr)
    for epoch in range(len(history) + 1, max_epochs + 1):
        if wait >= patience:
            break
        t0 = time.time()
        model.train()
        tot, n = 0.0, 0
        for batch in data.train_loader:
            batch = to_device(batch, device)
            loss = _loss(model, batch)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            tot += loss.item() * batch[12].size(0)
            n += batch[12].size(0)
        probs, ys = predict_all(model, data.val_loader, device)
        val, per = _score(probs, ys)
        history.append({"epoch": epoch, "train_loss": tot / n, "val_sample_auprc": val,
                        "val_by_depth": per, "time_s": time.time() - t0})
        log(f"[{tag}] epoch {epoch:>2} | {history[-1]['time_s'] / 60:.1f} min | loss {tot / n:.4f} | val {val:.4f}"
            + (" | by depth " + " ".join(f"{x:.4f}" for x in per) if per else ""))
        if val > best:
            best, wait = val, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        torch.save({"model": best_state, "history": history, "best": best, "wait": wait, "complete": False}, ckpt)
    model.load_state_dict(best_state)
    torch.save({"model": best_state, "history": history, "best": best, "complete": True}, ckpt)
    return model, history


# ---------------------------------------------------------------------------
# Adaptive exit (post hoc), following Popescu et al. (2026)
# ---------------------------------------------------------------------------
def _entropy_at_m(P, m):
    """P: (n, D, N) -> (n, D) mean binary entropy of the top-m probabilities at each depth."""
    top = np.sort(P, axis=2)[:, :, -m:]
    top = np.clip(top, 1e-7, 1 - 1e-7)
    return -(top * np.log(top) + (1 - top) * np.log(1 - top)).mean(2)


def _kl_consecutive(P):
    Pc = np.clip(P, 1e-7, 1 - 1e-7)
    kl = np.zeros(P.shape[:2])
    kl[:, 0] = np.inf
    a, b = Pc[:, 1:], Pc[:, :-1]
    kl[:, 1:] = (a * np.log(a / b) + (1 - a) * np.log((1 - a) / (1 - b))).mean(2)
    return kl


def _exit_depth(score, tau):
    """First depth whose score is <= tau (else the last)."""
    ok = score <= tau
    d = np.where(ok.any(1), ok.argmax(1), score.shape[1] - 1)
    return d


def adaptive_exit(Pv, Yv, Pt, Yt, m, target=0.99):
    """Choose, on validation, the most aggressive threshold that keeps `target` x full-depth sample-AUPRC."""
    full_v = sample_auprc(Pv[:, -1], Yv)
    out = {}
    for rule, fn in (("entropy@m", lambda P: _entropy_at_m(P, m)), ("kl_consecutive", _kl_consecutive)):
        sv, st = fn(Pv), fn(Pt)
        finite = sv[np.isfinite(sv)]
        best_tau, best_depth = -np.inf, Pv.shape[1]
        for tau in np.quantile(finite, np.linspace(0, 1, 41)):
            d = _exit_depth(sv, tau)
            if sample_auprc(Pv[np.arange(len(d)), d], Yv) >= target * full_v and d.mean() < best_depth:
                best_tau, best_depth = tau, d.mean()
        d_t = _exit_depth(st, best_tau)
        out[rule] = {"tau": float(best_tau), "depths": d_t, "probs": Pt[np.arange(len(d_t)), d_t]}
    return out


def run(name, processed, out_dir, ckpt_dir, seed=42, device=None, max_epochs=50, max_train=None,
        knowledge_npz=None, tag_suffix=""):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)
    data = Data(processed, max_train=max_train, knowledge_npz=knowledge_npz)
    set_seed(seed)
    model = build(name, data).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tag = f"{name}{tag_suffix}_s{seed}"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    model, history = train(model, data, device, Path(ckpt_dir) / f"bench_{tag}.pt", max_epochs=max_epochs, tag=tag)
    train_time = time.time() - t0
    t1 = time.time()
    Pt, Yt = predict_all(model, data.test_loader, device)
    infer_time = time.time() - t1
    ms = (data.meta["m_median"], data.meta["m_90"])
    res = {"model": name, "family": MODELS[name][0], "description": MODELS[name][1],
           "block_applications": MODELS[name][2], "seed": seed, "n_params": n_params,
           "train_time_s": train_time, "test_inference_s": infer_time, "history": history,
           "n_train": len(data.train_samples), "n_test": len(data.test_samples)}
    q1 = np.zeros(len(Yt), bool)
    q1[Yt[:, data.groups["Q1 (rarest)"]].sum(1) > 0] = True
    if Pt.ndim == 3:
        Pv, Yv = predict_all(model, data.val_loader, device)
        res["test_by_depth"] = [evaluate(Pt[:, t], Yt, data.groups, ms)[0] for t in range(Pt.shape[1])]
        res["val_by_depth"] = [sample_auprc(Pv[:, t], Yv) for t in range(Pv.shape[1])]
        ad = adaptive_exit(Pv, Yv, Pt, Yt, data.meta["m_median"])
        res["adaptive"] = {}
        for rule, a in ad.items():
            r, _ = evaluate(a["probs"], Yt, data.groups, ms)
            d = a["depths"] + 1
            res["adaptive"][rule] = {"tau": a["tau"], "mean_depth": float(d.mean()),
                                     "mean_depth_q1_samples": float(d[q1].mean()) if q1.any() else None,
                                     "mean_depth_other_samples": float(d[~q1].mean()),
                                     "compute_saving": float(1 - d.mean() / Pt.shape[1]), "test": r}
        P_final = Pt[:, -1]
    else:
        P_final = Pt
    res["test"], ap = evaluate(P_final, Yt, data.groups, ms)
    res["test_patient_strata"], _ = evaluate(P_final, Yt, data.groups_patient, ms)
    res["test_medco_groups"], _ = evaluate(P_final, Yt, data.groups_mass, ms)
    res["per_label_ap"] = [None if not np.isfinite(x) else float(x) for x in ap]
    with open(Path(out_dir) / f"{tag}.json", "w") as f:
        json.dump(res, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    np.savez_compressed(Path(ckpt_dir) / f"bench_preds_{tag}.npz", probs=P_final.astype(np.float16))
    return res
