"""Evaluation metrics.

Two families of AUPRC are reported and must not be confused:

* **sample-AUPRC** (MedCo / PyHealth convention, Notebook 2): average precision
  computed *within each test sample* over the label set, then averaged over
  samples. The quartile version restricts the label set to one frequency
  quartile and keeps samples with at least one positive in it.
* **label (macro) AUPRC**: average precision computed *per label* over test
  samples, then averaged over labels of a stratum (Hypothesis C, per-code
  analyses). For rare labels this is a much harder and noisier quantity.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score


def sample_auprc(probs, ys):
    keep = ys.sum(axis=1) > 0
    if keep.sum() == 0:
        return float("nan")
    return float(average_precision_score(ys[keep], probs[keep], average="samples"))


def group_sample_auprc(probs, ys, idx):
    p, y = probs[:, idx], ys[:, idx]
    keep = y.sum(axis=1) > 0
    if keep.sum() == 0:
        return float("nan"), 0
    return float(average_precision_score(y[keep], p[keep], average="samples")), int(keep.sum())


def per_label_ap(probs, ys):
    """AP for every label (NaN when the label has no positive test sample)."""
    out = np.full(ys.shape[1], np.nan)
    for j in range(ys.shape[1]):
        if ys[:, j].sum() > 0:
            out[j] = average_precision_score(ys[:, j], probs[:, j])
    return out


def acc_at_m(probs, ys, m):
    order = np.argsort(-probs, axis=1)[:, :m]
    hits = denom = 0.0
    for i in range(probs.shape[0]):
        t = np.flatnonzero(ys[i])
        if t.size == 0:
            continue
        hits += np.intersect1d(t, order[i]).size
        denom += t.size
    return hits / denom if denom else float("nan")


def evaluate(probs, ys, groups, m_values=(11, 20)):
    """Full evaluation used by all new notebooks."""
    res = {"sample_auprc": sample_auprc(probs, ys)}
    ap = per_label_ap(probs, ys)
    res["macro_label_auprc"] = float(np.nanmean(ap))
    for name, idx in groups.items():
        s, n = group_sample_auprc(probs, ys, idx)
        res[f"sample_auprc[{name}]"] = s
        res[f"n_samples[{name}]"] = n
        res[f"label_auprc[{name}]"] = float(np.nanmean(ap[idx])) if np.isfinite(ap[idx]).any() else float("nan")
        res[f"n_labels_evaluable[{name}]"] = int(np.isfinite(ap[idx]).sum())
    names = list(groups)
    res["gap_sample_auprc"] = res[f"sample_auprc[{names[-1]}]"] - res[f"sample_auprc[{names[0]}]"]
    res["gap_label_auprc"] = res[f"label_auprc[{names[-1]}]"] - res[f"label_auprc[{names[0]}]"]
    for m in m_values:
        res[f"acc@{m}"] = acc_at_m(probs, ys, m)
    return res, ap


def bootstrap_ci(stat_fn, n, n_boot=1000, seed=0, alpha=0.05):
    """Percentile bootstrap over test *samples* (rows); stat_fn(idx) -> float."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        v = stat_fn(idx)
        if np.isfinite(v):
            vals.append(v)
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def paired_bootstrap_delta(probs_a, probs_b, ys, idx_labels=None, n_boot=1000, seed=0):
    """CI for sample-AUPRC(b) - sample-AUPRC(a) on the same test samples."""
    if idx_labels is not None:
        probs_a, probs_b, ys = probs_a[:, idx_labels], probs_b[:, idx_labels], ys[:, idx_labels]
    keep = ys.sum(axis=1) > 0
    pa, pb, y = probs_a[keep], probs_b[keep], ys[keep]

    def per_sample_ap(p):
        return np.array([average_precision_score(y[i], p[i]) for i in range(len(y))])

    a, b = per_sample_ap(pa), per_sample_ap(pb)
    d = b - a
    rng = np.random.default_rng(seed)
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)
