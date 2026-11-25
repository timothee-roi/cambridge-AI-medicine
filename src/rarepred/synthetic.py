"""Synthetic artifacts with the exact format of Notebook 1's outputs.

Used only to test notebooks end to end without MIMIC-IV access. The generator
plants simple, known structure (latent patient profiles, persistent chronic
codes, a few rare codes triggered by a diagnosis + medication conjunction) so
that a correct pipeline learns something; the numbers it produces mean nothing
clinically and must never be reported.
"""
from __future__ import annotations

import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np


def make_synthetic(out_dir, dx_codes=None, n_patients=3000, n_rx=120, n_px=60, n_profiles=12, seed=0):
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if dx_codes is None:
        dx_codes = [f"{a}.{b}" for a in range(1, 19) for b in range(1, 13)]
    dx_codes = list(dx_codes)
    rx_codes = [f"R{i:03d}" for i in range(n_rx)]
    px_codes = [str(i) for i in range(1, n_px + 1)]
    nd = len(dx_codes)

    # Zipf-like base rates so that a long tail of rare labels exists
    base = 1.0 / np.arange(1, nd + 1) ** 1.1
    base = base[rng.permutation(nd)]
    prof = rng.dirichlet(np.full(nd, 0.05), size=n_profiles) * 0.6 + base / base.sum() * 0.4
    rx_prof = rng.dirichlet(np.full(n_rx, 0.1), size=n_profiles)
    # rare "trigger" codes: appear after a (dx, rx) conjunction
    tail = np.argsort(base)[:12]
    triggers = [(int(rng.integers(nd)), int(rng.integers(n_rx)), int(t)) for t in tail]

    seqs = {}
    for p in range(n_patients):
        pid = 10_000_000 + p
        z = rng.integers(n_profiles)
        K = int(rng.integers(2, 9))
        chronic = set(rng.choice(nd, size=rng.integers(1, 4), replace=False, p=prof[z]).tolist())
        visits = []
        prev_dx, prev_rx = set(), set()
        for t in range(K):
            dx = set(rng.choice(nd, size=rng.integers(2, 9), replace=False, p=prof[z]).tolist()) | chronic
            for d0, r0, tgt in triggers:
                if d0 in prev_dx and r0 in prev_rx and rng.random() < 0.7:
                    dx.add(tgt)
            rx = set(rng.choice(n_rx, size=rng.integers(1, 10), replace=False, p=rx_prof[z]).tolist())
            if rng.random() < 0.6:
                d0, r0, _ = triggers[rng.integers(len(triggers))]
                if rng.random() < 0.3:
                    dx.add(d0); rx.add(r0)
            px = set(rng.choice(n_px, size=rng.integers(0, 3), replace=False).tolist())
            visits.append((30_000_000 + p * 100 + t,
                           [dx_codes[i] for i in sorted(dx)],
                           [px_codes[i] for i in sorted(px)],
                           [rx_codes[i] for i in sorted(rx)]))
            prev_dx, prev_rx = dx, rx
        seqs[pid] = visits

    pids = list(seqs)
    rng.shuffle(pids)
    n_tr, n_va = int(0.7 * len(pids)), int(0.15 * len(pids))
    split = {"train": set(pids[:n_tr]), "val": set(pids[n_tr:n_tr + n_va]), "test": set(pids[n_tr + n_va:])}

    def build(name):
        out_s = []
        for pid in split[name]:
            K = len(seqs[pid])
            ks = list(range(1, K)) if name == "train" else [K // 2]
            for k in ks:
                lab = sorted(set(seqs[pid][k][1]))
                if lab:
                    out_s.append({"patient_id": pid, "hadm_id": seqs[pid][k][0], "k": k,
                                  "history_hadm_ids": [v[0] for v in seqs[pid][:k]], "label": lab})
        return out_s

    tr, va, te = build("train"), build("val"), build("test")
    per_pid = {}
    for s in tr:
        per_pid.setdefault(s["patient_id"], set()).update(s["label"])
    cnt = Counter(c for labs in per_pid.values() for c in labs)
    vocab = sorted(cnt)
    code2idx = {c: i for i, c in enumerate(vocab)}
    freqs = np.array([cnt[c] for c in vocab])
    q25, q75 = np.percentile(freqs, [25, 75])
    filt = lambda S: [{**s, "label": [c for c in s["label"] if c in code2idx]} for s in S]
    tr, va, te = filt(tr), filt(va), filt(te)
    n_lab = [len(s["label"]) for s in tr]
    meta = {"vocab": vocab, "code2idx": code2idx, "N": len(vocab), "q25": float(q25), "q75": float(q75),
            "rare_idx": [i for i, f in enumerate(freqs) if f <= q25],
            "medium_idx": [i for i, f in enumerate(freqs) if q25 < f <= q75],
            "common_idx": [i for i, f in enumerate(freqs) if f > q75],
            "m_median": int(np.median(n_lab)), "m_90": int(np.percentile(n_lab, 90)),
            "label_space": "CCS", "synthetic": True}
    json.dump(meta, open(out / "meta_ccs.json", "w"))
    for name, obj in [("train_samples.pkl", tr), ("val_samples.pkl", va), ("test_samples.pkl", te),
                      ("patient_sequences.pkl", seqs)]:
        with open(out / name, "wb") as f:
            pickle.dump(obj, f)
    return meta, len(tr), len(va), len(te)
