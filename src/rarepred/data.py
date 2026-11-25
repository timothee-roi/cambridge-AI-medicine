"""Loading the Notebook 1 artifacts and building batches.

The batching code reproduces Notebook 2 exactly (three modality streams, each a
(B, T, L) tensor of code indices with index 0 reserved for padding), so that
models trained through this module are comparable with the Notebook 2 backbone.
"""
from __future__ import annotations

import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------
def load_artifacts(processed: str | Path):
    processed = Path(processed)
    with open(processed / "meta_ccs.json") as f:
        meta = json.load(f)

    def _pkl(name):
        with open(processed / name, "rb") as f:
            return pickle.load(f)

    return (meta, _pkl("train_samples.pkl"), _pkl("val_samples.pkl"),
            _pkl("test_samples.pkl"), _pkl("patient_sequences.pkl"))


def build_input_vocabs(patient_sequences):
    """Three separate input vocabularies; index 0 = PAD (Notebook 2, cell 3)."""
    dx, rx, px = set(), set(), set()
    for visits in patient_sequences.values():
        for _, d, p, r in visits:
            dx.update(d); px.update(p); rx.update(r)
    to_idx = lambda vocab: {c: i + 1 for i, c in enumerate(sorted(vocab))}
    return to_idx(dx), to_idx(rx), to_idx(px)


def input_code_counts(train_samples, patient_sequences):
    """Number of training samples whose history contains each input code.

    Used for frequency-dependent shrinkage of knowledge-initialised embeddings.
    Only the training split is read.
    """
    cnt = {"dx": Counter(), "rx": Counter(), "px": Counter()}
    for s in train_samples:
        pid, k = s["patient_id"], s["k"]
        # count each code once per (patient, history length) sample
        dx, rx, px = set(), set(), set()
        for _, d, p, r in patient_sequences[pid][:k]:
            dx.update(d); px.update(p); rx.update(r)
        cnt["dx"].update(dx); cnt["rx"].update(rx); cnt["px"].update(px)
    return cnt


# ---------------------------------------------------------------------------
# Frequency groups
# ---------------------------------------------------------------------------
def label_frequency_groups(train_samples, vocab, level="sample"):
    """Quartile groups of the label vocabulary by training frequency.

    level='sample'  : count of training *samples* containing the label
                      (Notebook 2 test cell; Q25 = 200 on the full data)
    level='patient' : count of training *patients* (Notebook 1 meta; Q25 = 166)
    level='mass'    : MedCo's grouping by cumulative share of occurrences (sample counts)
    Returns (groups: dict name -> list of label indices, freqs, boundaries).
    """
    cnt = Counter()
    if level == "sample":
        for s in train_samples:
            cnt.update(set(s["label"]))
    elif level == "mass":
        for s in train_samples:
            cnt.update(set(s["label"]))
    else:
        per_pid = {}
        for s in train_samples:
            per_pid.setdefault(s["patient_id"], set()).update(s["label"])
        for labs in per_pid.values():
            cnt.update(labs)
    freqs = np.array([cnt[c] for c in vocab], dtype=float)
    if level == "mass":
        # MedCo (utils/eval_test.py, get_group_labels1): labels sorted by frequency, a label belongs to the
        # group in which the cumulative share of all label occurrences falls. Groups therefore hold 25% of
        # the occurrences each, not 25% of the labels. (MedCo counts on all samples; we count on train.)
        order = np.argsort(freqs, kind="stable")
        cum = np.cumsum(freqs[order]) / freqs.sum()
        names = ["Q1 (rarest)", "Q2", "Q3", "Q4 (common)"]
        groups = {n: [] for n in names}
        for i, c in zip(order, cum):
            groups[names[0 if c <= .25 else 1 if c <= .5 else 2 if c <= .75 else 3]].append(int(i))
        return groups, freqs, None
    q25, q50, q75 = np.percentile(freqs, [25, 50, 75])
    groups = {
        "Q1 (rarest)": [i for i, f in enumerate(freqs) if f <= q25],
        "Q2": [i for i, f in enumerate(freqs) if q25 < f <= q50],
        "Q3": [i for i, f in enumerate(freqs) if q50 < f <= q75],
        "Q4 (common)": [i for i, f in enumerate(freqs) if f > q75],
    }
    return groups, freqs, (q25, q50, q75)


# ---------------------------------------------------------------------------
# Tensors
# ---------------------------------------------------------------------------
def precompute(samples, patient_sequences, code2idx, dx2idx, rx2idx, px2idx, D_out):
    out = []
    for s in samples:
        pid, k = s["patient_id"], s["k"]
        dx_seq, rx_seq, px_seq = [], [], []
        for _, dx, px, rx in patient_sequences[pid][:k]:
            dx_seq.append([dx2idx[c] for c in dx if c in dx2idx] or [0])
            rx_seq.append([rx2idx[c] for c in rx if c in rx2idx] or [0])
            px_seq.append([px2idx[c] for c in px if c in px2idx] or [0])
        y = torch.zeros(D_out)
        for c in s["label"]:
            if c in code2idx:
                y[code2idx[c]] = 1.0
        out.append((dx_seq, rx_seq, px_seq, y))
    return out


def pad_modality(seqs_batch, pad_idx=0):
    B = len(seqs_batch)
    T = max(len(s) for s in seqs_batch)
    L = max((len(v) for s in seqs_batch for v in s), default=1)
    code_idx = torch.zeros(B, T, L, dtype=torch.long)
    code_mask = torch.zeros(B, T, L, dtype=torch.float)
    pad_mask = torch.ones(B, T, dtype=torch.bool)
    lengths = torch.zeros(B, dtype=torch.long)
    for b, seq in enumerate(seqs_batch):
        lengths[b] = len(seq) - 1
        for t, visit in enumerate(seq):
            pad_mask[b, t] = False
            for l, code in enumerate(visit):
                code_idx[b, t, l] = code
                if code != pad_idx:
                    code_mask[b, t, l] = 1.0
    return code_idx, code_mask, pad_mask, lengths


def collate(batch):
    dx, rx, px, ys = zip(*batch)
    return (*pad_modality(dx), *pad_modality(rx), *pad_modality(px), torch.stack(ys))


class _ListDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


def make_loader(data, batch_size=64, shuffle=False, seed=42):
    g = torch.Generator().manual_seed(seed)
    return DataLoader(_ListDataset(data), batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate, num_workers=0, generator=g)


def to_device(batch, device):
    return [x.to(device) for x in batch]
