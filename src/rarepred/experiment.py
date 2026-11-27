"""One function to train and evaluate a backbone configuration.

A configuration is a plain dict, e.g.
    {'fusion': 'concat', 'pooling': 'sum', 'emb': 'random', 'readout': 'linear', 'lambda0': 0.0}
with emb in {'random', 'desc', 'rag'} and readout in {'linear', 'knowledge'}.
'desc' / 'rag' require the knowledge embeddings written by Notebook 5.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .data import (build_input_vocabs, input_code_counts, label_frequency_groups, load_artifacts,
                   make_loader, precompute, to_device)
from .metrics import evaluate
from .models import KnowledgeEmbedding, MultiModalBackbone, SharedBackbone
from .train import predict, set_seed, train_model


class Setup:
    """Everything that does not depend on the configuration (loaded once)."""

    def __init__(self, processed, device, batch_size=64, knowledge_npz=None, max_train=None):
        self.device = device
        meta, tr, va, te, seqs = load_artifacts(processed)
        if max_train:
            tr = tr[:max_train]
        self.meta, self.train_samples, self.test_samples, self.seqs = meta, tr, te, seqs
        self.code2idx, self.D_out, self.vocab = meta["code2idx"], meta["N"], meta["vocab"]
        self.dx2idx, self.rx2idx, self.px2idx = build_input_vocabs(seqs)
        self.C = (len(self.dx2idx) + 1, len(self.rx2idx) + 1, len(self.px2idx) + 1)
        args = (seqs, self.code2idx, self.dx2idx, self.rx2idx, self.px2idx, self.D_out)
        self.train_loader = make_loader(precompute(tr, *args), batch_size, shuffle=True)
        self.val_loader = make_loader(precompute(va, *args), batch_size)
        self.test_loader = make_loader(precompute(te, *args), batch_size)
        self.groups_sample, self.freq_sample, self.bounds_sample = label_frequency_groups(tr, self.vocab, "sample")
        self.groups_mass, _, _ = label_frequency_groups(tr, self.vocab, "mass")
        self.groups_patient = {"rare": meta["rare_idx"], "medium": meta["medium_idx"], "common": meta["common_idx"]}
        self.counts = input_code_counts(tr, seqs)
        self.knowledge = None
        if knowledge_npz is not None and Path(knowledge_npz).exists():
            self.knowledge = dict(np.load(knowledge_npz, allow_pickle=True))

    # -- knowledge matrices -------------------------------------------------
    def _matrix(self, modality, source):
        """(C, e) matrix in input-vocabulary order, row 0 = PAD."""
        k = self.knowledge
        codes = list(k[f"{modality}_codes"])
        E_src = k[f"{modality}_{source}"] if f"{modality}_{source}" in k else k[f"{modality}_desc"]
        lookup = dict(zip(codes, E_src))
        vocab = {"dx": self.dx2idx, "rx": self.rx2idx, "px": self.px2idx}[modality]
        E = np.zeros((len(vocab) + 1, E_src.shape[1]), dtype=np.float32)
        miss = 0
        for c, i in vocab.items():
            if c in lookup:
                E[i] = lookup[c]
            else:
                miss += 1
        cnt = torch.zeros(len(vocab) + 1)
        for c, i in vocab.items():
            cnt[i] = self.counts[modality].get(c, 0)
        return torch.tensor(E), cnt, miss

    def label_matrix(self, source):
        k = self.knowledge
        lookup = dict(zip(list(k["dx_codes"]), k[f"dx_{source}"]))
        dim = k[f"dx_{source}"].shape[1]
        return torch.tensor(np.stack([lookup.get(c, np.zeros(dim, np.float32)) for c in self.vocab]))

    # -- models -------------------------------------------------------------
    def build(self, cfg):
        C_dx, C_rx, C_px = self.C
        embs = {}
        if cfg.get("emb", "random") != "random":
            assert self.knowledge is not None, "knowledge embeddings missing: run Notebook 5"
            for m in ("dx", "rx", "px"):
                E, cnt, miss = self._matrix(m, cfg["emb"])
                embs[f"emb_{m}"] = KnowledgeEmbedding(E, 256, counts=cnt)
                if miss:
                    print(f"  {m}: {miss} codes without a text embedding (kept at zero + delta)")
        label_E = None
        if cfg.get("readout", "linear") == "knowledge":
            label_E = self.label_matrix(cfg.get("label_source", "rag"))
        return MultiModalBackbone(C_dx, C_rx, C_px, self.D_out, d=256, n_heads=1, n_layers=1, dropout=0.5,
                                  fusion=cfg.get("fusion", "concat"), pooling=cfg.get("pooling", "sum"),
                                  readout=cfg.get("readout", "linear"), label_E=label_E, **embs).to(self.device)

    def equivalence_check(self):
        """Default MultiModalBackbone == Notebook 2 SharedBackbone (eval mode)."""
        C_dx, C_rx, C_px = self.C
        ref = SharedBackbone(C_dx, C_rx, C_px, self.D_out).to(self.device).eval()
        new = MultiModalBackbone(C_dx, C_rx, C_px, self.D_out).to(self.device).eval()
        new.load_state_dict(ref.state_dict())
        batch = to_device(next(iter(self.val_loader)), self.device)
        with torch.no_grad():
            diff = (ref(*batch[:12]) - new(*batch[:12])).abs().max().item()
        return diff


def run_config(setup: Setup, name, cfg, checkpoints, results_dir, max_epochs=50, seed=42, retrain=False):
    set_seed(seed)
    model = setup.build(cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{name}] trainable parameters: {n_params:,}")
    model, history = train_model(model, setup.train_loader, setup.val_loader, setup.device,
                                 Path(checkpoints) / f"{name}_seed{seed}.pt", max_epochs=max_epochs,
                                 penalty_weight=cfg.get("lambda0", 0.0), retrain=retrain, tag=name,
                                 meta={"cfg": cfg, "seed": seed})
    probs, ys = predict(model, setup.test_loader, setup.device)
    ms = (setup.meta["m_median"], setup.meta["m_90"])
    res_s, ap = evaluate(probs, ys, setup.groups_sample, ms)
    res_p, _ = evaluate(probs, ys, setup.groups_patient, ms)
    res_m, _ = evaluate(probs, ys, setup.groups_mass, ms)
    out = {"name": name, "cfg": cfg, "seed": seed, "n_params": n_params, "history": history,
           "test_sample_quartiles": res_s, "test_patient_strata": res_p,
           "test_medco_mass_groups": res_m, "medco_group_sizes": {k: len(v) for k, v in setup.groups_mass.items()}}
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(Path(checkpoints) / f"preds_{name}_seed{seed}.npz", probs=probs.astype(np.float16))
    with open(Path(results_dir) / f"run_{name}_seed{seed}.json", "w") as f:
        json.dump(out, f, indent=2)
    return out, probs, ys, model
