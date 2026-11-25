"""Training loop shared by the new notebooks (same protocol as Notebook 2).

Adam, lr 1e-3, BCE on all labels, gradient clipping at 1.0, early stopping on
validation sample-AUPRC with patience 5. The best state is written to disk at
every improvement so that a job killed at its wall-time limit still leaves
the best model; ``complete=True`` is written when training ends normally.
"""
from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .data import to_device
from .metrics import sample_auprc


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    P, Y = [], []
    for batch in loader:
        batch = to_device(batch, device)
        P.append(torch.sigmoid(model(*batch[:12])).cpu())
        Y.append(batch[12].cpu())
    return torch.cat(P).numpy(), torch.cat(Y).numpy()


def train_model(model, train_loader, val_loader, device, ckpt_path, max_epochs=50, patience=5,
                lr=1e-3, penalty_weight=0.0, retrain=False, log=print, tag="model", meta=None):
    ckpt_path = Path(ckpt_path)
    if ckpt_path.exists() and not retrain:
        state = torch.load(ckpt_path, map_location=device)
        if state.get("meta") == meta:
            model.load_state_dict(state["model"])
            flag = "" if state.get("complete") else "  (WARNING: interrupted run, best model so far)"
            log(f"[{tag}] loaded {ckpt_path.name}{flag}")
            return model, state["history"]
        log(f"[{tag}] {ckpt_path.name} was written with a different configuration: retraining")

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, best_state, wait, history = -1.0, None, 0, []
    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        model.train()
        tot = torch.zeros((), device=device)
        n = 0
        for batch in train_loader:
            batch = to_device(batch, device)
            logits = model(*batch[:12])
            loss = nn.functional.binary_cross_entropy_with_logits(logits, batch[12])
            if penalty_weight:
                loss = loss + penalty_weight * model.penalty()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.detach() * batch[12].size(0)
            n += batch[12].size(0)
        probs, ys = predict(model, val_loader, device)
        val = sample_auprc(probs, ys)
        history.append({"epoch": epoch, "train_loss": (tot / n).item(), "val_sample_auprc": val,
                        "time_s": time.time() - t0})
        log(f"[{tag}] epoch {epoch:>2} | {history[-1]['time_s'] / 60:.1f} min | "
            f"loss {history[-1]['train_loss']:.4f} | val sample-AUPRC {val:.4f}")
        if val > best:
            best, wait = val, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save({"model": best_state, "history": history, "complete": False, "meta": meta}, ckpt_path)
        else:
            wait += 1
            if wait >= patience:
                log(f"[{tag}] early stop at epoch {epoch} (best {best:.4f})")
                break
    model.load_state_dict(best_state)
    torch.save({"model": best_state, "history": history, "complete": True, "meta": meta}, ckpt_path)
    return model, history
