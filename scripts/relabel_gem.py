"""Rebuild the Notebook 1 artifacts with the corrected ICD-10 -> ICD-9 mapping.

Cohort, visit order, splits, history lengths k, procedures and medications are kept exactly as in the
original `processed/` directory; only the diagnosis codes (inputs) and the targets are recomputed:
combination GEM entries map to the union of their targets, lists of alternatives to the target most
frequently coded in the ICD-9 era (see Notebook 9). The label vocabulary and the frequency strata are then
rebuilt with the Notebook 1 procedure (patient-level counts on the training samples, no threshold).

    python scripts/relabel_gem.py --processed processed --hosp data/hosp --out processed_gemfix
"""
import argparse
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rarepred.resources import (gem_targets_corrected, load_multilevel_ccs_from_hcup_csv,  # noqa: E402
                                load_multilevel_ccs_from_rda)

ap = argparse.ArgumentParser()
ap.add_argument("--processed", default=str(ROOT / "processed"))
ap.add_argument("--hosp", default=str(ROOT / "data" / "hosp"))
ap.add_argument("--external", default=str(ROOT / "data" / "external"))
ap.add_argument("--hcup-csv", default=str(ROOT / "data" / "mappings" / "Multi_Level_CCS_2015" / "ccs_multi_dx_tool_2015.csv"))
ap.add_argument("--out", default=str(ROOT / "processed_gemfix"))
a = ap.parse_args()
src, out = Path(a.processed), Path(a.out)
out.mkdir(parents=True, exist_ok=True)

# --- mappings -----------------------------------------------------------------------------------------
if Path(a.hcup_csv).exists():
    ccs, _ = load_multilevel_ccs_from_hcup_csv(a.hcup_csv)
else:
    ccs, _ = load_multilevel_ccs_from_rda(a.external)
icd9_to_ccs = dict(zip(ccs.icd9, ccs.ccs_ext))


def to_ccs(icd9):
    icd9 = str(icd9).strip().replace(".", "")
    return icd9_to_ccs.get(icd9) or icd9_to_ccs.get(icd9.zfill(5))


dx = pd.read_csv(Path(a.hosp) / "diagnoses_icd.csv.gz", usecols=["hadm_id", "icd_code", "icd_version"],
                 dtype={"icd_code": str})
dx["icd_code"] = dx.icd_code.str.strip()
freq9 = dx[dx.icd_version == 9].icd_code.value_counts().to_dict()
corr = gem_targets_corrected(Path(a.external) / "2018_I10gem.txt", freq9)

dx["targets"] = [[c] if v == 9 else corr.get(c, []) for c, v in zip(dx.icd_code, dx.icd_version)]
dx = dx.explode("targets").dropna(subset=["targets"])
dx["ccs"] = dx.targets.map(to_ccs)
ccs_by_hadm = dx.dropna(subset=["ccs"]).groupby("hadm_id").ccs.apply(lambda s: sorted(set(s))).to_dict()

# --- sequences ----------------------------------------------------------------------------------------
with open(src / "patient_sequences.pkl", "rb") as f:
    seqs = pickle.load(f)
changed_visits = 0
new_seqs = {}
for pid, visits in seqs.items():
    nv = []
    for hadm, d, p, r in visits:
        nd = ccs_by_hadm.get(hadm, [])
        changed_visits += set(nd) != set(d)
        nv.append((hadm, nd, p, r))
    new_seqs[pid] = nv


def relabel(samples):
    outs, dropped = [], 0
    for s in samples:
        lab = ccs_by_hadm.get(s["hadm_id"], [])
        if not lab:
            dropped += 1
            continue
        outs.append({**s, "label": list(lab)})
    return outs, dropped


splits = {}
for name in ("train", "val", "test"):
    with open(src / f"{name}_samples.pkl", "rb") as f:
        splits[name], dropped = relabel(pickle.load(f))
    print(f"{name}: {len(splits[name]):,} samples ({dropped} dropped: no CCS target)")

# --- vocabulary and strata, Notebook 1 procedure ------------------------------------------------------
per_pid = {}
for s in splits["train"]:
    per_pid.setdefault(s["patient_id"], set()).update(s["label"])
cnt = Counter(c for labs in per_pid.values() for c in labs)
vocab = sorted(cnt)
code2idx = {c: i for i, c in enumerate(vocab)}
freqs = np.array([cnt[c] for c in vocab])
q25, q75 = np.percentile(freqs, [25, 75])
for name in splits:
    splits[name] = [{**s, "label": [c for c in s["label"] if c in code2idx]} for s in splits[name]]
n_lab = [len(s["label"]) for s in splits["train"]]
with open(src / "meta_ccs.json") as f:
    old = json.load(f)
meta = {"vocab": vocab, "code2idx": code2idx, "N": len(vocab), "q25": float(q25), "q75": float(q75),
        "rare_idx": [i for i, f in enumerate(freqs) if f <= q25],
        "medium_idx": [i for i, f in enumerate(freqs) if q25 < f <= q75],
        "common_idx": [i for i, f in enumerate(freqs) if f > q75],
        "m_median": int(np.median(n_lab)), "m_90": int(np.percentile(n_lab, 90)),
        "n_train_pts": old.get("n_train_pts"), "n_val_pts": old.get("n_val_pts"), "n_test_pts": old.get("n_test_pts"),
        "label_space": "CCS (GEM corrected)"}
with open(out / "meta_ccs.json", "w") as f:
    json.dump(meta, f, indent=2)
with open(out / "patient_sequences.pkl", "wb") as f:
    pickle.dump(new_seqs, f)
for name, S in splits.items():
    with open(out / f"{name}_samples.pkl", "wb") as f:
        pickle.dump(S, f)
print(f"labels {old['N']} -> {meta['N']} | visits whose diagnoses changed: {changed_visits:,} | "
      f"rare stratum <= {q25:.0f} patients (was {old['q25']:.0f})")
print("written", out)
