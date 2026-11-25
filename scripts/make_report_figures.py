"""Figures of the report built from measured results only (results/*.json, figures/*.png)."""
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({
    "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#666666",
    "xtick.color": "#666666", "ytick.color": "#666666", "axes.grid": True, "grid.color": "#e6e6e6",
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "savefig.dpi": 300,
})
BLUE, ORANGE = "#257de0", "#f47230"

R = json.load(open(ROOT / "results" / "measured_results.json"))

# Figure 1: Notebook 2 validation curves
h = R["backbone_nb2"]["val_history"]
ep = [x["epoch"] for x in h]
fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.3))
ax[0].plot(ep, [x["val_sample_auprc"] for x in h], color=BLUE, lw=2, marker="o", ms=3.5)
ax[0].axvline(R["backbone_nb2"]["best_epoch"], color="#9c9c9c", lw=1, ls="--")
ax[0].set_title("(a) validation sample-AUPRC", loc="left")
ax[1].plot(ep, [x["val_macro_label_auprc"] for x in h], color=ORANGE, lw=2, marker="o", ms=3.5)
ax[1].axvline(R["backbone_nb2"]["best_epoch"], color="#9c9c9c", lw=1, ls="--")
ax[1].set_title("(b) validation macro label-AUPRC", loc="left")
for a in ax:
    a.set_xlabel("epoch")
    a.set_xticks(ep)
plt.tight_layout()
plt.savefig(OUT / "nb2_validation_curves.pdf", bbox_inches="tight")

# Figure 2: rare diseases per CCS category (produced by Notebook 4)
shutil.copy(ROOT / "figures" / "rare_diseases_per_ccs.png", OUT / "rare_diseases_per_ccs.png")
print("figures written to", OUT)
