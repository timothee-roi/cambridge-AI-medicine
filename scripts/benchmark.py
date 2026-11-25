"""Train and evaluate one benchmark model.

    python scripts/benchmark.py --model modelloop_L2T2_adaptive --seed 42 \
        --processed processed_gemfix --out results/benchmark/gemfix
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rarepred.bench import MODELS, run  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, choices=list(MODELS) + ["list"])
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--processed", default=str(ROOT / "processed_gemfix"))
ap.add_argument("--out", default=str(ROOT / "results" / "benchmark" / "gemfix"))
ap.add_argument("--checkpoints", default=str(ROOT / "checkpoints" / "benchmark"))
ap.add_argument("--max-epochs", type=int, default=50)
ap.add_argument("--max-train", type=int, default=None, help="subsample training samples (tests only)")
ap.add_argument("--knowledge", default=None, help="knowledge_embeddings_*.npz from Notebook 5")
ap.add_argument("--tag", default="")
a = ap.parse_args()
if a.model == "list":
    print("\n".join(MODELS))
    sys.exit()
if a.model == "knowledge_rag_readout" and not a.knowledge:
    a.knowledge = str(Path(a.processed) / "knowledge_embeddings_BioLORD-2023.npz")
r = run(a.model, a.processed, a.out, Path(a.checkpoints) / Path(a.out).name, seed=a.seed,
        max_epochs=a.max_epochs, max_train=a.max_train, knowledge_npz=a.knowledge, tag_suffix=a.tag)
t = r["test"]
print(f"{a.model} seed {a.seed}: sample-AUPRC {t['sample_auprc']:.4f} | Q1 {t['sample_auprc[Q1 (rarest)]']:.4f} | "
      f"Q4 {t['sample_auprc[Q4 (common)]']:.4f} | label-AUPRC Q1 {t['label_auprc[Q1 (rarest)]']:.4f}")
