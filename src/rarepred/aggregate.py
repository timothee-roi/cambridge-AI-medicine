"""Aggregate benchmark JSONs (mean +/- sd over seeds) and write the LaTeX tables of the report."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .bench import MODELS

MAIN_COLS = [
    ("sample_auprc", "all"),
    ("sample_auprc[Q1 (rarest)]", "Q1"),
    ("sample_auprc[Q2]", "Q2"),
    ("sample_auprc[Q3]", "Q3"),
    ("sample_auprc[Q4 (common)]", "Q4"),
    ("gap_sample_auprc", "Q4$-$Q1"),
    ("label_auprc[Q1 (rarest)]", "label Q1"),
    ("label_auprc[Q4 (common)]", "label Q4"),
]


def load_runs(result_dir) -> pd.DataFrame:
    rows = []
    for f in sorted(Path(result_dir).glob("*.json")):
        r = json.load(open(f))
        row = {"model": r["model"], "seed": r["seed"], "family": r["family"], "params": r["n_params"],
               "applications": r["block_applications"], "epochs": len(r["history"]),
               "train_h": r["train_time_s"] / 3600}
        for k, v in r["test"].items():
            row[k] = v
        row["medco_Q1"] = r["test_medco_groups"]["sample_auprc[Q1 (rarest)]"]
        for rule, a in (r.get("adaptive") or {}).items():
            row[f"{rule}:mean_depth"] = a["mean_depth"]
            row[f"{rule}:depth_q1"] = a["mean_depth_q1_samples"]
            row[f"{rule}:depth_other"] = a["mean_depth_other_samples"]
            row[f"{rule}:sample_auprc"] = a["test"]["sample_auprc"]
            row[f"{rule}:Q1"] = a["test"]["sample_auprc[Q1 (rarest)]"]
        if r.get("test_by_depth"):
            for t, d in enumerate(r["test_by_depth"]):
                row[f"depth{t + 1}:sample_auprc"] = d["sample_auprc"]
                row[f"depth{t + 1}:Q1"] = d["sample_auprc[Q1 (rarest)]"]
        rows.append(row)
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    order = [m for m in MODELS if m in set(df.model)]
    num = df.drop(columns=["family"]).groupby("model")
    mean, sd, n = num.mean(numeric_only=True), num.std(numeric_only=True), num.size()
    out = mean.add_suffix(":mean").join(sd.add_suffix(":sd")).join(n.rename("n_seeds"))
    return out.loc[order]


def _fmt(m, s, n):
    if m is None or not np.isfinite(m):
        return "--"
    return f"{m:.3f}" if (n < 2 or not np.isfinite(s)) else f"{m:.3f}\\,{{\\scriptsize$\\pm${s:.3f}}}"


def latex_main(summary: pd.DataFrame, path):
    lines = [r"\begin{tabular}{ll" + "r" * len(MAIN_COLS) + "}", r"\toprule",
             "family & model & " + " & ".join(c for _, c in MAIN_COLS) + r"\\", r"\midrule"]
    fam_prev = None
    for name, row in summary.iterrows():
        fam = MODELS[name][0]
        if fam_prev is not None and fam != fam_prev:
            lines.append(r"\midrule")
        cells = [_fmt(row.get(f"{k}:mean"), row.get(f"{k}:sd"), row["n_seeds"]) for k, _ in MAIN_COLS]
        lines.append(f"{fam if fam != fam_prev else ''} & \\texttt{{{name.replace('_', chr(92) + '_')}}} & "
                     + " & ".join(cells) + r"\\")
        fam_prev = fam
    lines += [r"\bottomrule", r"\end{tabular}"]
    Path(path).write_text("\n".join(lines) + "\n")


def latex_depth(summary: pd.DataFrame, path):
    names = [n for n in summary.index if MODELS[n][0] in ("deep", "looped", "adaptive")] + \
            [n for n in ("transformer",) if n in summary.index]
    lines = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
             r"model & params & appl. & all & Q1 & adaptive all & adaptive Q1 & mean exit (Q1 / other)\\", r"\midrule"]
    for n in names:
        r = summary.loc[n]
        g = lambda k: r.get(f"{k}:mean", np.nan)
        exit_ = "--"
        if np.isfinite(g("entropy@m:mean_depth")):
            exit_ = f"{g('entropy@m:mean_depth'):.2f} ({g('entropy@m:depth_q1'):.2f} / {g('entropy@m:depth_other'):.2f})"
        lines.append(f"\\texttt{{{n.replace('_', chr(92) + '_')}}} & {g('params') / 1e6:.2f}M & {int(g('applications'))} & "
                     f"{_fmt(g('sample_auprc'), np.nan, 1)} & {_fmt(g('sample_auprc[Q1 (rarest)]'), np.nan, 1)} & "
                     f"{_fmt(g('entropy@m:sample_auprc'), np.nan, 1)} & {_fmt(g('entropy@m:Q1'), np.nan, 1)} & {exit_}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    Path(path).write_text("\n".join(lines) + "\n")
