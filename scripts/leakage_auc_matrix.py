"""Paper C1, step S2: how much AUC each inference path buys on the same checkpoint.

For every deep cell (dataset x model x fold) we already have three probability
vectors saved side by side in the prediction npz. They come from the *same*
trained model and differ only in how question-level probabilities are obtained:

  pykt_question   `y_prob`               pyKT `evaluate_question`: the cq/cshft input
                                         never feeds the concept expansion of the
                                         current question back into the context.
  late_mean       `test_y_prob_q_ours`   our own aggregation: mean of the concept-row
                                         probabilities belonging to one question.
                                         On `is_repeat=1` rows the model has already
                                         seen the label of that same question.
  concept_raw     `concept_y_prob`       no aggregation at all, scored on the
                                         unrolled concept rows.

The script scores all three with AUC and writes a long table plus a 5-fold
summary. Nothing is trained and nothing is re-run through a model: this is a
read-only pass over artifacts that already exist on disk.

Anchor (verified by hand 2026-09-02, ednet / simplekt / fold 0):
  pykt_question 0.6585 (n=132209) | late_mean 0.8965 (n=133516) | concept_raw 0.9406 (n=301372)
The script re-checks these three numbers and fails loudly if they move.

Outputs (artifacts/cross_dataset/):
  leakage_auc_matrix.csv    one row per (dataset, model, fold, source)
  leakage_auc_summary.csv   one row per (dataset, model): 5-fold mean/std + deltas

Usage:
  python -m scripts.leakage_auc_matrix
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from ktx import paths

CROSS_DIR = paths.ARTIFACTS_DIR / "cross_dataset"
PRED_DIR = paths.ARTIFACTS_DIR / "predictions"

DEEP_MODELS = ("dkt", "sakt", "akt", "simplekt")

# source name -> (label array, probability array)
SOURCES: dict[str, tuple[str, str]] = {
    "pykt_question": ("y_true", "y_prob"),
    "late_mean": ("test_y_true_q_ours", "test_y_prob_q_ours"),
    "concept_raw": ("concept_y_true", "concept_y_prob"),
}

ANCHORS = {
    ("ednet", "simplekt", 0, "pykt_question"): (0.6585, 132209),
    ("ednet", "simplekt", 0, "late_mean"): (0.8965, 133516),
    ("ednet", "simplekt", 0, "concept_raw"): (0.9406, 301372),
}


def cell_files() -> list[tuple[str, str, int, Path]]:
    """Every seed-42 deep prediction file, as (dataset, model, fold, path)."""
    out: list[tuple[str, str, int, Path]] = []
    for path in sorted(PRED_DIR.glob("*/*_fold[0-4].npz")):
        if "seed43" in path.name:
            continue
        dataset = path.parent.name
        model, fold = path.stem.rsplit("_fold", 1)
        if model not in DEEP_MODELS:
            continue
        out.append((dataset, model, int(fold), path))
    return out


def score_cell(path: Path) -> dict[str, tuple[float, int]]:
    """AUC and n for each source present in this npz."""
    npz = np.load(path, allow_pickle=True)
    present = set(npz.files)
    scored: dict[str, tuple[float, int]] = {}
    for source, (y_key, p_key) in SOURCES.items():
        if y_key not in present or p_key not in present:
            continue
        y_true = np.asarray(npz[y_key]).astype(int)
        y_prob = np.asarray(npz[p_key], dtype=float)
        if len(y_true) != len(y_prob):
            raise SystemExit(
                f"{path}: {source} length mismatch {len(y_true)} vs {len(y_prob)}"
            )
        if len(np.unique(y_true)) < 2:
            continue
        scored[source] = (float(roc_auc_score(y_true, y_prob)), int(len(y_true)))
    return scored


def check_anchors(rows: list[dict]) -> list[str]:
    """Compare the three hand-verified numbers; return a list of failures."""
    got = {(r["dataset"], r["model"], r["fold"], r["source"]): (r["auc"], r["n"]) for r in rows}
    failures = []
    for key, (want_auc, want_n) in ANCHORS.items():
        if key not in got:
            failures.append(f"anchor missing: {key}")
            continue
        auc, n = got[key]
        if abs(auc - want_auc) > 5e-5 or n != want_n:
            failures.append(
                f"anchor moved: {key} expected {want_auc}/{want_n}, got {round(auc, 4)}/{n}"
            )
    return failures


def summarise(rows: list[dict]) -> list[dict]:
    """5-fold mean/std per (dataset, model, source), plus the two deltas."""
    per_source: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for r in rows:
        per_source[(r["dataset"], r["model"], r["source"])].append(r["auc"])

    cells = sorted({(d, m) for d, m, _s in per_source})
    summary = []
    for dataset, model in cells:
        row: dict[str, object] = {"dataset": dataset, "model": model}
        means: dict[str, float] = {}
        for source in SOURCES:
            vals = per_source.get((dataset, model, source), [])
            row[f"{source}_n_folds"] = len(vals)
            if vals:
                means[source] = float(np.mean(vals))
                row[f"{source}_auc_mean"] = round(means[source], 4)
                row[f"{source}_auc_std"] = round(float(np.std(vals, ddof=1)), 4) if len(vals) > 1 else ""
            else:
                row[f"{source}_auc_mean"] = ""
                row[f"{source}_auc_std"] = ""
        row["delta_late_mean_minus_pykt"] = (
            round(means["late_mean"] - means["pykt_question"], 4)
            if {"late_mean", "pykt_question"} <= means.keys() else ""
        )
        row["delta_concept_minus_pykt"] = (
            round(means["concept_raw"] - means["pykt_question"], 4)
            if {"concept_raw", "pykt_question"} <= means.keys() else ""
        )
        summary.append(row)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=CROSS_DIR)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cells = cell_files()
    if not cells:
        raise SystemExit(f"no deep prediction files under {PRED_DIR}")

    rows: list[dict] = []
    gaps: list[str] = []
    for dataset, model, fold, path in cells:
        scored = score_cell(path)
        for source in SOURCES:
            if source not in scored:
                gaps.append(f"{dataset}/{model}/fold{fold}: no {source}")
                continue
            auc, n = scored[source]
            rows.append({
                "dataset": dataset, "model": model, "fold": fold,
                "source": source, "auc": round(auc, 4), "n": n,
                # full precision as well: the control in section 5.2 asks how far from
                # zero the difference is on datasets without unrolling, and four
                # decimals cannot answer that
                "auc_full": f"{auc:.10f}",
            })

    failures = check_anchors(rows)

    matrix_path = args.out_dir / "leakage_auc_matrix.csv"
    with matrix_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["dataset", "model", "fold", "source",
                                           "auc", "auc_full", "n"])
        w.writeheader()
        w.writerows(rows)

    summary = summarise(rows)
    summary_path = args.out_dir / "leakage_auc_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    print(f"cells scanned : {len(cells)}")
    print(f"rows written  : {len(rows)} -> {matrix_path.name}")
    print(f"summary rows  : {len(summary)} -> {summary_path.name}")
    if gaps:
        print(f"\nmissing source arrays ({len(gaps)}):")
        for g in gaps:
            print(f"  - {g}")
    if failures:
        print("\nANCHOR CHECK FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nanchor check: 3/3 ok (ednet/simplekt/fold0)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
