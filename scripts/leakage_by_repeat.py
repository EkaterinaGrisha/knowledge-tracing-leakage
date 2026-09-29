"""Paper C1, step S4: does the inflation really come from the repeated-concept rows?

The claim we have to test, not assume: when a question carrying several knowledge
components is unrolled into several rows that all share one answer, the model
sees that answer on the first row and can copy it on the following ones.

If that is the mechanism, then scoring the concept-level predictions separately
by position must show it directly:
  * `is_repeat = 0` rows  — first row of a question, the answer has not been shown yet;
  * `is_repeat = 1` rows  — continuation rows, the answer of this same question is
                            already in the context.
AUC on the continuation rows should be much higher, and should grow with how many
rows of the same question came before.

Row alignment follows the rule used throughout this work: each row
of `test_sequences.csv` contributes ``sum(selectmask == 1) - 1`` predictions,
because pyKT shifts every sequence by one position. The script refuses to score a
cell unless the derived total matches the saved `concept_y_true` exactly, so an
alignment error cannot pass silently.

The test split is identical across the five folds (pyKT holds out `fold = -1`),
so `test_sequences.csv` is parsed once per dataset and reused.

Outputs (artifacts/cross_dataset/):
  leakage_by_repeat.csv          one row per (dataset, model, fold, position class)
  leakage_by_repeat_summary.csv  one row per (dataset, model): 5-fold means
  leakage_kc_per_question.csv    components per question, counted on the very rows the
                                 metrics use, so the paper's own tables divide to it

Usage:
  python -m scripts.leakage_by_repeat
  python -m scripts.leakage_by_repeat --datasets ednet assist2009
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ktx import paths

CROSS_DIR = paths.ARTIFACTS_DIR / "cross_dataset"
PRED_DIR = paths.ARTIFACTS_DIR / "predictions"
DEEP_MODELS = ("dkt", "sakt", "akt", "simplekt")


def parse_test_sequences(dataset: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (is_repeat, repeat_index) aligned with `concept_y_true`.

    `repeat_index` counts how many rows of the same question precede this one:
    0 on a first row, 1 on the first continuation, 2 on the second, and so on.
    It is computed over the full valid sequence and only then shifted, so the
    first prediction of a chunk still gets its true index.
    """
    path = paths.PYKT_ROOT / "data" / dataset / "test_sequences.csv"
    if not path.exists():
        raise SystemExit(f"нет файла {path}")
    header = pd.read_csv(path, nrows=0).columns
    if "is_repeat" not in header:
        # A concept-only dataset has no question ids, so pyKT emits no is_repeat.
        # Treating every row as a first row is not an assumption here: such a dataset
        # carries one component per row by construction. We verify that against
        # data_config rather than trusting the file name.
        cfg = json.loads(paths.DATA_CONFIG_JSON.read_text())
        max_c = cfg.get(dataset, {}).get("max_concepts")
        if max_c != 1:
            raise SystemExit(
                f"{dataset}: нет колонки is_repeat, но max_concepts={max_c} — развёртка возможна, "
                "и восстановить позиции нельзя")
        tseq = pd.read_csv(path, usecols=["selectmasks"])
        tseq["is_repeat"] = tseq["selectmasks"].map(
            lambda sm: ",".join("0" for _ in str(sm).split(",")))
    else:
        tseq = pd.read_csv(path, usecols=["selectmasks", "is_repeat"])

    rep_parts: list[np.ndarray] = []
    idx_parts: list[np.ndarray] = []
    for sm_raw, rep_raw in zip(tseq["selectmasks"], tseq["is_repeat"]):
        sm = np.fromstring(str(sm_raw), sep=",", dtype=np.int64)
        rep = np.fromstring(str(rep_raw), sep=",", dtype=np.int64)
        valid = sm == 1
        rep_valid = rep[: len(sm)][valid]
        if rep_valid.size < 2:
            continue
        # running index within a question: reset on every is_repeat == 0
        idx = np.empty_like(rep_valid)
        run = 0
        for i, r in enumerate(rep_valid):
            run = run + 1 if r == 1 else 0
            idx[i] = run
        rep_parts.append(rep_valid[1:])   # pyKT shift: drop the first valid position
        idx_parts.append(idx[1:])
    return np.concatenate(rep_parts), np.concatenate(idx_parts)


def safe_auc(y: np.ndarray, p: np.ndarray) -> float | str:
    if y.size == 0 or np.unique(y).size < 2:
        return ""
    return round(float(roc_auc_score(y, p)), 4)


def score_cell(npz_path: Path, is_repeat: np.ndarray, repeat_idx: np.ndarray) -> list[dict]:
    d = np.load(npz_path, allow_pickle=True)
    if "concept_y_true" not in d.files:
        return []
    y = np.asarray(d["concept_y_true"]).astype(int)
    p = np.asarray(d["concept_y_prob"], dtype=float)
    if y.size != is_repeat.size:
        raise SystemExit(
            f"{npz_path}: alignment failed — concept_y_true has {y.size} entries, "
            f"test_sequences.csv derives {is_repeat.size}"
        )

    groups: list[tuple[str, np.ndarray]] = [
        ("all", np.ones_like(is_repeat, dtype=bool)),
        ("first_row", is_repeat == 0),
        ("repeat_row", is_repeat == 1),
        ("repeat_1", repeat_idx == 1),
        ("repeat_2", repeat_idx == 2),
        ("repeat_3plus", repeat_idx >= 3),
    ]
    out = []
    for name, mask in groups:
        yy, pp = y[mask], p[mask]
        # Median predicted probability split by the true label. On continuation rows
        # these two numbers separate to the ends of the scale, which shows the copying
        # at the level of the probabilities themselves rather than through a ranking
        # metric: an AUC of 1.000 alone would not rule out a degenerate output.
        med1 = float(np.median(pp[yy == 1])) if (yy == 1).any() else float("nan")
        med0 = float(np.median(pp[yy == 0])) if (yy == 0).any() else float("nan")
        out.append({
            "position_class": name,
            "n": int(yy.size),
            "auc": safe_auc(yy, pp),
            "mean_prob": round(float(pp.mean()), 4) if pp.size else "",
            "base_rate": round(float(yy.mean()), 4) if yy.size else "",
            # how often the most likely label is the right one, at threshold 0.5
            "accuracy": round(float(((pp >= 0.5).astype(int) == yy).mean()), 4) if yy.size else "",
            "median_prob_y1": round(med1, 4) if med1 == med1 else "",
            "median_prob_y0": round(med0, 4) if med0 == med0 else "",
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--out-dir", type=Path, default=CROSS_DIR)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    datasets = args.datasets or sorted(p.name for p in PRED_DIR.iterdir() if p.is_dir())
    rows: list[dict] = []
    kcq_rows: list[dict] = []
    for dataset in datasets:
        try:
            is_repeat, repeat_idx = parse_test_sequences(dataset)
        except SystemExit as exc:
            print(f"{dataset}: skipped — {exc}")
            continue
        share = float((is_repeat == 1).mean())
        n_questions = int((is_repeat == 0).sum())
        kcq_rows.append({
            "dataset": dataset,
            "n_component_rows": int(is_repeat.size),
            "n_questions": n_questions,
            # components per question on exactly the rows the metrics are computed on
            "kc_per_question": round(is_repeat.size / n_questions, 4),
        })
        print(f"{dataset:20} предсказаний {is_repeat.size:>8}  доля строк-продолжений {share:.3f}"
              f"  компонентов на задание {is_repeat.size / n_questions:.4f}")
        for model in DEEP_MODELS:
            for fold in range(5):
                npz_path = PRED_DIR / dataset / f"{model}_fold{fold}.npz"
                if not npz_path.exists():
                    continue
                for rec in score_cell(npz_path, is_repeat, repeat_idx):
                    rows.append({"dataset": dataset, "model": model, "fold": fold, **rec})

    kcq_path = args.out_dir / "leakage_kc_per_question.csv"
    with kcq_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["dataset", "n_component_rows", "n_questions",
                                           "kc_per_question"])
        w.writeheader()
        w.writerows(kcq_rows)

    matrix_path = args.out_dir / "leakage_by_repeat.csv"
    fields = ["dataset", "model", "fold", "position_class", "n", "auc",
              "mean_prob", "base_rate", "accuracy", "median_prob_y1", "median_prob_y0"]
    with matrix_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # 5-fold means per (dataset, model, position_class)
    acc: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    ns: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for r in rows:
        if r["auc"] == "":
            continue
        key = (r["dataset"], r["model"], r["position_class"])
        acc[key].append(float(r["auc"]))
        ns[key].append(int(r["n"]))
    summary = []
    for (dataset, model, klass), vals in sorted(acc.items()):
        summary.append({
            "dataset": dataset, "model": model, "position_class": klass,
            "n_folds": len(vals),
            "n_mean": int(np.mean(ns[(dataset, model, klass)])),
            "auc_mean": round(float(np.mean(vals)), 4),
            "auc_std": round(float(np.std(vals, ddof=1)), 4) if len(vals) > 1 else "",
        })
    summary_path = args.out_dir / "leakage_by_repeat_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    print(f"\nrows written : {len(rows)} -> {matrix_path.name}")
    print(f"summary rows : {len(summary)} -> {summary_path.name}")
    print(f"kc per question: {len(kcq_rows)} -> {kcq_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
