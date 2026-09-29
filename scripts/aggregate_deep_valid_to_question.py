"""Aggregate concept-level valid predictions to question-level.

pyKT stores per-fold deep predictions with `valid_y_true` / `valid_y_prob` at
CONCEPT granularity (the runner's `evaluate_with_preds` on `valid_loader`).
The paired test-side predictions `y_true` / `y_prob` are at QUESTION
granularity (from `evaluate_question` with `late_mean` fusion). For
apples-to-apples calibration we need valid at question granularity
too — otherwise the calibrator is fit on one distribution and applied to
another which is exactly how the calibration failure of the paper arises.

Approach. pyKT's `evaluate_with_preds` returns predictions row-by-row in CSV
order, taking (per row) the positions where `selectmasks == 1`, dropping the
first one due to the pyKT shift (predictions start at position 1). Therefore
we can re-derive the (row, position) address of every element of
`valid_y_prob`, look up the corresponding `question` / `response` /
`is_repeat` from `train_valid_sequences.csv` (filtered to the valid fold),
and aggregate multi-KC concept expansions via `late_mean` (mean of concept
probs).

Grouping rule (matches pyKT `effective_fusion`'s groupby on `qidx`): a new
interaction starts at every `is_repeat=0` position; subsequent `is_repeat=1`
positions are concept-row continuations of the same interaction and get
averaged in. Grouping by "consecutive same question-id" (the earlier v1
logic) is WRONG when a student attempts the same question twice back-to-back
— those are two `is_repeat=0` interactions with same q-id and must stay
separate. Verified on TEST side: with the is_repeat rule, aggregated
probabilities match stored `y_prob` to ±5e-5 (pyKT rounds `late_mean` to 4
decimals) on assist2017 and assist2012 (single-KC only); on multi-KC datasets
a residual ~0.1–1% count delta remains due to chunk-boundary questions that
pyKT merges via its "rest" mechanism across DataLoader batches — this cannot
be reconstructed from the CSV alone, but affects only the group count, not
the aggregated probability distribution, so calibrator fit is unchanged.

Output: `valid_y_true_q` / `valid_y_prob_q` / `valid_groups_q` added to the
existing npz **in place** (safe: extra fields, no overwrite of existing).
Also produces a small summary CSV
`artifacts/cross_dataset/deep_valid_qlvl_summary.csv` recording per-cell
counts for audit.
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from ktx import paths

DEEP = ["dkt", "sakt", "akt", "simplekt"]
DATASETS = ["assist2009", "assist2015", "assist2017", "algebra2005",
            "bridge2algebra2006", "assist2012", "ednet"]
FOLD_RE = re.compile(r"(?P<model>[a-z+]+)_fold(?P<fold>\d+)\.npz$")


def _load_valid_csv(dataset: str) -> pd.DataFrame:
    p = paths.PYKT_ROOT / "data" / dataset / "train_valid_sequences.csv"
    if not p.exists():
        raise FileNotFoundError(str(p))
    return pd.read_csv(p)


def _aggregate_row(sm_str, q_str, r_str, ir_str, uid: int,
                   offset: int, y_true_flat: np.ndarray, y_prob_flat: np.ndarray):
    """Given one CSV row (already selected as valid-fold) and the flat-array
    offset into concept-level valid predictions, return
    (n_concepts_this_row, list_of_(uid, q_id, q_true, q_prob)).

    Grouping is by pyKT's ``is_repeat`` field, not by "consecutive same q-id":
    is_repeat=0 opens a new interaction, is_repeat=1 continues (multi-KC
    concept row of the same interaction).
    """
    sm = [int(x) for x in str(sm_str).split(",")]
    qs = [int(x) for x in str(q_str).split(",")]
    rs = [int(x) for x in str(r_str).split(",")]
    ir = [int(x) for x in str(ir_str).split(",")]

    selected = [i for i, s in enumerate(sm) if s == 1]
    predicted = selected[1:]  # pyKT shift
    n = len(predicted)
    if n == 0:
        return 0, []

    row_probs = y_prob_flat[offset:offset + n]
    row_trues = y_true_flat[offset:offset + n]

    out = []
    i = 0
    while i < n:
        # new interaction starts at is_repeat=0 (guaranteed at position 0 of
        # each row, since chunk boundaries never split a multi-KC group before
        # is_repeat=0); extend while subsequent positions have is_repeat=1.
        j = i + 1
        while j < n and ir[predicted[j]] == 1:
            j += 1
        q_prob = float(row_probs[i:j].mean())
        q_true = int(row_trues[i])   # all responses within an interaction span are equal
        cur_q = qs[predicted[i]]
        out.append((int(uid), int(cur_q), q_true, q_prob))
        i = j
    return n, out


def _process_one(dataset: str, model: str, fold: int, valid_df: pd.DataFrame,
                 force: bool) -> str:
    npz = paths.ARTIFACTS_DIR / "predictions" / dataset / f"{model}_fold{fold}.npz"
    if not npz.exists():
        return "npz missing"
    data = dict(np.load(npz))
    if ("valid_y_true" not in data) or ("valid_y_prob" not in data):
        return "no concept-level valid_y_true/_prob"
    if ("valid_y_true_q" in data) and not force:
        return f"already has valid_y_true_q (size={data['valid_y_true_q'].size}); skip"

    vy = data["valid_y_true"]; vp = data["valid_y_prob"]
    valid_rows = valid_df[valid_df["fold"] == fold]

    # If the pyKT preprocessing has no 'questions' column, this is a
    # concept-only dataset (assist2015): question == concept, no aggregation
    # needed. Copy concept-level valid to *_q fields verbatim, use 'concepts'
    # as question id, derive uid per prediction via `groups` if present or via
    # per-row selectmask expansion.
    if "questions" not in valid_rows.columns:
        # Build per-prediction uid from selectmasks (same shift rule)
        per_row_n, uids_flat = [], []
        for _, r in valid_rows.iterrows():
            sm = [int(x) for x in str(r["selectmasks"]).split(",")]
            selected = [i for i, s in enumerate(sm) if s == 1]
            n = max(0, len(selected) - 1)
            per_row_n.append(n)
            uids_flat.extend([int(r["uid"])] * n)
        if sum(per_row_n) != vy.size:
            return (f"MISMATCH (concept-only): total {sum(per_row_n)} != valid_y_true "
                    f"size {vy.size}")
        # For question ids, use concept ids from CSV (same 1-to-1 mapping)
        qids_flat = []
        for _, r in valid_rows.iterrows():
            sm = [int(x) for x in str(r["selectmasks"]).split(",")]
            cs = [int(x) for x in str(r["concepts"]).split(",")]
            selected = [i for i, s in enumerate(sm) if s == 1]
            predicted = selected[1:]
            qids_flat.extend([cs[p] for p in predicted])
        data["valid_y_true_q"] = np.asarray(vy, dtype=np.int64)
        data["valid_y_prob_q"] = np.asarray(vp, dtype=np.float64)
        data["valid_groups_q"] = np.asarray(uids_flat, dtype=np.int64)
        data["valid_qids_q"] = np.asarray(qids_flat, dtype=np.int64)
        np.savez_compressed(npz, **data)
        n_q = int(vy.size)
        n_students = int(np.unique(uids_flat).size)
        return (f"OK (concept-only, no aggregation) n_questions={n_q} "
                f"n_students={n_students}")

    if "is_repeat" not in valid_rows.columns:
        return ("no is_repeat column in train_valid_sequences.csv; cannot honor "
                "pyKT groupby-qidx semantics — refusing to aggregate")

    uids, qids, y_q, p_q = [], [], [], []
    offset = 0
    total_concepts = 0
    for _, r in valid_rows.iterrows():
        n_row, agg = _aggregate_row(
            r["selectmasks"], r["questions"], r["responses"], r["is_repeat"],
            int(r["uid"]), offset, vy, vp,
        )
        for uid, qid, yt, pt in agg:
            uids.append(uid); qids.append(qid); y_q.append(yt); p_q.append(pt)
        offset += n_row
        total_concepts += n_row

    if offset != vy.size:
        return (f"MISMATCH: aggregated concept-total {offset} != valid_y_true.size "
                f"{vy.size}; alignment assumption broken for this cell")

    data["valid_y_true_q"] = np.asarray(y_q, dtype=np.int64)
    data["valid_y_prob_q"] = np.asarray(p_q, dtype=np.float64)
    data["valid_groups_q"] = np.asarray(uids, dtype=np.int64)
    data["valid_qids_q"] = np.asarray(qids, dtype=np.int64)
    np.savez_compressed(npz, **data)

    n_q = len(y_q)
    n_students = int(np.unique(uids).size)
    return f"OK n_concepts={total_concepts} -> n_questions={n_q} (n_students={n_students})"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--models", nargs="+", default=DEEP)
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing valid_y_*_q fields")
    ap.add_argument("--summary-out",
                    default=str(paths.ARTIFACTS_DIR / "cross_dataset"
                                / "deep_valid_qlvl_summary.csv"))
    args = ap.parse_args()

    rows = []
    for ds in args.datasets:
        try:
            valid_df = _load_valid_csv(ds)
        except FileNotFoundError as e:
            print(f"[{ds:24s}] SKIP: {e}")
            continue
        for m in args.models:
            for f in args.folds:
                msg = _process_one(ds, m, f, valid_df, args.force)
                print(f"[{ds:24s} {m:9s} f{f}] {msg}")
                # record summary
                npz = paths.ARTIFACTS_DIR / "predictions" / ds / f"{m}_fold{f}.npz"
                if npz.exists():
                    d = np.load(npz)
                    if "valid_y_true_q" in d.files:
                        rows.append({
                            "dataset": ds, "model": m, "fold": f,
                            "n_concept_valid": int(d.get("valid_y_true", np.array([])).size),
                            "n_question_valid": int(d["valid_y_true_q"].size),
                            "n_students_valid": int(np.unique(d["valid_groups_q"]).size),
                        })

    if rows:
        Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.summary_out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\nwrote {len(rows)} summary rows -> {args.summary_out}")


if __name__ == "__main__":
    main()
