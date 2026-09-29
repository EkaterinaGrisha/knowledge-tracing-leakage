"""Aggregate concept-level TEST predictions to question level using the
SAME operator as the valid-side aggregator uses on valid.

Motivation. The npz files store two question-level test objects:
  * `y_true` / `y_prob` — the pyKT `evaluate_question` output (a separate
    inference path with cq/cshft inputs and cross-batch "rest"-merging).
  * a *would-be* consistent version — obtained by applying the SAME
    `late_mean` (or `logit_mean`) reduction to `concept_y_true` /
    `concept_y_prob` that we already apply to `valid_y_true` / `valid_y_prob`
    (see the valid-side aggregator).

Post-hoc calibrators fit on the second and applied to the first are broken:
the gap between the two paths reaches an order of magnitude on the datasets
with multi-component questions. This script produces the missing test-side counterpart so calibrator
fit/apply are on the *same* distribution.

Fields written (added to the existing npz in place, never overwriting):
  test_y_true_q_ours, test_y_prob_q_ours, test_groups_q_ours,
  test_qids_q_ours, test_agg_operator (str field: "late_mean"|"logit_mean").

For SINGLE-KC datasets (concept == question, one row per interaction) the
aggregation is trivial and MUST equal `y_prob` bit-for-bit; we verify this
as a built-in sanity gate.
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ktx import paths

DEEP = ["dkt", "sakt", "akt", "simplekt"]
DATASETS = ["assist2009", "assist2015", "assist2017", "algebra2005",
            "bridge2algebra2006", "assist2012", "ednet"]
FOLD_RE = re.compile(r"(?P<model>[a-z+]+)_fold(?P<fold>\d+)\.npz$")

_EPS = 1e-6


def _logit(p):
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _reduce(probs, op: str) -> float:
    if op == "late_mean":
        return float(np.mean(probs))
    if op == "logit_mean":
        return float(_sigmoid(np.mean(_logit(np.asarray(probs, dtype=np.float64)))))
    raise ValueError(f"unknown aggregator: {op}")


def _load_test_csv(dataset: str) -> pd.DataFrame:
    p = paths.PYKT_ROOT / "data" / dataset / "test_sequences.csv"
    if not p.exists():
        raise FileNotFoundError(str(p))
    return pd.read_csv(p)


def _aggregate_row(sm_str, q_str, r_str, ir_str, uid: int,
                   offset: int, y_true_flat: np.ndarray,
                   y_prob_flat: np.ndarray, op: str):
    """Same is_repeat grouping as the valid aggregator: is_repeat=0 opens a
    new interaction, is_repeat=1 continues (multi-KC concept row of the same
    interaction). Reduction over concepts within an interaction is `op`.
    """
    sm = [int(x) for x in str(sm_str).split(",")]
    qs = [int(x) for x in str(q_str).split(",")]
    ir = [int(x) for x in str(ir_str).split(",")]

    selected = [i for i, s in enumerate(sm) if s == 1]
    predicted = selected[1:]  # pyKT shift: predictions start at position 1
    n = len(predicted)
    if n == 0:
        return 0, []

    row_probs = y_prob_flat[offset:offset + n]
    row_trues = y_true_flat[offset:offset + n]

    out = []
    i = 0
    while i < n:
        j = i + 1
        while j < n and ir[predicted[j]] == 1:
            j += 1
        q_prob = _reduce(row_probs[i:j], op)
        q_true = int(row_trues[i])
        cur_q = qs[predicted[i]]
        out.append((int(uid), int(cur_q), q_true, q_prob))
        i = j
    return n, out


def _process_one(dataset: str, model: str, fold: int, test_df: pd.DataFrame,
                 aggregator: str, force: bool) -> dict:
    npz = paths.ARTIFACTS_DIR / "predictions" / dataset / f"{model}_fold{fold}.npz"
    if not npz.exists():
        return {"status": "npz_missing"}
    data = dict(np.load(npz))
    need = {"concept_y_true", "concept_y_prob"}
    if not need <= set(data):
        return {"status": "no_concept_level_test"}
    # allow re-run with different aggregator
    field_true = "test_y_true_q_ours"
    field_prob = "test_y_prob_q_ours"
    field_grp = "test_groups_q_ours"
    field_qid = "test_qids_q_ours"
    if field_true in data and not force:
        current_op = str(data.get("test_agg_operator", ""))
        if current_op == aggregator:
            return {"status": "cached", "aggregator": current_op,
                    "n_q": int(data[field_true].size)}

    cy = data["concept_y_true"]
    cp = data["concept_y_prob"]

    if "questions" not in test_df.columns:
        # concept-only dataset (assist2015): 1 concept per interaction; no
        # aggregation. Copy over from concept fields verbatim.
        per_row_n, uids_flat, qids_flat = [], [], []
        for _, r in test_df.iterrows():
            sm = [int(x) for x in str(r["selectmasks"]).split(",")]
            cs = [int(x) for x in str(r["concepts"]).split(",")]
            selected = [i for i, s in enumerate(sm) if s == 1]
            predicted = selected[1:]
            per_row_n.append(len(predicted))
            uids_flat.extend([int(r["uid"])] * len(predicted))
            qids_flat.extend([cs[p] for p in predicted])
        if sum(per_row_n) != cy.size:
            return {"status": f"MISMATCH_concept_only n={sum(per_row_n)} vs {cy.size}"}
        data[field_true] = np.asarray(cy, dtype=np.int64)
        data[field_prob] = np.asarray(cp, dtype=np.float64)
        data[field_grp] = np.asarray(uids_flat, dtype=np.int64)
        data[field_qid] = np.asarray(qids_flat, dtype=np.int64)
        data["test_agg_operator"] = np.array(aggregator)
        np.savez_compressed(npz, **data)
        return {"status": "ok_concept_only", "n_q": int(cy.size),
                "n_students": int(np.unique(uids_flat).size),
                "aggregator": aggregator}

    if "is_repeat" not in test_df.columns:
        return {"status": "no_is_repeat"}

    uids, qids, y_q, p_q = [], [], [], []
    offset = 0
    for _, r in test_df.iterrows():
        n_row, agg = _aggregate_row(
            r["selectmasks"], r["questions"], r["responses"], r["is_repeat"],
            int(r["uid"]), offset, cy, cp, aggregator,
        )
        for uid, qid, yt, pt in agg:
            uids.append(uid); qids.append(qid); y_q.append(yt); p_q.append(pt)
        offset += n_row

    if offset != cy.size:
        return {"status": f"MISMATCH concept-total {offset} vs {cy.size}"}

    data[field_true] = np.asarray(y_q, dtype=np.int64)
    data[field_prob] = np.asarray(p_q, dtype=np.float64)
    data[field_grp] = np.asarray(uids, dtype=np.int64)
    data[field_qid] = np.asarray(qids, dtype=np.int64)
    data["test_agg_operator"] = np.array(aggregator)
    np.savez_compressed(npz, **data)

    return {"status": "ok", "n_q": len(y_q),
            "n_students": int(np.unique(uids).size),
            "aggregator": aggregator}


def _sanity_single_kc(data: dict) -> str:
    """For a cell we just aggregated, compare test_y_prob_q_ours against pyKT
    y_prob when both are aligned by (uid, qid) — if the dataset is single-KC
    (concept==question) they should match nearly bit-for-bit under late_mean."""
    if "y_prob" not in data or "test_y_prob_q_ours" not in data:
        return "n/a"
    if str(data.get("test_agg_operator", "")) != "late_mean":
        return "n/a (not late_mean)"
    a = np.asarray(data["y_prob"], dtype=np.float64)
    b = np.asarray(data["test_y_prob_q_ours"], dtype=np.float64)
    if a.size != b.size:
        return f"size differs: pyKT={a.size} ours={b.size}"
    diff = np.abs(a - b)
    return (f"max|Δ|={diff.max():.5f} mean|Δ|={diff.mean():.5f} "
            f"n={a.size} within_1e-3={float((diff<1e-3).mean()):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--models", nargs="+", default=DEEP)
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--aggregator", choices=["late_mean", "logit_mean"],
                    default="late_mean")
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing test_y_*_q_ours fields")
    ap.add_argument("--summary-out",
                    default=str(paths.ARTIFACTS_DIR / "cross_dataset"
                                / "deep_test_qlvl_summary.csv"))
    args = ap.parse_args()

    rows = []
    for ds in args.datasets:
        try:
            test_df = _load_test_csv(ds)
        except FileNotFoundError as e:
            print(f"[{ds:24s}] SKIP: {e}")
            continue
        for m in args.models:
            for f in args.folds:
                r = _process_one(ds, m, f, test_df, args.aggregator, args.force)
                # sanity check pyKT vs ours (single-KC datasets should match)
                if r.get("status") in ("ok", "ok_concept_only", "cached"):
                    npz = (paths.ARTIFACTS_DIR / "predictions" / ds
                           / f"{m}_fold{f}.npz")
                    d = np.load(npz)
                    r["pykt_vs_ours"] = _sanity_single_kc(dict(d))
                print(f"[{ds:24s} {m:9s} f{f}] {r}")
                if r.get("status") in ("ok", "ok_concept_only", "cached"):
                    rows.append({"dataset": ds, "model": m, "fold": f,
                                 "aggregator": r.get("aggregator", args.aggregator),
                                 "n_q_ours": r.get("n_q"),
                                 "n_students": r.get("n_students", ""),
                                 "pykt_vs_ours": r.get("pykt_vs_ours", "")})

    if rows:
        Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.summary_out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\nwrote {len(rows)} summary rows -> {args.summary_out}")


if __name__ == "__main__":
    main()
