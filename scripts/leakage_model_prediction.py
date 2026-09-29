"""Предсказание величины завышения из механизма, без подгоняемых параметров.

Связь величины завышения со средним числом компонентов на задание показана
в статье коэффициентом корреляции по семи наборам данных. Семь точек, из которых
три совпадают, — слабое основание: коэффициент близок к единице почти при любом
монотонном отношении. Если механизм описан верно, он должен предсказывать не
тенденцию по семи точкам, а величину в каждой ячейке.

Механизм даёт такую возможность. На строках-продолжениях ответ на текущее задание
уже находится в контексте, и модель его воспроизводит. Значит усреднённая
вероятность задания с кратностью k приблизительно равна

    p_avg = (p_1 + (k - 1) * y) / k,

где p_1 — предсказание первой строки, а y — истинный ответ. Величина p_1 берётся
из данных, распределение кратности берётся из данных, ничего не подбирается.
Площадь под ROC-кривой, вычисленная по такому p_avg, и есть предсказание.

Проверка состоит в сравнении предсказанного значения с наблюдаемым в той же
ячейке. Совпадение означает, что механизм объясняет величину, а не только
направление; расхождение показывает, какая часть эффекта механизмом не описана.

Вывод (artifacts/cross_dataset/):
  leakage_model_prediction.csv          строка на (набор, модель, фолд)
  leakage_model_prediction_summary.csv  строка на набор

Запуск:
  python -m scripts.leakage_model_prediction
  python -m scripts.leakage_model_prediction --datasets ednet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ktx import paths
from scripts.aggregate_deep_test_to_question import _load_test_csv

CROSS = paths.ARTIFACTS_DIR / "cross_dataset"
PRED = paths.ARTIFACTS_DIR / "predictions"
MODELS = ("dkt", "sakt", "akt", "simplekt")
DATASETS = ["assist2012", "assist2017", "bridge2algebra2006",
            "assist2009", "algebra2005", "ednet"]


def question_view(test_df: pd.DataFrame, cy: np.ndarray, cp: np.ndarray):
    """По каждому заданию: кратность, истинный ответ, предсказание первой строки
    и наблюдаемое среднее по его строкам."""
    ks, ys, p1s, obs = [], [], [], []
    offset, truncated = 0, 0
    for row in test_df.itertuples(index=False):
        sm = [int(x) for x in str(row.selectmasks).split(",")]
        ir = [int(x) for x in str(row.is_repeat).split(",")]
        sel = [i for i, s in enumerate(sm) if s == 1][1:]
        n = len(sel)
        if n == 0:
            continue
        probs = cp[offset:offset + n]
        trues = cy[offset:offset + n]
        i = 0
        while i < n:
            # После сдвига pyKT последовательность может начаться в середине
            # задания. Тогда группа открывается строкой-продолжением: её первая
            # вероятность уже содержит утечку, а кратность усечена. Такие группы
            # считаются, чтобы их доля была видна, а не подразумевалась.
            if ir[sel[i]] != 0:
                truncated += 1
            j = i + 1
            while j < n and ir[sel[j]] == 1:
                j += 1
            ks.append(j - i)
            ys.append(int(trues[i]))
            p1s.append(float(probs[i]))
            obs.append(float(np.mean(probs[i:j])))
            i = j
        offset += n
    if offset != cy.size:
        raise ValueError(f"выравнивание не сошлось: {offset} против {cy.size}")
    return (np.asarray(ks), np.asarray(ys, dtype=int),
            np.asarray(p1s), np.asarray(obs), truncated)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--out-dir", type=Path, default=CROSS)
    args = ap.parse_args()

    rows = []
    for ds in args.datasets:
        test_df = _load_test_csv(ds)
        if not {"questions", "is_repeat"} <= set(test_df.columns):
            print(f"{ds}: развёртка не определена, пропуск", flush=True)
            continue
        print(f"{ds}: последовательностей {len(test_df)}", flush=True)
        for model in MODELS:
            for fold in range(5):
                p = PRED / ds / f"{model}_fold{fold}.npz"
                if not p.exists():
                    continue
                with np.load(p, allow_pickle=True) as z:
                    if "concept_y_prob" not in z.files:
                        continue
                    cy = np.asarray(z["concept_y_true"], dtype=float).ravel()
                    cp = np.asarray(z["concept_y_prob"], dtype=float).ravel()
                    yq = np.asarray(z["y_true"]).ravel()
                    pq = np.asarray(z["y_prob"]).ravel()
                k, y, p1, obs, truncated = question_view(test_df, cy, cp)
                if len(np.unique(y)) < 2:
                    continue
                # предсказание механизма: продолжения воспроизводят ответ
                pred = (p1 + (k - 1) * y) / k
                rows.append({
                    "dataset": ds, "model": model, "fold": fold,
                    "auc_question_level": round(float(roc_auc_score(yq, pq)), 4),
                    "auc_observed_avg": round(float(roc_auc_score(y, obs)), 4),
                    "auc_predicted_avg": round(float(roc_auc_score(y, pred)), 4),
                    "auc_first_row_only": round(float(roc_auc_score(y, p1)), 4),
                    "share_k1": round(float((k == 1).mean()), 4),
                    "mean_k": round(float(k.mean()), 4),
                    "n_questions": int(k.size),
                    "n_truncated_groups": int(truncated),
                })
        print(f"  ячеек накоплено {len(rows)}", flush=True)

    if not rows:
        sys.exit("ФАТАЛЬНО: ни одной ячейки")
    out = pd.DataFrame(rows)
    out["delta_observed"] = (out.auc_observed_avg - out.auc_question_level).round(4)
    out["delta_predicted"] = (out.auc_predicted_avg - out.auc_question_level).round(4)
    out["residual"] = (out.delta_observed - out.delta_predicted).round(4)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_dir / "leakage_model_prediction.csv", index=False)

    g = (out.groupby("dataset")
         .agg(share_k1=("share_k1", "first"), mean_k=("mean_k", "first"),
              auc_question_level=("auc_question_level", "mean"),
              observed=("delta_observed", "mean"),
              predicted=("delta_predicted", "mean"),
              residual=("residual", "mean"),
              abs_residual=("residual", lambda v: float(np.abs(v).mean())))
         .reset_index().sort_values("mean_k").round(4))
    g.to_csv(args.out_dir / "leakage_model_prediction_summary.csv", index=False)
    print("\n" + g.to_string(index=False))
    tr = out.groupby("dataset").n_truncated_groups.first()
    nq = out.groupby("dataset").n_questions.first()
    print("\nгрупп, начатых со строки-продолжения (усечённая кратность):")
    for ds in tr.index:
        print(f"  {ds:20s} {int(tr[ds]):5d} из {int(nq[ds]):7d}  "
              f"({tr[ds] / nq[ds] * 100:.2f} %)")
    corr = float(np.corrcoef(out.delta_observed, out.delta_predicted)[0, 1])
    print(f"\nпо {len(out)} ячейкам: корреляция предсказанного и наблюдаемого "
          f"{corr:.4f}, наибольший остаток {out.residual.abs().max():.4f}, "
          f"средний по модулю {out.residual.abs().mean():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
