"""Устойчивость измеренного расхождения к начальному значению генератора.

Раздел «Ограничения» статьи говорит, что модели обучены при одном начальном
значении и не переобучались. На диске, однако, лежит полный второй набор
предсказаний: 140 ячеек с суффиксом seed43. Сценарий, считающий основную матрицу,
их отфильтровывает, и получается, что проверка устойчивости была возможна, но
не сделана. Этот сценарий её выполняет.

Второй набор предсказаний хранит не всё, что первый: массива с усреднением по
строкам задания в нём нет, поскольку дописывающий его сценарий на этих файлах
не запускался. Поэтому усреднение здесь вычисляется заново из предсказаний
уровня компонентов — и для обоих наборов одинаковым кодом, чтобы сравнение
не зависело от того, что и когда было сохранено.

Совпадение пересчитанного значения с сохранённым на первом наборе служит
самопроверкой: если реализация усреднения разошлась бы с той, что применялась
ранее, сравнение сидов потеряло бы смысл.

Вывод (artifacts/cross_dataset/):
  seed_robustness.csv          строка на (набор, модель, фолд, сид)
  seed_robustness_summary.csv  строка на набор: расхождение по каждому сиду

Запуск:
  python -m scripts.seed_robustness
  python -m scripts.seed_robustness --datasets ednet algebra2005
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ktx import paths
from scripts.aggregate_deep_test_to_question import (
    _aggregate_row,
    _load_test_csv,
)

CROSS = paths.ARTIFACTS_DIR / "cross_dataset"
PRED = paths.ARTIFACTS_DIR / "predictions"
MODELS = ("dkt", "sakt", "akt", "simplekt")
DATASETS = ["assist2009", "assist2015", "assist2017", "algebra2005",
            "bridge2algebra2006", "assist2012", "ednet"]
SEEDS = {42: "", 43: "_seed43"}
# Расхождение пересчитанного усреднения с сохранённым, выше которого сравнение
# считается недействительным: расходиться они могут только на уровне округления.
SELFCHECK_TOL = 5e-4


def aggregate(npz: dict, test_df: pd.DataFrame, op: str = "late_mean"):
    """Свести предсказания уровня компонентов к уровню задания."""
    cy = np.asarray(npz["concept_y_true"], dtype=float).ravel()
    cp = np.asarray(npz["concept_y_prob"], dtype=float).ravel()
    offset, trues, probs = 0, [], []
    for row in test_df.itertuples(index=False):
        n, out = _aggregate_row(row.selectmasks, row.questions, row.responses,
                                row.is_repeat, getattr(row, "uid", 0),
                                offset, cy, cp, op)
        offset += n
        for _, _, t, p in out:
            trues.append(t)
            probs.append(p)
    if offset != cy.size:
        raise ValueError(f"выравнивание не сошлось: разобрано {offset}, в файле {cy.size}")
    return np.asarray(trues, dtype=int), np.asarray(probs, dtype=float)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--out-dir", type=Path, default=CROSS)
    args = ap.parse_args()

    rows, selfcheck, skipped = [], [], []
    for ds in args.datasets:
        test_df = _load_test_csv(ds)
        if not {"questions", "is_repeat"} <= set(test_df.columns):
            # У набора нет идентификаторов заданий, поэтому развёртка не определена
            # и оба пути тождественны по построению: на каждое взаимодействие
            # приходится ровно одна строка компонента, усреднять нечего.
            skipped.append(ds)
            print(f"{ds}: нет колонок questions/is_repeat, развёртка не определена — "
                  f"расхождение равно нулю по построению", flush=True)
            continue
        print(f"{ds}: последовательностей {len(test_df)}", flush=True)
        for model in MODELS:
            for fold in range(5):
                for seed, suffix in SEEDS.items():
                    p = PRED / ds / f"{model}_fold{fold}{suffix}.npz"
                    if not p.exists():
                        continue
                    with np.load(p, allow_pickle=True) as z:
                        d = {k: z[k] for k in z.files}
                    if "concept_y_prob" not in d:
                        continue
                    yq, pq = np.asarray(d["y_true"]).ravel(), np.asarray(d["y_prob"]).ravel()
                    ty, tp = aggregate(d, test_df)
                    if len(np.unique(yq)) < 2 or len(np.unique(ty)) < 2:
                        continue
                    auc_q = float(roc_auc_score(yq, pq))
                    auc_l = float(roc_auc_score(ty, tp))
                    rows.append({"dataset": ds, "model": model, "fold": fold, "seed": seed,
                                 "auc_question_level": round(auc_q, 4),
                                 "auc_late_mean": round(auc_l, 4),
                                 "delta": round(auc_l - auc_q, 4),
                                 "n_question_level": int(yq.size), "n_late_mean": int(ty.size)})
                    if seed == 42 and "test_y_prob_q_ours" in d:
                        stored = float(roc_auc_score(
                            np.asarray(d["test_y_true_q_ours"]).ravel(),
                            np.asarray(d["test_y_prob_q_ours"]).ravel()))
                        selfcheck.append(abs(stored - auc_l))
        print(f"  готово, ячеек накоплено {len(rows)}", flush=True)

    if not rows:
        sys.exit("ФАТАЛЬНО: ни одной ячейки не посчитано")
    out = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_dir / "seed_robustness.csv", index=False)

    if skipped:
        print(f"\nне рассматривались (развёртка не определена): {', '.join(skipped)}")
    if selfcheck:
        worst = max(selfcheck)
        print(f"\nсамопроверка на сиде 42: сверено {len(selfcheck)} ячеек, "
              f"наибольшее расхождение с сохранённым {worst:.6f}")
        if worst > SELFCHECK_TOL:
            sys.exit(f"ФАТАЛЬНО: пересчёт разошёлся с сохранённым на {worst:.6f}")

    piv = (out.groupby(["dataset", "seed"]).delta
           .agg(["mean", "std", "count"]).reset_index())
    wide = piv.pivot(index="dataset", columns="seed", values="mean")
    wide.columns = [f"delta_seed{c}" for c in wide.columns]
    sd = piv.pivot(index="dataset", columns="seed", values="std")
    sd.columns = [f"sd_seed{c}" for c in sd.columns]
    summary = wide.join(sd).reset_index()
    if {"delta_seed42", "delta_seed43"} <= set(summary.columns):
        summary["abs_diff"] = (summary.delta_seed42 - summary.delta_seed43).abs().round(4)
    summary = summary.round(4)
    summary.to_csv(args.out_dir / "seed_robustness_summary.csv", index=False)
    print("\n" + summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
