"""Зависит ли разность на первых строках от доли повторов в истории.

Зачем. На первых строках задания ответ модели ещё не показан, и утечки текущего
ответа там нет по построению, однако развёрнутый проход всё равно даёт площадь
под ROC-кривой выше отдельного. Работа объясняет это различием в представлении
истории: при развёртке один и тот же прошлый ответ входит в контекст столько раз,
сколько у задания компонентов. Объяснение до сих пор опиралось только на то, что
разность растёт со средней кратностью набора, то есть на семи точках.

Здесь та же связь проверяется внутри набора, по заданиям. Для каждого задания
считается, сколько строк-продолжений было в контексте до него, и задания делятся
на четверти по доле таких строк. Если объяснение верно, разность должна расти от
нижней четверти к верхней.

Длина истории удерживается окном по позиции задания в последовательности. Без
окна связь смешалась бы с обычным улучшением предсказания по мере накопления
истории: чем дальше позиция, тем больше в ней и повторов, и полезных сведений.
Окно разводит эти две величины, а несколько окон показывают, что вывод от выбора
одного из них не зависит.

Чего расчёт не даёт. Направления он не определяет. И выигрыш развёрнутого прохода
от повторов, и потеря отдельного прохода из-за непривычного формата входа растут
с одной и той же величиной, поэтому развести их наблюдением нельзя — нужен прогон
модели на входе, из истории которого повторы удалены.

Ключ соединения двух способов — сквозной номер взаимодействия; соглашение о нём
описано в `leakage_row_alignment.py`.

Запуск:
  python -m scripts.leakage_history_dose
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ktx import paths
from scripts.leakage_row_alignment import layout

CROSS = paths.ARTIFACTS_DIR / "cross_dataset"
PRED = paths.ARTIFACTS_DIR / "predictions"
MODELS = ("dkt", "sakt", "akt", "simplekt")
DATASETS = ["ednet", "algebra2005", "assist2009", "bridge2algebra2006"]
# Окна по позиции задания в последовательности. None — без ограничения.
WINDOWS = [(5, 20), (20, 60), (60, 120), (0, None)]
MAIN_WINDOW = (20, 60)


def history(dataset: str) -> tuple[np.ndarray, np.ndarray]:
    """На каждое задание: сколько строк-продолжений было в контексте до него
    и какая это позиция в последовательности."""
    seq = pd.read_csv(paths.PYKT_ROOT / "data" / dataset / "test_sequences.csv")
    hist, pos = [], []
    for r in seq.itertuples(index=False):
        sm = [int(x) for x in str(r.selectmasks).split(",")]
        ir = [int(x) for x in str(r.is_repeat).split(",")]
        pred = [i for i, s in enumerate(sm) if s == 1][1:]
        n = len(pred)
        reps = k = i = 0
        while i < n:
            j = i + 1
            while j < n and ir[pred[j]] == 1:
                j += 1
            hist.append(reps)
            pos.append(k)
            reps += j - i - 1
            k += 1
            i = j
    return np.asarray(hist), np.asarray(pos)


def cell_table(dataset: str, model: str, fold: int, lay, hist, pos) -> pd.DataFrame | None:
    npz = PRED / dataset / f"{model}_fold{fold}.npz"
    if not npz.exists():
        return None
    z = np.load(npz, allow_pickle=True)
    ours = pd.DataFrame({
        "k": lay.keys,
        "y": z["concept_y_true"][lay.starts].astype(int),
        "p_first": z["concept_y_prob"][lay.starts],
        "hist": hist, "pos": pos,
    })
    pykt = pd.DataFrame({"k": z["test_cidxs"], "y_q": z["y_true"], "p_q": z["y_prob"]})
    both = ours.merge(pykt, on="k")
    if not bool((both.y == both.y_q).all()):
        raise SystemExit(f"{dataset}/{model}/{fold}: на пересечении разошлись метки")
    # Доля повторов на одно пройденное задание: именно доля, а не их число,
    # иначе величина повторяет длину истории.
    both["frac"] = both["hist"] / np.maximum(both["pos"], 1)
    return both


def quartiles(both: pd.DataFrame, lo: int, hi: int | None) -> list[dict]:
    w = both[both.pos >= lo] if hi is None else both[(both.pos >= lo) & (both.pos <= hi)]
    if len(w) < 1000:
        return []
    w = w.copy()
    try:
        w["q"] = pd.qcut(w.frac, 4, labels=[1, 2, 3, 4], duplicates="drop")
    except ValueError:
        return []
    out = []
    for q, g in w.groupby("q", observed=True):
        if len(np.unique(g.y)) < 2:
            continue
        out.append({
            "quartile": int(q), "n": len(g), "frac_mean": float(g.frac.mean()),
            "auc_first_row": float(roc_auc_score(g.y, g.p_first)),
            "auc_question_level": float(roc_auc_score(g.y_q, g.p_q)),
        })
    return out if len(out) == 4 else []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--folds", nargs="+", type=int, default=list(range(5)))
    args = ap.parse_args()

    rows = []
    for ds in args.datasets:
        lay = layout(ds)
        hist, pos = history(ds)
        print(f"{ds}: заданий {len(lay.keys)}, повторов в истории — медиана "
              f"{int(np.median(hist))}, максимум {int(hist.max())}", flush=True)
        for model in args.models:
            for fold in args.folds:
                both = cell_table(ds, model, fold, lay, hist, pos)
                if both is None:
                    continue
                for lo, hi in WINDOWS:
                    for rec in quartiles(both, lo, hi):
                        rows.append({"dataset": ds, "model": model, "fold": fold,
                                     "pos_lo": lo, "pos_hi": -1 if hi is None else hi,
                                     **rec})
    if not rows:
        print("нечего считать")
        return 1

    out = pd.DataFrame(rows)
    out["gap"] = out.auc_first_row - out.auc_question_level
    CROSS.mkdir(parents=True, exist_ok=True)
    out.round(6).to_csv(CROSS / "leakage_history_dose.csv", index=False)

    # Сводка: разность по четвертям и доля ячеек, где верхняя четверть выше нижней.
    recs = []
    for (ds, lo, hi), g in out.groupby(["dataset", "pos_lo", "pos_hi"]):
        wide = g.pivot_table(index=["model", "fold"], columns="quartile", values="gap")
        recs.append({
            "dataset": ds, "pos_lo": lo, "pos_hi": hi, "n_cells": len(wide),
            "gap_q1": float(wide[1].mean()), "gap_q2": float(wide[2].mean()),
            "gap_q3": float(wide[3].mean()), "gap_q4": float(wide[4].mean()),
            "cells_q4_gt_q1": int((wide[4] > wide[1]).sum()),
        })
    summary = pd.DataFrame(recs).round(6)
    summary.to_csv(CROSS / "leakage_history_dose_summary.csv", index=False)

    main_ = summary[(summary.pos_lo == MAIN_WINDOW[0]) & (summary.pos_hi == MAIN_WINDOW[1])]
    print()
    print(main_.to_string(index=False))
    total = summary.n_cells.sum()
    agree = summary.cells_q4_gt_q1.sum()
    print(f"\nверхняя четверть выше нижней в {agree} случаях из {total} "
          f"(все наборы, все окна)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
