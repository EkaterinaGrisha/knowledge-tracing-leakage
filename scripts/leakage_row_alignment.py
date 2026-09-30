"""Выравнивание двух способов оценивания по общему ключу строки.

Зачем. Два способа получения вероятности на уровне задания перечисляют задания
немного по-разному: на EdNet-KT1-5k это 132 209 строк против 133 516. Пока строки
не сопоставлены, остаётся возражение, что измеренное расхождение создано не
развёрткой, а разными наборами оцениваемых строк. В статье это возражение
закрывалось косвенно — контролем на наборах, где развёртки нет и длины совпадают
точно. Здесь оно закрывается прямо: показатели пересчитываются на пересечении
строк, и разность сравнивается с разностью на полных векторах.

Ключ. Библиотека нумерует взаимодействия сквозным номером `cidxs`, который
присваивается по `test.csv` в порядке файла (`split_datasets.get_inter_qidx`).
Номер и есть ключ. Две тонкости, обе проверены на данных:

  * отдельный проход помечает задание номером его ПОСЛЕДНЕЙ строки-компонента,
    а не первой: на EdNet доля попаданий на последнюю строку равна 1.0, на первую
    только 0.53, то есть ровно доле односоставных заданий. Наш обход поэтому тоже
    берёт последнюю строку группы;

  * `test_cidxs` индексирует склейку взаимодействий `test.csv`, а не строк
    предсказаний: у EdNet 303 612 взаимодействий против 301 372 предсказаний,
    потому что нарезка на последовательности часть из них не берёт.

Проверки, без которых счёт не имеет смысла и которые сценарий выполняет сам:
  * столбец ответов `test.csv`, проиндексированный `test_cidxs`, обязан совпасть
    с сохранённым `y_true` побитово;
  * пересчитанное усреднение обязано совпасть с сохранённым `test_y_prob_q_ours`;
  * на пересечении метка задания обязана быть одной и той же с обеих сторон.

ASSISTments-2015 пропускается: идентификаторов заданий в исходных данных нет,
развёртка не определена, и оба способа читают один и тот же массив.

Запуск:
  python -m scripts.leakage_row_alignment
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ktx import paths

CROSS = paths.ARTIFACTS_DIR / "cross_dataset"
PRED = paths.ARTIFACTS_DIR / "predictions"
MODELS = ("dkt", "sakt", "akt", "simplekt")
DATASETS = ["assist2012", "assist2017", "bridge2algebra2006",
            "assist2009", "algebra2005", "ednet"]


@dataclass
class Layout:
    """Разбор тестовых последовательностей набора: один раз на набор, а не на ячейку.

    starts/stops — границы групп строк одного задания в плоском массиве
    предсказаний уровня компонентов; keys — сквозной номер последней строки
    каждой группы, то есть ключ соединения с отдельным проходом.
    """
    starts: np.ndarray
    stops: np.ndarray
    keys: np.ndarray
    n_rows: int


def layout(dataset: str) -> Layout:
    p = paths.PYKT_ROOT / "data" / dataset / "test_sequences.csv"
    if not p.exists():
        raise FileNotFoundError(str(p))
    seq = pd.read_csv(p)
    if not {"is_repeat", "cidxs"} <= set(seq.columns):
        raise KeyError(f"{dataset}: в test_sequences.csv нет is_repeat/cidxs")
    starts, stops, keys = [], [], []
    off = 0
    for r in seq.itertuples(index=False):
        sm = [int(x) for x in str(r.selectmasks).split(",")]
        ir = [int(x) for x in str(r.is_repeat).split(",")]
        cx = [int(x) for x in str(r.cidxs).split(",")]
        pred = [i for i, s in enumerate(sm) if s == 1][1:]
        n = len(pred)
        i = 0
        while i < n:
            j = i + 1
            while j < n and ir[pred[j]] == 1:
                j += 1
            starts.append(off + i)
            stops.append(off + j)
            keys.append(cx[pred[j - 1]])
            i = j
        off += n
    return Layout(np.asarray(starts), np.asarray(stops), np.asarray(keys), off)


def responses(dataset: str) -> np.ndarray:
    """Ответы всех взаимодействий `test.csv` в порядке сквозного номера."""
    t = pd.read_csv(paths.PYKT_ROOT / "data" / dataset / "test.csv")
    out: list[int] = []
    for r in t.itertuples(index=False):
        out.extend(int(x) for x in str(r.responses).split(","))
    return np.asarray(out)


def group_mean(values: np.ndarray, lay: Layout) -> np.ndarray:
    counts = lay.stops - lay.starts
    return np.add.reduceat(values, lay.starts) / counts


def cell(dataset: str, model: str, fold: int, lay: Layout,
         resp: np.ndarray) -> dict | None:
    npz = PRED / dataset / f"{model}_fold{fold}.npz"
    if not npz.exists():
        return None
    z = np.load(npz, allow_pickle=True)
    need = {"concept_y_true", "concept_y_prob", "y_true", "y_prob", "test_cidxs"}
    if not need <= set(z.files):
        return None

    ct, cp = z["concept_y_true"], z["concept_y_prob"]
    if len(ct) != lay.n_rows:
        raise SystemExit(f"{dataset}/{model}/{fold}: предсказаний {len(ct)}, "
                         f"а обход даёт {lay.n_rows}")

    # Первая проверка: сквозной номер действительно указывает на то взаимодействие,
    # ответ которого сохранён рядом с предсказанием отдельного прохода.
    ci = z["test_cidxs"]
    if not np.array_equal(resp[ci], z["y_true"]):
        raise SystemExit(f"{dataset}/{model}/{fold}: ответы по cidxs не совпали "
                         f"с сохранёнными — соглашение о ключе неверно")

    ours_p = group_mean(cp, lay)
    ours_y = ct[lay.starts].astype(int)

    # Вторая проверка: пересчёт обязан воспроизвести сохранённый массив.
    if "test_y_prob_q_ours" in z.files:
        stored = z["test_y_prob_q_ours"]
        if len(stored) != len(ours_p) or not np.allclose(stored, ours_p, atol=1e-12):
            raise SystemExit(f"{dataset}/{model}/{fold}: пересчитанное усреднение "
                             f"разошлось с сохранённым")

    ours = pd.DataFrame({"k": lay.keys, "y": ours_y, "p": ours_p})
    pykt = pd.DataFrame({"k": ci, "y": z["y_true"], "p": z["y_prob"]})
    both = ours.merge(pykt, on="k", suffixes=("_o", "_p"))

    # Третья проверка: одно и то же задание не может иметь двух разных ответов.
    if not bool((both.y_o == both.y_p).all()):
        raise SystemExit(f"{dataset}/{model}/{fold}: на пересечении разошлись метки")

    def auc(y, p):
        return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")

    a_of, a_pf = auc(ours.y, ours.p), auc(pykt.y, pykt.p)
    a_oc, a_pc = auc(both.y_o, both.p_o), auc(both.y_p, both.p_p)
    return {
        "dataset": dataset, "model": model, "fold": fold,
        "n_ours": len(ours), "n_pykt": len(pykt), "n_common": len(both),
        "share_common_of_pykt": len(both) / len(pykt),
        "auc_ours_full": a_of, "auc_pykt_full": a_pf, "delta_full": a_of - a_pf,
        "auc_ours_common": a_oc, "auc_pykt_common": a_pc,
        "delta_common": a_oc - a_pc, "shift": (a_oc - a_pc) - (a_of - a_pf),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--folds", nargs="+", type=int, default=list(range(5)))
    args = ap.parse_args()

    rows = []
    for ds in args.datasets:
        try:
            lay = layout(ds)
        except (FileNotFoundError, KeyError) as e:
            print(f"{ds}: пропуск — {e}", flush=True)
            continue
        resp = responses(ds)
        print(f"{ds}: заданий {len(lay.keys)}, взаимодействий {len(resp)}", flush=True)
        for model in args.models:
            for fold in args.folds:
                r = cell(ds, model, fold, lay, resp)
                if r is not None:
                    rows.append(r)
        done = sum(1 for r in rows if r["dataset"] == ds)
        print(f"  ячеек посчитано {done}", flush=True)

    if not rows:
        print("нечего считать")
        return 1
    out = pd.DataFrame(rows)
    CROSS.mkdir(parents=True, exist_ok=True)
    out.round(6).to_csv(CROSS / "leakage_row_alignment.csv", index=False)

    summary = (out.groupby("dataset")
               .agg(n_cells=("delta_full", "size"),
                    share_common=("share_common_of_pykt", "mean"),
                    delta_full=("delta_full", "mean"),
                    delta_common=("delta_common", "mean"),
                    shift_mean=("shift", "mean"),
                    shift_max_abs=("shift", lambda s: float(s.abs().max())))
               .reset_index())
    summary = summary.round(6)
    summary.to_csv(CROSS / "leakage_row_alignment_summary.csv", index=False)
    print()
    print(summary.to_string(index=False))
    print(f"\nнаибольший сдвиг разности по всем ячейкам: "
          f"{out['shift'].abs().max():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
