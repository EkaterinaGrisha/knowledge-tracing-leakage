"""Свести все столбцы табл. 1 статьи С1 в один артефакт.

Табл. 1 собиралась из трёх мест, и одно из них — конфигурация форка pyKT, которой
нет в репозитории воспроизведения. Читатель, открывший первую же таблицу статьи,
не мог сверить её ни с чем. Скрипт сводит все шесть столбцов в один файл и
записывает происхождение каждого.

Происхождение столбцов:
  n_questions_total, n_components_total  configs/data_config.json форка pyKT
  max_kc_per_question                    concept_vs_question_inflation.csv рабочего
                                         репозитория; в готовом артефакте уже сведён
  kc_per_question_mean, n_test_rows      leakage_kc_per_question.csv
  repeat_row_share                       вычисляется как 1 - n_questions / n_component_rows

Среднее число компонентов берётся из leakage_kc_per_question.csv, а не из
concept_vs_question_inflation.csv: там та же величина посчитана на другом
множестве строк (для EdNet 2.3062 против 2.2745), и статья пользуется первой,
поскольку она посчитана на тех же строках, на которых считаются показатели.

Вывод: artifacts/cross_dataset/dataset_profile.csv

Запуск:
  python -m scripts.build_dataset_profile
"""
from __future__ import annotations

import json
import sys

import pandas as pd

from ktx import paths

CROSS = paths.ARTIFACTS_DIR / "cross_dataset"
ORDER = ["assist2012", "assist2015", "assist2017", "bridge2algebra2006",
         "assist2009", "algebra2005", "ednet"]
# Ожидаемые значения табл. 1 — якорь против молчаливого расхождения.
ANCHORS = {"ednet": (11901, 188, 7, 2.2745, 301372, 0.560),
           "algebra2005": (173113, 112, 7, 1.4472, 134674, 0.309)}


def main() -> int:
    kc = pd.read_csv(CROSS / "leakage_kc_per_question.csv").set_index("dataset")
    inf_path = CROSS / "concept_vs_question_inflation.csv"
    inf = pd.read_csv(inf_path).set_index("dataset") if inf_path.exists() else None
    cfg_path = paths.DATA_CONFIG_JSON
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    if not cfg or inf is None:
        # Без конфигурации два столбца заполнить нечем. Записать файл в таком виде
        # означало бы затереть уже готовый артефакт неполным, поэтому скрипт
        # отказывается: в репозитории воспроизведения конфигурации нет, и там
        # артефакт нужно читать, а не пересобирать.
        target = CROSS / "dataset_profile.csv"
        missing = [str(x) for x, ok in ((cfg_path, bool(cfg)), (inf_path, inf is not None))
                   if not ok]
        print(f"недоступно: {', '.join(missing)} — часть столбцов заполнить нечем.")
        if target.exists():
            print(f"Готовый артефакт оставлен без изменений: {target}")
            print(pd.read_csv(target).to_string(index=False))
            return 0
        sys.exit("ФАТАЛЬНО: нет ни конфигурации, ни готового артефакта")

    rows = []
    for ds in ORDER:
        c = cfg.get(ds, {})
        n_comp = int(kc.loc[ds, "n_component_rows"])
        n_q = int(kc.loc[ds, "n_questions"])
        rows.append({
            "dataset": ds,
            "n_questions_total": int(c["num_q"]) if c.get("num_q") else "",
            "n_components_total": int(c["num_c"]) if c.get("num_c") else "",
            "max_kc_per_question": int(inf.loc[ds, "max_kc"]),
            "kc_per_question_mean": round(float(kc.loc[ds, "kc_per_question"]), 4),
            "n_test_rows": n_comp,
            "repeat_row_share": round(1.0 - n_q / n_comp, 4),
        })
    out = pd.DataFrame(rows)
    out.to_csv(CROSS / "dataset_profile.csv", index=False)
    print(out.to_string(index=False))

    bad = []
    for ds, exp in ANCHORS.items():
        r = out[out.dataset == ds].iloc[0]
        got = (r.n_questions_total, r.n_components_total, r.max_kc_per_question,
               r.kc_per_question_mean, r.n_test_rows, round(r.repeat_row_share, 3))
        if cfg and got != exp:
            bad.append(f"{ds}: получено {got}, ожидалось {exp}")
    if bad:
        for b in bad:
            print(f"ЯКОРЬ НЕ СОШЁЛСЯ: {b}")
        return 1
    print(f"\nякоря держатся; записано {CROSS / 'dataset_profile.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
