# Материалы к статье об утечке меток при развёртке заданий на компоненты знания

Репозиторий содержит артефакты и код, которыми проверяются числа статьи
«Утечка меток при развёртке заданий на компоненты знания, величина смещения оценок
в стандартном протоколе вывода».

## Что здесь есть

```
artifacts/cross_dataset/   четырнадцать CSV — источник всех чисел статьи
scripts/                   код, который эти CSV производит
ktx/                       разрешение путей, от которого зависят сценарии
```

Репозиторий существует, чтобы числа статьи можно было проверить по артефактам.

## Соответствие таблиц и артефактов

| место в статье | артефакт |
|---|---|
| табл. 1 | `dataset_profile.csv` (сводит все столбцы), `leakage_kc_per_question.csv` |
| табл. 2, рис. 2 | `leakage_auc_matrix.csv`, `leakage_auc_summary.csv` |
| табл. 3 | `leakage_model_prediction.csv`, `leakage_model_prediction_summary.csv` |
| табл. 4, табл. 5, рис. 1, рис. 3 | `leakage_by_repeat.csv`, `leakage_by_repeat_summary.csv` |
| табл. 6 | `seed_robustness.csv`, `seed_robustness_summary.csv` |
| п. 6.2, выравнивание строк | `leakage_row_alignment.csv`, `leakage_row_alignment_summary.csv` |
| п. 6.5, доля повторов в истории | `leakage_history_dose.csv`, `leakage_history_dose_summary.csv` |


Профиль наборов данных, из которого собрана табл. 1:

```
python -m scripts.build_dataset_profile
```

Два его столбца берутся из конфигурации форка pyKT, которой здесь нет, поэтому
в этом репозитории сценарий печатает готовый артефакт.

## Что требует исходных данных

Пересчёт самих CSV. Сценарии `leakage_auc_matrix.py`, `leakage_by_repeat.py`,
`leakage_model_prediction.py` и `seed_robustness.py` читают сохранённые предсказания
моделей, которые в репозиторий не помещены.
Сценарии `leakage_by_repeat.py`, `leakage_row_alignment.py` и `leakage_history_dose.py`
дополнительно читают файлы `test_sequences.csv` и `test.csv` из каталога данных pyKT, путь к нему задаётся
переменной окружения `KT_PYKT_ROOT`.

## Установка

Python 3.10 или новее.

```
pip install -r requirements.txt
```

## Лицензия

MIT, см. файл LICENSE.
