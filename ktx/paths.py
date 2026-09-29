"""Пути внутри репозитория воспроизведения.

Раскладка плоская: ktx/, scripts/, artifacts/, paper/. Всё разрешается
относительно корня репозитория, поэтому сценарии не зависят от текущего каталога.
"""
from __future__ import annotations

import os
from pathlib import Path

# ktx/paths.py -> ktx -> корень репозитория
REPO_ROOT = Path(__file__).resolve().parents[1]

RESEARCH_DIR = REPO_ROOT
ARTIFACTS_DIR = REPO_ROOT / "artifacts"

# Последовательности pyKT в репозиторий не входят: это часть исходных наборов
# данных. Сценарий leakage_by_repeat.py читает их отсюда, поэтому путь можно
# задать переменной окружения KT_PYKT_ROOT, указав на свою копию форка.
PYKT_ROOT = Path(os.environ["KT_PYKT_ROOT"]) if os.environ.get("KT_PYKT_ROOT") \
    else REPO_ROOT / "vendor" / "pykt-toolkit"
PYKT_DATA = PYKT_ROOT / "data"
PYKT_CONFIGS = PYKT_ROOT / "configs"
KT_CONFIG_JSON = PYKT_CONFIGS / "kt_config.json"
DATA_CONFIG_JSON = PYKT_CONFIGS / "data_config.json"

CHECKPOINTS_DIR = ARTIFACTS_DIR / "checkpoints"
MLRUNS_DIR = REPO_ROOT / "mlruns"


def ensure_output_dirs() -> None:
    for d in (ARTIFACTS_DIR, CHECKPOINTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
