"""Утилиты для безопасной работы в Kaggle: атомарные записи, детект окружения.

Зависимости — только стандартная библиотека.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json_atomic(path: str | Path, data: Any) -> None:
    """Запись JSON через временный файл и os.replace, чтобы не получить
    половинчатый файл при падении сессии в момент записи."""
    target = Path(path)
    ensure_dir(target.parent)
    fd, tmp_path = tempfile.mkstemp(prefix=target.name + ".", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, target)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def load_json_or_default(path: str | Path, default: Any) -> Any:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return default
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def detect_environment() -> str:
    """Возвращает 'kaggle' если код выполняется в Kaggle, иначе 'local'."""
    if Path("/kaggle").exists():
        return "kaggle"
    return "local"


def get_data_root(kaggle_dataset_slug: str | None = None) -> Path:
    """Корень с входными данными для текущего окружения.

    На Kaggle: /kaggle/input/<slug> если передан slug, иначе /kaggle/input.
    Локально: data/raw в текущей рабочей директории.
    """
    if detect_environment() == "kaggle":
        base = Path("/kaggle/input")
        if kaggle_dataset_slug:
            return base / kaggle_dataset_slug
        return base
    return Path("data/raw")
