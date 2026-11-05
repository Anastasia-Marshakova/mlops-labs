"""Общие функции чтения параметров и сохранения таблиц."""

from pathlib import Path

import yaml


def load_params():
    return yaml.safe_load(Path("params.yaml").read_text(encoding="utf-8"))


def save_csv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
