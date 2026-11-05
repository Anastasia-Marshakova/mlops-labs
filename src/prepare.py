"""Проверка данных и стратифицированное разбиение."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import load_params, save_csv


def main():
    params = load_params()
    config = params["data"]
    target = config["target"]

    frame = pd.read_csv(params["ingest"]["raw"])

    if frame.empty:
        raise ValueError("Набор данных пуст")

    if target not in frame.columns:
        raise ValueError(f"Отсутствует целевая колонка {target}")

    if not all(pd.api.types.is_numeric_dtype(t) for t in frame.dtypes):
        raise ValueError("Все колонки должны быть числовыми")

    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Обнаружены пропуски или бесконечные значения")

    if not frame[target].isin(range(1, 8)).all():
        raise ValueError("Классы Cover_Type должны быть целыми от 1 до 7")

    before = len(frame)
    frame = frame.drop_duplicates().reset_index(drop=True)
    print(f"Удалено полных дубликатов: {before - len(frame)}")

    train, test = train_test_split(
        frame,
        test_size=float(config["test_size"]),
        random_state=int(config["split_seed"]),
        stratify=frame[target],
    )

    save_csv(train, "data/processed/train.csv")
    save_csv(test, "data/processed/test.csv")

    print(f"Обучающая выборка: {len(train)} строк")
    print(f"Тестовая выборка: {len(test)} строк")


if __name__ == "__main__":
    main()
