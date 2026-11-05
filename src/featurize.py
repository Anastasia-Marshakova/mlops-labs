"""Подготовка признаков для обучения и оценки."""

import pandas as pd

from src.utils import load_params, save_csv


def main():
    params = load_params()
    target = params["data"]["target"]
    drop_columns = params["featurize"]["drop"]

    if target in drop_columns:
        raise ValueError("Нельзя удалять целевую переменную")

    for part in ("train", "test"):
        frame = pd.read_csv(f"data/processed/{part}.csv")
        features = frame.drop(columns=drop_columns)

        # Целевая переменная хранится последней колонкой.
        columns = [column for column in features.columns if column != target]
        features = features[columns + [target]]

        save_csv(features, f"data/features/{part}.csv")

        print(f"{part}: {len(features)} строк, {len(columns)} признаков")


if __name__ == "__main__":
    main()
