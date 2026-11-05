"""Загрузка Covertype и создание воспроизводимого среза одной зоны."""

from pathlib import Path

import yaml
from sklearn.datasets import fetch_covtype


def main():
    params = yaml.safe_load(Path("params.yaml").read_text(encoding="utf-8"))
    config = params["ingest"]
    target = params["data"]["target"]

    dataset = fetch_covtype(
        data_home=config["cache_dir"],
        as_frame=True,
        shuffle=False,
    )

    frame = dataset.data.copy()
    frame[target] = dataset.target.astype("int64")

    zone = frame.loc[frame[config["wilderness_column"]] == 1].copy()

    # Один и тот же порядок позволяет расширять v1, сохраняя все её строки.
    zone = zone.sample(frac=1, random_state=config["seed"])
    n_rows = int(config["n_rows"])

    if not 0 < n_rows <= len(zone):
        raise ValueError(f"n_rows должно быть от 1 до {len(zone)}")

    selected = zone.iloc[:n_rows].reset_index(drop=True)

    if selected.isna().any().any():
        raise ValueError("В данных обнаружены пропуски")

    output = Path(config["raw"])
    output.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output, index=False)

    print(f"Сохранено строк: {len(selected)}")
    print(f"Число признаков: {len(selected.columns) - 1}")
    print(f"Файл: {output}")
    print("Распределение классов:")
    print(selected[target].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
