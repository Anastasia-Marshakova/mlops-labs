"""Обучение классификатора Covertype."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.utils import load_params


def main():
    params = load_params()
    config = params["train"]
    target = params["data"]["target"]

    if config["model"] != "random_forest":
        raise ValueError(f"Неизвестная модель: {config['model']}")

    frame = pd.read_csv("data/features/train.csv")
    x_train = frame.drop(columns=[target])
    y_train = frame[target]

    model = RandomForestClassifier(
        n_estimators=int(config["n_estimators"]),
        max_depth=config["max_depth"],
        min_samples_leaf=int(config["min_samples_leaf"]),
        random_state=int(config["seed"]),
        n_jobs=int(config["n_jobs"]),
    )
    model.fit(x_train, y_train)

    output = Path("models/model.joblib")
    output.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "features": list(x_train.columns),
        },
        output,
    )

    print(f"Модель обучена на {len(frame)} строках")
    print(f"Сохранена: {output}")


if __name__ == "__main__":
    main()
