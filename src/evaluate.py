"""Оценка классификатора на тестовой выборке."""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src.utils import load_params


def main():
    params = load_params()
    target = params["data"]["target"]

    bundle = joblib.load("models/model.joblib")
    frame = pd.read_csv("data/features/test.csv")

    x_test = frame[bundle["features"]]
    y_test = frame[target]
    predictions = bundle["model"].predict(x_test)

    accuracy = float(accuracy_score(y_test, predictions))
    metrics = {
        "accuracy": accuracy,
        "f1_macro": float(
            f1_score(y_test, predictions, average="macro", zero_division=0)
        ),
        "n_test": int(len(frame)),
    }

    output = Path("reports/metrics.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(metrics, indent=2))

    if accuracy < float(params["evaluate"]["min_accuracy"]):
        raise ValueError("Accuracy ниже заданного порога")


if __name__ == "__main__":
    main()
