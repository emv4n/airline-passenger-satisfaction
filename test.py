"""Inference: load the saved model and predict the satisfaction of new, unseen passengers.

Independent of the training: it only needs models/best_model.pkl.

Usage:
    python test.py                                  # example passengers of data/samples/
    python test.py --input my_passengers.csv        # any CSV with the columns of the Kaggle files
"""

import argparse
import json

import joblib
import pandas as pd

from src.data import clean_columns
from src.utils import load_config, project_path


def parse_arguments():
    parser = argparse.ArgumentParser(description="Predict the satisfaction of new passengers.")
    parser.add_argument("--input", default=None, help="CSV of passengers (default: the example file)")
    return parser.parse_args()


def main():
    args = parse_arguments()
    config = load_config()
    model = joblib.load(project_path(config["paths"]["model_file"]))
    card = json.loads(project_path(config["paths"]["model_card"]).read_text(encoding="utf-8"))

    raw = pd.read_csv(args.input or project_path(config["paths"]["new_samples"]))
    passengers = clean_columns(raw, config).drop(columns=config["dataset"]["target"], errors="ignore")

    probability = model.predict_proba(passengers)[:, 1]
    report = pd.DataFrame({
        "type_of_travel": passengers["type_of_travel"],
        "class": passengers["class"],
        "age": passengers["age"],
        "P(satisfied)": probability.round(3),
        "prediction": ["satisfied" if p >= 0.5 else "neutral or dissatisfied" for p in probability],
    })
    print(f"Model: {card['model']} (test F1 {card['test_scores']['f1']:.3f})")
    print(report.to_string())


if __name__ == "__main__":
    main()
