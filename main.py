"""Training entry point: load the data, tune every model, evaluate them and save the best one.

Usage:
    python main.py                                   # all models of the configuration
    python main.py --models logistic_regression lda  # only some models
"""

import argparse
import json
from datetime import datetime

import joblib

from src.data import load_split, split_features_target
from src.evaluate import plot_confusion_matrix, plot_model_comparison, plot_roc_curves, results_table
from src.train import train_all
from src.utils import load_config, project_path


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train and compare the satisfaction models.")
    parser.add_argument("--config", default=None, help="path to a YAML configuration file")
    parser.add_argument("--models", nargs="+", default=None, help="subset of models to train")
    return parser.parse_args()


def main():
    args = parse_arguments()
    config = load_config(args.config) if args.config else load_config()
    X_train, y_train = split_features_target(load_split(config, "train"), config)
    X_test, y_test = split_features_target(load_split(config, "test"), config)

    results = train_all(X_train, y_train, config, args.models)

    # No choice depends on the test set: models and hyperparameters are selected in cross-validation.
    table = results_table(results, X_test, y_test)
    results_dir = project_path(config["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(results_dir / "model_results.csv")
    print(table.filter(like="test_").round(4).to_string())

    figures_dir = project_path(config["paths"]["figures_dir"])
    refit_metric = config["training"]["refit_metric"]
    plot_model_comparison(table, refit_metric, figures_dir)
    plot_roc_curves(results, X_test, y_test, figures_dir)

    # The best model is chosen on the cross-validation score, never on the test score.
    best_name = table[f"cv_{refit_metric}"].idxmax()
    best = results[best_name]
    plot_confusion_matrix(best["model"], X_test, y_test, f"Best model ({best_name}) on the test set", figures_dir)

    model_file = project_path(config["paths"]["model_file"])
    model_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best["model"], model_file, compress=3)
    card = {
        "model": best_name,
        "hyperparameters": best["best_params"],
        "selection_metric": refit_metric,
        "cv_scores": best["cv_scores"],
        "test_scores": {key.removeprefix("test_"): value for key, value in table.loc[best_name].items()
                        if key.startswith("test_")},
        "trained_on": f"{len(X_train)} passengers",
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    project_path(config["paths"]["model_card"]).write_text(json.dumps(card, indent=2, default=str), encoding="utf-8")
    print(f"best model: {best_name} -> {model_file}")


if __name__ == "__main__":
    main()
