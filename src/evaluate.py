"""Evaluation: test-set metrics, comparison figures and permutation importance."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)

from src.visualization import SPLIT_COLORS, TARGET_NAMES, save_figure


def classification_metrics(model, X, y):
    """Accuracy, precision, recall and F1 of the positive class (satisfied), and ROC-AUC."""
    predicted = model.predict(X)
    probability = model.predict_proba(X)[:, 1]
    return {
        "accuracy": accuracy_score(y, predicted),
        "precision": precision_score(y, predicted),
        "recall": recall_score(y, predicted),
        "f1": f1_score(y, predicted),
        "roc_auc": roc_auc_score(y, probability),
    }


def results_table(results, X_test, y_test):
    """One row per model: chosen hyperparameters, cross-validation scores and test scores."""
    rows = []
    for name, result in results.items():
        row = {"model": name, "best_params": result["best_params"], "search_seconds": round(result["seconds"], 1)}
        row.update({f"cv_{metric}": value for metric, value in result["cv_scores"].items()})
        row.update({f"test_{metric}": value for metric, value in
                    classification_metrics(result["model"], X_test, y_test).items()})
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def plot_model_comparison(table, metric, figures_dir=None):
    """Cross-validation and test score of each model for one metric."""
    scores = table[[f"cv_{metric}", f"test_{metric}"]].sort_values(f"cv_{metric}")
    scores.columns = ["cross-validation (train)", "test"]
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(scores) + 1.5))
    scores.plot.barh(ax=ax, color=["#9aa5b1", "#2a7ab0"], width=0.75)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8, padding=2)
    low = scores.min().min()
    ax.set(xlim=(max(0, low - 0.05), 1.0), xlabel=metric, ylabel="", title=f"Model comparison ({metric})")
    ax.legend(loc="lower right", fontsize=8)
    save_figure(fig, figures_dir, f"13_model_comparison_{metric}")
    return fig


def summarize_folds(folds, table, metric):
    """Mean train and validation scores, overfitting gap, fold-to-fold deviation and test score of each model."""
    grouped = folds.groupby("model")
    summary = pd.DataFrame({
        "train": grouped[f"train_{metric}"].mean(),
        "validation": grouped[f"validation_{metric}"].mean(),
        "validation std": grouped[f"validation_{metric}"].std(),
        "test": table[f"test_{metric}"],
        "fit seconds": grouped["fit_seconds"].mean(),
    })
    summary.insert(3, "train - validation", summary["train"] - summary["validation"])
    return summary.sort_values("validation", ascending=False)


def plot_train_validation_test(summary, metric, figures_dir=None):
    """Train, validation and test score of each model on one line: the line length is the overfitting gap."""
    ordered = summary.sort_values("validation")
    rows = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(ordered) + 1.6))
    ax.hlines(rows, ordered["validation"], ordered["train"], color="#c3c9d0", linewidth=2, zorder=1)
    # Small vertical shifts keep the three markers visible when the scores are almost equal.
    for name, marker, shift in (("train", "o", 0.2), ("validation", "s", 0), ("test", "D", -0.2)):
        ax.scatter(ordered[name], rows + shift, s=60, marker=marker, color=SPLIT_COLORS[name], edgecolor="white",
                   linewidth=1.2, label=name, zorder=3)
    ax.set_yticks(rows, [name.replace("_", " ") for name in ordered.index])
    label = metric.replace("_", " ").upper()
    ax.set(xlabel=f"{label} score", title=f"Train, validation and test {label} of each model")
    ax.legend(loc="lower right")
    sns.despine(ax=ax)
    save_figure(fig, figures_dir, "24_train_validation_test")
    return fig


def plot_roc_curves(results, X_test, y_test, figures_dir=None):
    """ROC curve of every model on the test set."""
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for name, result in results.items():
        RocCurveDisplay.from_estimator(result["model"], X_test, y_test, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)
    ax.set(title="ROC curves on the test set")
    save_figure(fig, figures_dir, "14_roc_curves")
    return fig


def plot_confusion_matrix(model, X_test, y_test, title, figures_dir=None, name="15_confusion_matrix"):
    """Confusion matrix (counts and row percentages) of one model on the test set."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    labels = [TARGET_NAMES[0], TARGET_NAMES[1]]
    for ax, normalize in zip(axes, (None, "true")):
        ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, display_labels=labels, normalize=normalize,
                                              values_format=".0f" if normalize is None else ".1%",
                                              cmap="Blues", colorbar=False, ax=ax)
        ax.grid(False)
    axes[0].set_title("counts")
    axes[1].set_title("share of each true class")
    fig.suptitle(title)
    fig.tight_layout()
    save_figure(fig, figures_dir, name)
    return fig


def permutation_importance_table(model, X, y, config, scoring="f1", n_repeats=5):
    """Drop of the score when the values of one original column are shuffled.

    The model is a full pipeline, so each of the 22 original columns is shuffled before
    the feature engineering: the importance of a service includes its engineered versions.
    """
    importance = permutation_importance(model, X, y, scoring=scoring, n_repeats=n_repeats,
                                        random_state=config["seed"], n_jobs=config["training"]["n_jobs"])
    return pd.DataFrame({"mean drop": importance.importances_mean, "std": importance.importances_std},
                        index=X.columns).sort_values("mean drop", ascending=False)


def plot_permutation_importance(table, title, figures_dir=None, name="16_permutation_importance"):
    """Bar chart of a permutation-importance table."""
    ordered = table.sort_values("mean drop")
    fig, ax = plt.subplots(figsize=(7, 0.3 * len(ordered) + 1.2))
    ax.barh(ordered.index, ordered["mean drop"], xerr=ordered["std"], color="#2a7ab0")
    ax.set(xlabel="drop of the F1-score when the column is shuffled", title=title)
    sns.despine(ax=ax)
    save_figure(fig, figures_dir, name)
    return fig
