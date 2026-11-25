"""Training logic: hyperparameter search in stratified cross-validation for each model."""

import time

import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split

from src.model import build_model, param_grid


def cross_validation(config):
    """Stratified K-fold splitter shared by every model, so they are compared on the same folds."""
    return StratifiedKFold(n_splits=config["training"]["cv_folds"], shuffle=True, random_state=config["seed"])


def tune_model(name, X, y, config):
    """Search the best hyperparameters of one model and refit it on the whole training set.

    Returns a dictionary with the fitted pipeline, the chosen hyperparameters, the mean
    cross-validation score of each metric and the search time.
    """
    settings = config["training"]
    folds = cross_validation(config)
    start = time.perf_counter()

    X_search, y_search = X, y
    if name == "knn" and len(X) > settings["knn_search_sample"]:
        X_search, _, y_search, _ = train_test_split(
            X, y, train_size=settings["knn_search_sample"], stratify=y, random_state=config["seed"])

    search = GridSearchCV(build_model(name, config), param_grid(name, config), cv=folds,
                          scoring=settings["metrics"], refit=settings["refit_metric"], n_jobs=settings["n_jobs"])
    search.fit(X_search, y_search)
    best_index = search.best_index_
    cv_scores = {metric: search.cv_results_[f"mean_test_{metric}"][best_index] for metric in settings["metrics"]}

    model = search.best_estimator_
    if X_search is not X:
        # Scores of the sample search are not comparable with the other models:
        # cross-validate the chosen setting on the full training set, then refit on it.
        model = build_model(name, config).set_params(**search.best_params_)
        scores = cross_validate(model, X, y, cv=folds, scoring=settings["metrics"], n_jobs=settings["n_jobs"])
        cv_scores = {metric: scores[f"test_{metric}"].mean() for metric in settings["metrics"]}
        model.fit(X, y)

    return {
        "model": model,
        "best_params": {key.removeprefix("model__"): value for key, value in search.best_params_.items()},
        "cv_scores": cv_scores,
        "seconds": time.perf_counter() - start,
    }


def fold_scores(name, params, X, y, config):
    """Train and validation scores of one model on each cross-validation fold, with given hyperparameters.

    The gap between the two shows how much the model overfits; the spread of the validation scores shows
    whether a difference between two models is larger than the fold-to-fold noise.
    Returns a DataFrame with one row per fold.
    """
    settings = config["training"]
    metric = settings["refit_metric"]
    model = build_model(name, config).set_params(**{f"model__{key}": value for key, value in params.items()})
    scores = cross_validate(model, X, y, cv=cross_validation(config), scoring=[metric],
                            return_train_score=True, n_jobs=settings["n_jobs"])
    return pd.DataFrame({
        "model": name,
        "fold": range(1, settings["cv_folds"] + 1),
        f"train_{metric}": scores[f"train_{metric}"],
        f"validation_{metric}": scores[f"test_{metric}"],
        "fit_seconds": scores["fit_time"],
    })


def train_all(X, y, config, names=None):
    """Tune every model listed in the configuration (or only `names`) and return their results."""
    results = {}
    for name in names or config["models"]:
        print(f"[train] {name} ...", flush=True)
        results[name] = tune_model(name, X, y, config)
        score = results[name]["cv_scores"][config["training"]["refit_metric"]]
        print(f"[train] {name}: CV {config['training']['refit_metric']} = {score:.4f}, "
              f"params {results[name]['best_params']}, {results[name]['seconds']:.0f} s", flush=True)
    return results
