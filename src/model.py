"""Model definitions: the classifiers of the course and their hyperparameter grids."""

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.preprocessing import build_preprocessor

# Histogram-based gradient boosting: the gradient boosting of the course, with features binned
# into 255 values so that it trains in seconds on 100,000 rows.
MODEL_CLASSES = {
    "logistic_regression": LogisticRegression,
    "knn": KNeighborsClassifier,
    "lda": LinearDiscriminantAnalysis,
    "qda": QuadraticDiscriminantAnalysis,
    "decision_tree": DecisionTreeClassifier,
    "random_forest": RandomForestClassifier,
    "gradient_boosting": HistGradientBoostingClassifier,
}
RANDOMIZED_MODELS = {"decision_tree", "random_forest", "gradient_boosting"}


def build_model(name, config):
    """Pipeline = preprocessing + classifier with its fixed parameters from the configuration."""
    params = dict(config["models"][name].get("params") or {})
    if name in RANDOMIZED_MODELS:
        params["random_state"] = config["seed"]
    return Pipeline([("preprocess", build_preprocessor(config)), ("model", MODEL_CLASSES[name](**params))])


def param_grid(name, config):
    """Hyperparameter search space of a model, with the `model__` prefix used by the pipeline."""
    return {f"model__{key}": values for key, values in config["models"][name]["grid"].items()}
