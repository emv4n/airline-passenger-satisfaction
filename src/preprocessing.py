"""Feature engineering and representation pipeline.

Every statistic (medians) is learned in `fit` on the training data only, then reused in
`transform`, so the test set never leaks into the preprocessing.
Each engineering step can be switched off in `configs/config.yaml` (section
`feature_engineering.steps`), which is how the ablation study measures its contribution.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ARRIVAL_DELAY = "arrival_delay_in_minutes"
DEPARTURE_DELAY = "departure_delay_in_minutes"


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Clean the raw columns and add the engineered features.

    Parameters
    ----------
    ratings : list of str
        The 14 service-rating columns (scale 1-5, where 0 means "not applicable").
    not_applicable_flags : list of str
        Rating columns that get a binary "was not applicable" indicator.
    rating_groups : dict of str -> list of str
        Groups of correlated ratings, each summarised by the mean of its ratings.
    age_bins, age_labels : list
        Edges and names of the age groups.
    delay_threshold_minutes : int
        Arrival delay from which a flight counts as late.
    handle_not_applicable : bool
        If True, flag the 0 ratings and replace them by the training median;
        if False, keep the 0 as a number (used by the ablation study).
    """

    def __init__(self, ratings, not_applicable_flags, rating_groups, age_bins, age_labels,
                 delay_threshold_minutes=15, handle_not_applicable=True):
        self.ratings = ratings
        self.not_applicable_flags = not_applicable_flags
        self.rating_groups = rating_groups
        self.age_bins = age_bins
        self.age_labels = age_labels
        self.delay_threshold_minutes = delay_threshold_minutes
        self.handle_not_applicable = handle_not_applicable

    def fit(self, X, y=None):
        """Learn the median of each rating, ignoring the "not applicable" zeros."""
        self.rating_medians_ = X[self.ratings].replace(0, np.nan).median()
        return self

    def transform(self, X):
        """Return a new DataFrame with cleaned and engineered columns."""
        out = X.copy()

        # Arrival delay is missing for 0.3% of flights; it is almost equal to the departure
        # delay (correlation 0.97, median difference 0), which is the best available estimate.
        out[ARRIVAL_DELAY] = out[ARRIVAL_DELAY].fillna(out[DEPARTURE_DELAY])

        if self.handle_not_applicable:
            for column in self.not_applicable_flags:
                out[f"{column}_not_applicable"] = (out[column] == 0).astype(int)
            # Once flagged, a zero is replaced by the typical rating so it no longer reads as "very bad".
            out[self.ratings] = out[self.ratings].replace(0, np.nan).fillna(self.rating_medians_)

        for group, columns in self.rating_groups.items():
            out[f"{group}_score"] = out[columns].mean(axis=1)

        # Distance and delays are right-skewed (skewness 1.1 and 6.7): the log compresses the long tail.
        out["log_flight_distance"] = np.log1p(out["flight_distance"])
        out["log_departure_delay"] = np.log1p(out[DEPARTURE_DELAY])
        out["log_arrival_delay"] = np.log1p(out[ARRIVAL_DELAY])
        out["is_delayed"] = (out[ARRIVAL_DELAY] >= self.delay_threshold_minutes).astype(int)

        out["age_group"] = pd.cut(out["age"], bins=self.age_bins, labels=self.age_labels).astype(str)
        out["travel_class"] = out["type_of_travel"] + " | " + out["class"]
        return out


def build_feature_engineer(config):
    """Create a FeatureEngineer from the configuration."""
    settings = config["feature_engineering"]
    return FeatureEngineer(
        ratings=config["features"]["ratings"],
        not_applicable_flags=settings["not_applicable_flags"],
        rating_groups=settings["rating_groups"],
        age_bins=settings["age_bins"],
        age_labels=settings["age_labels"],
        delay_threshold_minutes=settings["delay_threshold_minutes"],
        handle_not_applicable=settings["steps"]["not_applicable"],
    )


def engineered_columns(config):
    """List the categorical and numeric model inputs among the FeatureEngineer outputs.

    Columns that are exact or near copies of others are left out, because they make the
    covariance matrix of LDA/QDA singular and blur the logistic-regression coefficients.
    """
    settings = config["feature_engineering"]
    steps = settings["steps"]

    categorical = ["gender", "customer_type"]
    if steps["age_groups"]:
        categorical.append("age_group")
    if steps["travel_class"]:
        # The combined category already contains both main effects.
        categorical.append("travel_class")
    else:
        categorical += ["type_of_travel", "class"]

    # The departure delay is dropped: it is 0.97-correlated with the arrival delay.
    if steps["log_transform"]:
        numeric = ["age", "log_flight_distance", "log_arrival_delay"]
    else:
        numeric = ["age", "flight_distance", ARRIVAL_DELAY]
    if steps["delay_flag"]:
        numeric.append("is_delayed")

    if settings["rating_representation"] == "grouped":
        # Each group score is the mean of its ratings, so the two cannot be used together.
        numeric += ["online_boarding"] + [f"{group}_score" for group in settings["rating_groups"]]
    else:
        numeric += config["features"]["ratings"]
    if steps["not_applicable"]:
        numeric += [f"{column}_not_applicable" for column in settings["not_applicable_flags"]]
    return categorical, numeric


def build_preprocessor(config):
    """Full representation pipeline: feature engineering, one-hot encoding and scaling.

    Scaling matters for the distance-based (KNN) and gradient-based (logistic regression)
    models of the course; it does not change the tree-based models.
    """
    categorical, numeric = engineered_columns(config)
    encode_and_scale = ColumnTransformer([
        ("categorical", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical),
        ("numeric", StandardScaler(), numeric),
    ])
    return Pipeline([("engineer", build_feature_engineer(config)), ("encode", encode_and_scale)])


def build_baseline_preprocessor(config):
    """Minimal pipeline on the raw columns, used to measure what the feature engineering adds."""
    numeric = config["features"]["numeric"] + config["features"]["ratings"]
    return ColumnTransformer([
        ("categorical", OneHotEncoder(drop="first", handle_unknown="ignore"), config["features"]["categorical"]),
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
    ])
