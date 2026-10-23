"""Leakage-safe Linear Discriminant Analysis model contract."""

import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .eda import feature_columns


def lda_classifier() -> Pipeline:
    """Return scaled shrinkage LDA for correlated event-shape measurements."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LinearDiscriminantAnalysis(
            solver="lsqr",
            shrinkage="auto",
        )),
    ])


def lda_development_feature_sets(
    events: pd.DataFrame,
    selected_features: tuple[str, ...] | list[str],
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Return development rows and validated full/reduced feature definitions."""
    development = events.loc[events["split"].isin(("train", "validation"))].copy()
    if development.empty or set(development["target"].unique()) != {0, 1}:
        raise ValueError("Binary development rows must contain both classes")
    full_features = feature_columns(development)
    reduced_features = list(selected_features)
    if (
        not reduced_features
        or len(reduced_features) != len(set(reduced_features))
        or not set(reduced_features) <= set(full_features)
    ):
        raise ValueError("LDA reduced features must be unique physics columns")
    return development, {
        "full": full_features,
        "reduced_rfe20": reduced_features,
    }
