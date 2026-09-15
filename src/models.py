"""Shared, split-safe interfaces for the team's binary models."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .config import RANDOM_STATE
from .eda import feature_columns, training_events
from .evaluate import binary_metrics


def ankit_models() -> dict[str, object]:
    """Return initial full-feature configurations for Ankit's two models."""
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                C=1.0, max_iter=1000, random_state=RANDOM_STATE
            )),
        ]),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=100, random_state=RANDOM_STATE
        ),
    }


def fit_ankit_validation(
    events: pd.DataFrame,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Fit stored train rows and score validation; never read test rows."""
    train = training_events(events)
    validation = events.loc[events["split"] == "validation"].copy()
    if validation.empty or set(validation["target"].unique()) != {0, 1}:
        raise ValueError("Validation split must contain both binary classes")
    if {"type", "Unnamed: 0"} & set(events.columns):
        raise ValueError("Original truth and exported row number cannot be model inputs")

    features = feature_columns(train)
    x_train, y_train = train[features], train["target"]
    x_validation, y_validation = validation[features], validation["target"]
    if not np.isfinite(x_validation.to_numpy()).all():
        raise ValueError("Validation features must be finite")

    models = ankit_models()
    records = []
    for name, model in models.items():
        model.fit(x_train, y_train)
        prediction = model.predict(x_validation)
        signal_probability = model.predict_proba(x_validation)[:, 1]
        records.append({
            "model": name,
            "feature_set": "full",
            "split": "validation",
            "train_rows": len(train),
            "validation_rows": len(validation),
            "features": len(features),
            **binary_metrics(y_validation, prediction, signal_probability),
        })
    return models, pd.DataFrame.from_records(records)
