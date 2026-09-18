"""Shared, split-safe interfaces for the team's binary models."""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
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


def vishruti_models() -> dict[str, object]:
    """Return scalable initial configurations for Vishruti's two models."""
    return {
        "Linear SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LinearSVC(
                C=1.0, dual="auto", max_iter=10_000,
                random_state=RANDOM_STATE,
            )),
        ]),
        "Histogram Gradient Boosting BDT": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=200,
            max_leaf_nodes=31,
            min_samples_leaf=50,
            l2_regularization=0.1,
            early_stopping=False,
            random_state=RANDOM_STATE,
        ),
    }


def koustav_mlp() -> Pipeline:
    """Return Koustav's initial scaled feed-forward neural network."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=256,
            learning_rate_init=1e-3,
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=12,
            random_state=RANDOM_STATE,
        )),
    ])


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


def fit_vishruti_validation(
    events: pd.DataFrame,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Fit Vishruti's models on train and score validation, never test."""
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

    models = vishruti_models()
    records = []
    for name, model in models.items():
        model.fit(x_train, y_train)
        prediction = model.predict(x_validation)
        if hasattr(model, "predict_proba"):
            signal_score = model.predict_proba(x_validation)[:, 1]
        else:
            signal_score = model.decision_function(x_validation)
        records.append({
            "model": name,
            "feature_set": "full",
            "split": "validation",
            "train_rows": len(train),
            "validation_rows": len(validation),
            "features": len(features),
            **binary_metrics(y_validation, prediction, signal_score),
        })
    return models, pd.DataFrame.from_records(records)


def fit_koustav_mlp_validation(
    events: pd.DataFrame,
    selected_features: tuple[str, ...] | list[str] | None = None,
) -> tuple[Pipeline, pd.DataFrame]:
    """Fit MLP on full or frozen reduced training features and score validation."""
    train = training_events(events)
    validation = events.loc[events["split"] == "validation"].copy()
    if validation.empty or set(validation["target"].unique()) != {0, 1}:
        raise ValueError("Validation split must contain both binary classes")
    available = feature_columns(train)
    features = available if selected_features is None else list(selected_features)
    if not features or not set(features) <= set(available):
        raise ValueError("Selected features must be non-empty prepared physics columns")

    model = koustav_mlp()
    model.fit(train[features], train["target"])
    prediction = model.predict(validation[features])
    signal_score = model.predict_proba(validation[features])[:, 1]
    feature_set = "full" if selected_features is None else "reduced_rfe20"
    result = pd.DataFrame([{
        "model": "MLP",
        "feature_set": feature_set,
        "split": "validation",
        "train_rows": len(train),
        "validation_rows": len(validation),
        "features": len(features),
        **binary_metrics(validation["target"], prediction, signal_score),
    }])
    return model, result
