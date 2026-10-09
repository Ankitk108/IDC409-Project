"""Post-hoc reduced-model studies without reopening the frozen test set."""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from .config import RANDOM_STATE
from .eda import feature_columns
from .evaluate import binary_metrics
from .models import vishruti_models


@dataclass(frozen=True)
class ReducedRandomForestResult:
    """Development-only OOF summary and plotting inputs."""

    metrics: pd.DataFrame
    truth: np.ndarray
    prediction: np.ndarray
    signal_score: np.ndarray


@dataclass(frozen=True)
class ReducedBDTComparisonResult:
    """Matched full-versus-reduced BDT OOF summaries and plotting inputs."""

    metrics: pd.DataFrame
    truth: np.ndarray
    predictions: dict[str, np.ndarray]
    signal_scores: dict[str, np.ndarray]


def reduced_random_forest(n_estimators: int = 300) -> RandomForestClassifier:
    """Return the fixed reduced-feature Random Forest configuration."""
    if n_estimators < 1:
        raise ValueError("Random Forest must contain at least one tree")
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def evaluate_reduced_random_forest_cv(
    events: pd.DataFrame,
    selected_features: tuple[str, ...] | list[str],
    fold_count: int = 5,
    n_estimators: int = 300,
) -> ReducedRandomForestResult:
    """Evaluate frozen RFE features with development-only OOF predictions."""
    development = events.loc[events["split"].isin(("train", "validation"))].copy()
    if development.empty or set(development["target"].unique()) != {0, 1}:
        raise ValueError("Binary development rows must contain both classes")
    available = feature_columns(development)
    features = list(selected_features)
    if not features or not set(features) <= set(available):
        raise ValueError("Reduced features must be prepared physics columns")
    if fold_count < 2 or development["target"].value_counts().min() < fold_count:
        raise ValueError("Both classes must support the requested stratified folds")

    folds = StratifiedKFold(
        n_splits=fold_count, shuffle=True, random_state=RANDOM_STATE
    )
    probabilities = cross_val_predict(
        reduced_random_forest(n_estimators),
        development[features],
        development["target"],
        cv=folds,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]
    prediction = (probabilities >= 0.5).astype(int)
    truth = development["target"].to_numpy()
    metrics = pd.DataFrame([{
        "model": "Random Forest (RFE-20, post-hoc)",
        "feature_set": "reduced_rfe20",
        "evaluation": f"development_{fold_count}fold_oof",
        "rows": len(development),
        "features": len(features),
        **binary_metrics(truth, prediction, probabilities),
    }])
    return ReducedRandomForestResult(metrics, truth, prediction, probabilities)


def plot_reduced_random_forest_diagnostics(
    result: ReducedRandomForestResult,
) -> plt.Figure:
    """Plot OOF ROC and confusion matrix for the reduced Random Forest."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    RocCurveDisplay.from_predictions(
        result.truth, result.signal_score, name="RF (RFE-20)", ax=axes[0]
    )
    axes[0].plot([0, 1], [0, 1], color="black", linestyle="--")
    axes[0].set_title("Development-only 5-fold OOF ROC")
    ConfusionMatrixDisplay.from_predictions(
        result.truth,
        result.prediction,
        display_labels=["background", "signal"],
        colorbar=False,
        ax=axes[1],
    )
    axes[1].set_title("Random Forest (frozen RFE-20)")
    fig.tight_layout()
    return fig


def evaluate_reduced_bdt_comparison_cv(
    events: pd.DataFrame,
    selected_features: tuple[str, ...] | list[str],
    fold_count: int = 5,
) -> ReducedBDTComparisonResult:
    """Compare the unchanged BDT on full and frozen reduced development inputs."""
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
        raise ValueError("Reduced features must be unique prepared physics columns")
    if fold_count < 2 or development["target"].value_counts().min() < fold_count:
        raise ValueError("Both classes must support the requested stratified folds")

    truth = development["target"].to_numpy()
    feature_sets = {
        "BDT (full, post-hoc)": (full_features, "full"),
        "BDT (RFE-20, post-hoc)": (reduced_features, "reduced_rfe20"),
    }
    metric_rows = []
    predictions = {}
    signal_scores = {}
    for model_name, (features, feature_set) in feature_sets.items():
        # Recreate identical shuffled folds for each feature set.
        folds = StratifiedKFold(
            n_splits=fold_count, shuffle=True, random_state=RANDOM_STATE
        )
        estimator = vishruti_models()["Histogram Gradient Boosting BDT"]
        probabilities = cross_val_predict(
            estimator,
            development[features],
            development["target"],
            cv=folds,
            method="predict_proba",
            n_jobs=1,
        )[:, 1]
        prediction = (probabilities >= 0.5).astype(int)
        predictions[model_name] = prediction
        signal_scores[model_name] = probabilities
        metric_rows.append({
            "model": model_name,
            "feature_set": feature_set,
            "evaluation": f"development_{fold_count}fold_oof",
            "rows": len(development),
            "features": len(features),
            **binary_metrics(truth, prediction, probabilities),
        })
    return ReducedBDTComparisonResult(
        pd.DataFrame(metric_rows), truth, predictions, signal_scores
    )


def plot_reduced_bdt_comparison(
    result: ReducedBDTComparisonResult,
) -> plt.Figure:
    """Plot matched OOF ROC curves and confusion matrices for both BDT inputs."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for model_name, score in result.signal_scores.items():
        label = "BDT full" if "full" in model_name else "BDT RFE-20"
        RocCurveDisplay.from_predictions(
            result.truth, score, name=label, ax=axes[0]
        )
    axes[0].plot([0, 1], [0, 1], color="black", linestyle="--")
    axes[0].set_title("Development-only 5-fold OOF ROC")

    for axis, model_name in zip(axes[1:], result.predictions):
        ConfusionMatrixDisplay.from_predictions(
            result.truth,
            result.predictions[model_name],
            display_labels=["background", "signal"],
            colorbar=False,
            ax=axis,
        )
        axis.set_title("BDT full" if "full" in model_name else "BDT RFE-20")
    fig.tight_layout()
    return fig
