"""Post-hoc reduced Random Forest without reopening the frozen test set."""

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


@dataclass(frozen=True)
class ReducedRandomForestResult:
    """Development-only OOF summary and plotting inputs."""

    metrics: pd.DataFrame
    truth: np.ndarray
    prediction: np.ndarray
    signal_score: np.ndarray


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
