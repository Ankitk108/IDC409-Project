"""Frozen one-shot held-out evaluation and final comparison plots."""

from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve

from .cnn import fit_cnn_test
from .eda import feature_columns
from .evaluate import binary_metrics
from .models import ankit_models, koustav_mlp, vishruti_models


@dataclass
class FinalEvaluation:
    """Fitted models, test metrics, and immutable plotting inputs."""

    fitted_models: dict[str, Any]
    metrics: pd.DataFrame
    predictions: dict[str, np.ndarray]
    signal_scores: dict[str, np.ndarray]
    y_test: np.ndarray


def _development_baselines(development: pd.DataFrame) -> tuple[int, float, float]:
    majority = int(development["target"].value_counts().idxmax())
    signal_fraction = float(development["target"].mean())
    false_positive, true_positive, thresholds = roc_curve(
        development["target"], -development["R2"]
    )
    finite = np.isfinite(thresholds)
    best = np.flatnonzero(finite)[
        np.argmax((true_positive - false_positive)[finite])
    ]
    return majority, signal_fraction, float(-thresholds[best])


def evaluate_frozen_test(
    events: pd.DataFrame,
    selected_features: tuple[str, ...] | list[str],
    include_cnn: bool = True,
    cnn_epochs: int = 20,
) -> FinalEvaluation:
    """Refit frozen configurations on development rows and score test once."""
    development = events.loc[events["split"].isin(("train", "validation"))].copy()
    test = events.loc[events["split"] == "test"].copy()
    if development.empty or test.empty or set(test["target"].unique()) != {0, 1}:
        raise ValueError("Development and binary test rows are required")
    features = feature_columns(development)
    reduced = list(selected_features)
    if not reduced or not set(reduced) <= set(features):
        raise ValueError("Frozen reduced features must be prepared physics columns")
    if not np.isfinite(test[features].to_numpy()).all():
        raise ValueError("Test features must be finite")

    y_test = test["target"].to_numpy()
    fitted: dict[str, Any] = {}
    predictions: dict[str, np.ndarray] = {}
    scores: dict[str, np.ndarray] = {}
    records: list[dict[str, Any]] = []

    majority, signal_fraction, r2_cut = _development_baselines(development)
    baseline_outputs = {
        "Training Majority": (
            np.full(len(test), majority), np.full(len(test), signal_fraction), 0
        ),
        "R2 Cut": (
            (test["R2"].to_numpy() <= r2_cut).astype(int),
            -test["R2"].to_numpy(), 1,
        ),
    }
    fitted["Training Majority"] = {
        "majority_class": majority, "signal_fraction": signal_fraction
    }
    fitted["R2 Cut"] = {"r2_cut": r2_cut}
    for name, (prediction, score, count) in baseline_outputs.items():
        predictions[name], scores[name] = prediction, score
        records.append({
            "model": name, "feature_set": "baseline", "split": "test",
            "fit_rows": len(development), "test_rows": len(test),
            "features": count, **binary_metrics(y_test, prediction, score),
        })

    model_specs: list[tuple[str, Any, list[str], str]] = []
    for name, model in {**ankit_models(), **vishruti_models()}.items():
        model_specs.append((name, model, features, "full"))
    model_specs.extend([
        ("MLP (full)", koustav_mlp(), features, "full"),
        ("MLP (RFE-20)", koustav_mlp(), reduced, "reduced_rfe20"),
    ])
    for name, model, columns, feature_set in model_specs:
        model.fit(development[columns], development["target"])
        prediction = model.predict(test[columns])
        score = (
            model.predict_proba(test[columns])[:, 1]
            if hasattr(model, "predict_proba")
            else model.decision_function(test[columns])
        )
        fitted[name] = model
        predictions[name], scores[name] = prediction, score
        records.append({
            "model": name, "feature_set": feature_set, "split": "test",
            "fit_rows": len(development), "test_rows": len(test),
            "features": len(columns), **binary_metrics(y_test, prediction, score),
        })

    if include_cnn:
        cnn, cnn_result = fit_cnn_test(events, epochs=cnn_epochs)
        cnn_result = cnn_result.rename(columns={"evaluation_rows": "test_rows"})
        name = "Physics-grouped 1D CNN"
        fitted[name] = cnn
        predictions[name] = cnn.evaluation_prediction
        scores[name] = cnn.evaluation_score
        records.extend(cnn_result.to_dict("records"))

    metrics = pd.DataFrame.from_records(records).sort_values(
        "roc_auc", ascending=False, kind="stable"
    ).reset_index(drop=True)
    required = {
        "model", "feature_set", "split", "fit_rows", "test_rows", "features",
        "accuracy", "precision", "recall", "f1", "roc_auc",
    }
    if set(metrics.columns) != required or metrics.isna().any().any():
        raise ValueError("Final metric rows must share one complete test schema")
    if set(metrics["split"]) != {"test"} or metrics["test_rows"].nunique() != 1:
        raise ValueError("Every final metric row must describe the same test split")
    return FinalEvaluation(fitted, metrics, predictions, scores, y_test)


def plot_test_roc(result: FinalEvaluation) -> plt.Figure:
    """Plot frozen test ROC curves from stored one-shot predictions."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, score in result.signal_scores.items():
        false_positive, true_positive, _ = roc_curve(result.y_test, score)
        auc = result.metrics.loc[result.metrics["model"] == name, "roc_auc"].item()
        ax.plot(false_positive, true_positive, label=f"{name} ({auc:.3f})")
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", label="Random (0.500)")
    ax.set(xlabel="Background false-positive rate", ylabel="Signal efficiency",
           title="Frozen held-out test ROC curves")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_selected_confusion_matrices(result: FinalEvaluation) -> plt.Figure:
    """Plot the primary BDT confusion matrix for slide and report use."""
    name = "Histogram Gradient Boosting BDT"
    if name not in result.predictions:
        raise ValueError(f"Final selected prediction is missing: {name}")
    matrix = confusion_matrix(result.y_test, result.predictions[name], labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.4, 4.5))
    cmap = sns.light_palette("#10243E", as_cmap=True)
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap=cmap,
        cbar=False,
        square=True,
        linewidths=1.5,
        linecolor="white",
        xticklabels=["Background", "Signal"],
        yticklabels=["Background", "Signal"],
        annot_kws={"size": 16, "weight": "bold"},
        ax=ax,
    )
    ax.set_xlabel("Predicted class", fontsize=12)
    ax.set_ylabel("True class", fontsize=12)
    ax.tick_params(axis="both", labelsize=11, rotation=0)
    fig.tight_layout()
    return fig
