"""Training-defined baselines for the binary continuum-suppression task."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from .eda import feature_columns, training_events
from .evaluate import binary_metrics


@dataclass(frozen=True)
class BaselineDefinition:
    """Values learned from training rows and frozen before validation."""

    majority_class: int
    signal_fraction: float
    r2_cut: float


def define_baselines(events: pd.DataFrame) -> BaselineDefinition:
    """Learn the majority prediction and R2 cut from training rows only."""
    train = training_events(events)
    feature_columns(train)  # Also rejects raw truth and exported row numbers.
    if "R2" not in train.columns:
        raise KeyError("The physics-cut baseline requires the R2 feature")

    counts = train["target"].value_counts().sort_index()
    majority_class = int(counts.idxmax())
    signal_fraction = float(train["target"].mean())

    # Lower R2 is signal-like, so -R2 is the increasing signal score.
    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        train["target"], -train["R2"]
    )
    finite = np.isfinite(thresholds)
    best = np.flatnonzero(finite)[
        np.argmax((true_positive_rate - false_positive_rate)[finite])
    ]
    r2_cut = float(-thresholds[best])
    return BaselineDefinition(majority_class, signal_fraction, r2_cut)


def evaluate_baselines(events: pd.DataFrame) -> tuple[BaselineDefinition, pd.DataFrame]:
    """Evaluate frozen training-defined baselines on validation rows only."""
    definition = define_baselines(events)
    validation = events.loc[events["split"] == "validation"].copy()
    if validation.empty or set(validation["target"].unique()) != {0, 1}:
        raise ValueError("Validation split must contain both binary classes")

    y_validation = validation["target"].to_numpy()
    majority_prediction = np.full(len(validation), definition.majority_class)
    majority_score = np.full(len(validation), definition.signal_fraction)
    cut_prediction = (validation["R2"].to_numpy() <= definition.r2_cut).astype(int)
    r2_signal_score = -validation["R2"].to_numpy()

    common = {
        "feature_set": "baseline",
        "split": "validation",
        "train_rows": int((events["split"] == "train").sum()),
        "validation_rows": len(validation),
    }
    records = [
        {
            "model": "Training Majority",
            **common,
            "features": 0,
            **binary_metrics(y_validation, majority_prediction, majority_score),
        },
        {
            "model": "R2 Cut",
            **common,
            "features": 1,
            **binary_metrics(y_validation, cut_prediction, r2_signal_score),
        },
    ]
    return definition, pd.DataFrame.from_records(records)
