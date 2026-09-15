"""Reusable positive-signal binary classification metrics."""

from collections.abc import Mapping

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)


def binary_metrics(y_true, y_pred, signal_probability) -> Mapping[str, float]:
    """Evaluate target=1 signal using the five required project metrics."""
    if not set(np.unique(y_true)) == {0, 1}:
        raise ValueError("Metric rows must contain both binary labels")
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, signal_probability)),
    }
