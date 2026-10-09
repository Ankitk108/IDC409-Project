"""Post-hoc deep dense network on the frozen RFE-20 measurements."""

from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from .config import RANDOM_STATE
from .eda import feature_columns
from .evaluate import binary_metrics


@dataclass(frozen=True)
class DeepDenseResult:
    """Development-only OOF metrics and diagnostic inputs."""

    metrics: pd.DataFrame
    truth: np.ndarray
    prediction: np.ndarray
    signal_score: np.ndarray
    fitted_epochs: tuple[int, ...]


def evaluate_deep_dense_cv(
    events: pd.DataFrame,
    selected_features: tuple[str, ...] | list[str],
    fold_count: int = 5,
    max_epochs: int = 60,
    patience: int = 8,
    batch_size: int = 256,
) -> DeepDenseResult:
    """Create fixed OOF predictions with inner-fold early stopping."""
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as error:
        raise ImportError(
            "PyTorch is optional; install requirements-cnn.txt for the dense NN"
        ) from error
    if min(fold_count, max_epochs, patience, batch_size) < 1 or fold_count < 2:
        raise ValueError("Fold, epoch, patience, and batch settings must be positive")

    development = events.loc[events["split"].isin(("train", "validation"))].copy()
    if development.empty or set(development["target"].unique()) != {0, 1}:
        raise ValueError("Binary development rows must contain both classes")
    available = feature_columns(development)
    features = list(selected_features)
    if (
        not features
        or len(features) != len(set(features))
        or not set(features) <= set(available)
    ):
        raise ValueError("Dense-NN features must be unique prepared physics columns")
    if development["target"].value_counts().min() < fold_count:
        raise ValueError("Both classes must support the requested stratified folds")

    class DenseClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(len(features), 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.Linear(128, 64),
                nn.LayerNorm(64),
                nn.ReLU(),
                nn.Dropout(0.20),
                nn.Linear(64, 1),
            )

        def forward(self, inputs):
            return self.network(inputs).squeeze(1)

    inputs = development[features].to_numpy(dtype=np.float32)
    truth = development["target"].to_numpy(dtype=np.int64)
    scores = np.empty(len(development), dtype=np.float64)
    fitted_epochs = []
    outer_folds = StratifiedKFold(
        n_splits=fold_count, shuffle=True, random_state=RANDOM_STATE
    )
    for fold_number, (outer_train, outer_holdout) in enumerate(
        outer_folds.split(inputs, truth), start=1
    ):
        inner_train, inner_stop = train_test_split(
            outer_train,
            test_size=0.10,
            random_state=RANDOM_STATE + fold_number,
            stratify=truth[outer_train],
        )
        scaler = StandardScaler().fit(inputs[inner_train])
        train_x = torch.tensor(
            scaler.transform(inputs[inner_train]), dtype=torch.float32
        )
        stop_x = torch.tensor(
            scaler.transform(inputs[inner_stop]), dtype=torch.float32
        )
        holdout_x = torch.tensor(
            scaler.transform(inputs[outer_holdout]), dtype=torch.float32
        )
        train_y = torch.tensor(truth[inner_train], dtype=torch.float32)
        stop_y = torch.tensor(truth[inner_stop], dtype=torch.float32)

        fold_seed = RANDOM_STATE + fold_number
        torch.manual_seed(fold_seed)
        generator = torch.Generator().manual_seed(fold_seed)
        model = DenseClassifier()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=1e-3, weight_decay=1e-4
        )
        loss_function = nn.BCEWithLogitsLoss()
        loader = DataLoader(
            TensorDataset(train_x, train_y),
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
        )
        best_loss = np.inf
        best_state: dict[str, Any] | None = None
        epochs_without_improvement = 0
        best_epoch = 0
        for epoch in range(1, max_epochs + 1):
            model.train()
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                loss = loss_function(model(batch_x), batch_y)
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                stopping_loss = loss_function(model(stop_x), stop_y).item()
            if stopping_loss < best_loss - 1e-5:
                best_loss = stopping_loss
                best_epoch = epoch
                best_state = {
                    key: value.detach().clone()
                    for key, value in model.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    break
        if best_state is None:
            raise RuntimeError("Dense NN failed to produce an early-stopping state")
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            fold_scores = torch.sigmoid(model(holdout_x)).numpy()
        scores[outer_holdout] = fold_scores
        fitted_epochs.append(best_epoch)

    prediction = (scores >= 0.5).astype(int)
    metrics = pd.DataFrame([{
        "model": "Deep Dense NN (RFE-20, post-hoc)",
        "feature_set": "reduced_rfe20",
        "evaluation": f"development_{fold_count}fold_oof",
        "rows": len(development),
        "features": len(features),
        "mean_fitted_epochs": float(np.mean(fitted_epochs)),
        **binary_metrics(truth, prediction, scores),
    }])
    return DeepDenseResult(
        metrics, truth, prediction, scores, tuple(fitted_epochs)
    )


def plot_deep_dense_diagnostics(result: DeepDenseResult) -> plt.Figure:
    """Plot the dense network's development-only OOF ROC and confusion matrix."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    RocCurveDisplay.from_predictions(
        result.truth, result.signal_score, name="Deep dense RFE-20", ax=axes[0]
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
    axes[1].set_title("Deep dense NN (frozen RFE-20)")
    fig.tight_layout()
    return fig
