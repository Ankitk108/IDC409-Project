"""Optional PyTorch multi-branch CNN with physics-defined local sequences."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import RANDOM_STATE
from .eda import feature_columns, training_events
from .evaluate import binary_metrics


CLEO_BC = tuple(f"CleoConeCS__bo{i}__bc" for i in range(1, 10))
CLEO_ROE = tuple(f"CleoConeCS__bo{i}__cmROE__bc" for i in range(1, 10))
KSFW_TERMS = (
    "hso00", "hso02", "hso04", "hso10", "hso12", "hso14",
    "hso20", "hso22", "hso24", "hoo0", "hoo1", "hoo2", "hoo3", "hoo4",
)
KSFW_BC = tuple(f"KSFWVariables__bo{term}__bc" for term in KSFW_TERMS)
KSFW_FS1 = tuple(f"KSFWVariables__bo{term}__cmFS1__bc" for term in KSFW_TERMS)


@dataclass(frozen=True)
class CNNFeatureGroups:
    scalar: tuple[str, ...]
    cleo_bc: tuple[str, ...]
    cleo_roe: tuple[str, ...]
    ksfw_bc: tuple[str, ...]
    ksfw_fs1: tuple[str, ...]


@dataclass
class FittedCNN:
    model: Any
    scaler: StandardScaler
    groups: CNNFeatureGroups
    evaluation_prediction: np.ndarray
    evaluation_score: np.ndarray


def cnn_feature_groups(features: list[str]) -> CNNFeatureGroups:
    """Separate only physically ordered families from scalar measurements."""
    sequences = set(CLEO_BC + CLEO_ROE + KSFW_BC + KSFW_FS1)
    missing = sequences - set(features)
    if missing:
        raise ValueError(f"CNN sequence features are missing: {sorted(missing)}")
    scalar = tuple(feature for feature in features if feature not in sequences)
    if not scalar:
        raise ValueError("CNN requires at least one scalar event-shape feature")
    return CNNFeatureGroups(scalar, CLEO_BC, CLEO_ROE, KSFW_BC, KSFW_FS1)


def _fit_cnn_evaluation(
    events: pd.DataFrame,
    fit_splits: tuple[str, ...],
    evaluation_split: str,
    epochs: int = 20,
    batch_size: int = 256,
) -> tuple[FittedCNN, pd.DataFrame]:
    """Fit the experimental CNN on named rows and score one held-out split."""
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as error:
        raise ImportError(
            "PyTorch is optional; install requirements-cnn.txt to run the CNN"
        ) from error
    if epochs < 1 or batch_size < 1:
        raise ValueError("CNN epochs and batch size must be positive")

    fit = events.loc[events["split"].isin(fit_splits)].copy()
    evaluation = events.loc[events["split"] == evaluation_split].copy()
    if fit.empty or evaluation.empty or set(evaluation["target"].unique()) != {0, 1}:
        raise ValueError("Fit and evaluation rows must exist with both evaluation classes")
    features = feature_columns(fit)
    groups = cnn_feature_groups(features)
    scaler = StandardScaler().fit(fit[features])
    fit_scaled = pd.DataFrame(
        scaler.transform(fit[features]), columns=features, index=fit.index
    )
    evaluation_scaled = pd.DataFrame(
        scaler.transform(evaluation[features]), columns=features, index=evaluation.index
    )

    def tensors(frame: pd.DataFrame):
        scalar = torch.tensor(frame[list(groups.scalar)].to_numpy(), dtype=torch.float32)
        cleo = torch.tensor(np.stack([
            frame[list(groups.cleo_bc)].to_numpy(),
            frame[list(groups.cleo_roe)].to_numpy(),
        ], axis=1), dtype=torch.float32)
        ksfw = torch.tensor(np.stack([
            frame[list(groups.ksfw_bc)].to_numpy(),
            frame[list(groups.ksfw_fs1)].to_numpy(),
        ], axis=1), dtype=torch.float32)
        return scalar, cleo, ksfw

    class ContinuumCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scalar = nn.Sequential(nn.Linear(len(groups.scalar), 16), nn.ReLU())
            self.cleo = nn.Sequential(
                nn.Conv1d(2, 8, kernel_size=3, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            )
            self.ksfw = nn.Sequential(
                nn.Conv1d(2, 8, kernel_size=3, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            )
            self.output = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))

        def forward(self, scalar, cleo, ksfw):
            combined = torch.cat(
                [self.scalar(scalar), self.cleo(cleo), self.ksfw(ksfw)], dim=1
            )
            return self.output(combined).squeeze(1)

    torch.manual_seed(RANDOM_STATE)
    generator = torch.Generator().manual_seed(RANDOM_STATE)
    model = ContinuumCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_function = nn.BCEWithLogitsLoss()
    fit_tensors = tensors(fit_scaled)
    labels = torch.tensor(fit["target"].to_numpy(), dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(*fit_tensors, labels), batch_size=batch_size,
        shuffle=True, generator=generator,
    )
    model.train()
    for _ in range(epochs):
        for scalar, cleo, ksfw, target in loader:
            optimizer.zero_grad()
            loss = loss_function(model(scalar, cleo, ksfw), target)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(*tensors(evaluation_scaled))
        scores = torch.sigmoid(logits).numpy()
    prediction = (scores >= 0.5).astype(int)
    result = pd.DataFrame([{
        "model": "Physics-grouped 1D CNN",
        "feature_set": "full_grouped",
        "split": evaluation_split,
        "fit_rows": len(fit),
        "evaluation_rows": len(evaluation),
        "features": len(features),
        **binary_metrics(evaluation["target"], prediction, scores),
    }])
    return FittedCNN(model, scaler, groups, prediction, scores), result


def fit_cnn_validation(
    events: pd.DataFrame,
    epochs: int = 20,
    batch_size: int = 256,
) -> tuple[FittedCNN, pd.DataFrame]:
    """Fit the experimental CNN on training rows and score validation only."""
    fitted, result = _fit_cnn_evaluation(
        events, ("train",), "validation", epochs, batch_size
    )
    result = result.rename(columns={
        "fit_rows": "train_rows", "evaluation_rows": "validation_rows"
    })
    return fitted, result


def fit_cnn_test(
    events: pd.DataFrame,
    epochs: int = 20,
    batch_size: int = 256,
) -> tuple[FittedCNN, pd.DataFrame]:
    """Refit the frozen CNN on development rows and score test once."""
    fitted, result = _fit_cnn_evaluation(
        events, ("train", "validation"), "test", epochs, batch_size
    )
    return fitted, result.rename(columns={"evaluation_rows": "test_rows"})
