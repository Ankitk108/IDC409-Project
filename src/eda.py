"""Training-only descriptive EDA for the professor's binary event sample."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score


def training_events(events: pd.DataFrame) -> pd.DataFrame:
    """Select the persisted train split; never explore held-out feature values."""
    if "split" not in events or "target" not in events:
        raise KeyError("EDA requires persisted split and binary target columns")
    train = events.loc[events["split"] == "train"].copy()
    if train.empty or set(train["target"].unique()) != {0, 1}:
        raise ValueError("Training split must contain both binary classes")
    return train


def feature_columns(train: pd.DataFrame) -> list[str]:
    """Return only finite numeric measurements, never truth or split columns."""
    features = [column for column in train if column not in {"target", "split"}]
    if not features or not all(pd.api.types.is_numeric_dtype(train[c]) for c in features):
        raise ValueError("EDA features must be non-empty and numeric")
    if not np.isfinite(train[features].to_numpy()).all():
        raise ValueError("EDA features must be finite")
    return features


def class_balance(train: pd.DataFrame) -> pd.DataFrame:
    """Return counts and fractions for signal (1) and background (0)."""
    counts = train["target"].value_counts().reindex([0, 1], fill_value=0)
    return pd.DataFrame({"rows": counts, "fraction": counts / len(train)})


def feature_separation(train: pd.DataFrame) -> pd.DataFrame:
    """Rank all features by orientation-free single-variable ROC-AUC."""
    rows = []
    signal = train["target"] == 1
    for feature in feature_columns(train):
        auc = float(roc_auc_score(train["target"], train[feature]))
        rows.append({
            "feature": feature,
            "separation_auc": max(auc, 1 - auc),
            "higher_for_signal": int(auc >= 0.5),
            "signal_median": float(train.loc[signal, feature].median()),
            "background_median": float(train.loc[~signal, feature].median()),
        })
    return pd.DataFrame(rows).sort_values(
        "separation_auc", ascending=False, kind="stable"
    ).reset_index(drop=True)


def feature_correlations(train: pd.DataFrame) -> pd.DataFrame:
    """Calculate Pearson correlations across all training features."""
    return train[feature_columns(train)].corr()


def plot_class_balance(balance: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(["Background (0)", "Signal (1)"], balance.loc[[0, 1], "rows"])
    ax.set_ylabel("Training events")
    ax.set_title("Training class balance")
    fig.tight_layout()
    return fig


def plot_feature_distributions(
    train: pd.DataFrame, features: list[str]
) -> plt.Figure:
    """Compare class-conditional histograms of explicitly selected features."""
    missing = set(features) - set(feature_columns(train))
    if not features or missing:
        raise ValueError(f"Select non-empty, existing physics features; missing: {missing}")
    fig, axes = plt.subplots(len(features), 1, figsize=(8, 3 * len(features)))
    for feature, ax in zip(features, np.atleast_1d(axes)):
        for target, label in [(0, "Background"), (1, "Signal")]:
            values = train.loc[train["target"] == target, feature]
            ax.hist(values, bins=50, density=True, alpha=0.5, label=label)
        ax.set_xlabel(feature)
        ax.set_ylabel("Density")
        ax.legend()
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(correlations: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(15, 13))
    sns.heatmap(correlations, cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=ax)
    ax.set_title("Training-feature Pearson correlations")
    fig.tight_layout()
    return fig
