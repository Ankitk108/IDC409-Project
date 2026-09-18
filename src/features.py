"""Training-only feature understanding and reduction utilities."""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .config import RANDOM_STATE
from .eda import feature_columns, feature_separation, training_events


@dataclass(frozen=True)
class ReductionResult:
    """Training-fitted diagnostics and one frozen measured-feature subset."""

    correlation_kept: tuple[str, ...]
    correlation_dropped: tuple[str, ...]
    selected_features: tuple[str, ...]
    pca_components_95: int
    pca_variance: pd.DataFrame
    tree_importance: pd.DataFrame
    rfe_ranking: pd.DataFrame


def analyze_feature_reduction(
    events: pd.DataFrame,
    correlation_threshold: float = 0.95,
    selected_count: int = 20,
) -> ReductionResult:
    """Fit correlation, PCA, importance, and RFE analysis on training only."""
    if not 0 < correlation_threshold < 1:
        raise ValueError("Correlation threshold must be between zero and one")
    train = training_events(events)
    features = feature_columns(train)
    if not 1 <= selected_count <= len(features):
        raise ValueError("Selected feature count is outside the available range")

    separation_order = feature_separation(train)["feature"].tolist()
    correlations = train[features].corr().abs()
    kept: list[str] = []
    dropped: list[str] = []
    for feature in separation_order:
        if any(correlations.loc[feature, other] > correlation_threshold for other in kept):
            dropped.append(feature)
        else:
            kept.append(feature)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(train[kept])
    pca = PCA(random_state=RANDOM_STATE).fit(scaled)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    components_95 = int(np.searchsorted(cumulative, 0.95) + 1)
    pca_variance = pd.DataFrame({
        "component": np.arange(1, len(cumulative) + 1),
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_variance": cumulative,
    })

    tree = DecisionTreeClassifier(
        max_depth=5, min_samples_leaf=100, random_state=RANDOM_STATE
    ).fit(train[kept], train["target"])
    tree_importance = pd.DataFrame({
        "feature": kept,
        "importance": tree.feature_importances_,
    }).sort_values("importance", ascending=False, kind="stable").reset_index(drop=True)

    rfe_count = min(selected_count, len(kept))
    rfe = RFE(
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        n_features_to_select=rfe_count,
        step=max(1, len(kept) // 10),
    ).fit(scaled, train["target"])
    rfe_ranking = pd.DataFrame({
        "feature": kept,
        "selected": rfe.support_,
        "rank": rfe.ranking_,
    }).sort_values(["rank", "feature"], kind="stable").reset_index(drop=True)
    selected = tuple(rfe_ranking.loc[rfe_ranking["selected"], "feature"])

    return ReductionResult(
        tuple(kept), tuple(dropped), selected, components_95,
        pca_variance, tree_importance, rfe_ranking,
    )


def plot_pca_variance(result: ReductionResult) -> plt.Figure:
    """Plot training-only cumulative explained variance and the 95% point."""
    fig, ax = plt.subplots(figsize=(7, 4))
    table = result.pca_variance
    ax.plot(table["component"], table["cumulative_variance"], marker=".")
    ax.axhline(0.95, color="tab:red", linestyle="--", label="95% variance")
    ax.axvline(result.pca_components_95, color="tab:gray", linestyle=":")
    ax.set(xlabel="Principal components", ylabel="Cumulative explained variance",
           title="Training-only standardized PCA")
    ax.legend()
    fig.tight_layout()
    return fig
