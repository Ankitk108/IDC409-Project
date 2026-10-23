"""Ten-seed development-only evaluation for full and reduced shrinkage LDA."""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.model_selection import StratifiedKFold

from .evaluate import binary_metrics
from .lda import lda_classifier, lda_development_feature_sets
from .robustness import METRICS, ROBUSTNESS_SEEDS


@dataclass(frozen=True)
class LDAStudyResult:
    """Seed metrics plus representative OOF inputs for diagnostic plots."""

    per_seed: pd.DataFrame
    summary: pd.DataFrame
    truth: np.ndarray
    probabilities: dict[str, np.ndarray]
    predictions: dict[str, np.ndarray]
    discriminant_scores: dict[str, np.ndarray]


def summarize_lda_seed_metrics(per_seed: pd.DataFrame) -> pd.DataFrame:
    """Summarize complete seed-level OOF metrics for each LDA feature set."""
    required = {"model", "seed", *METRICS}
    if not required <= set(per_seed.columns):
        raise ValueError(f"LDA seed table is missing: {sorted(required-set(per_seed))}")
    rows = []
    for model_name, group in per_seed.groupby("model", sort=False):
        for metric in METRICS:
            values = group[metric].astype(float)
            rows.append({
                "model": model_name,
                "metric": metric,
                "mean": values.mean(),
                "std": values.std(ddof=1),
                "minimum": values.min(),
                "maximum": values.max(),
                "seeds": len(values),
            })
    return pd.DataFrame(rows)


def evaluate_lda_study(
    events: pd.DataFrame,
    selected_features: tuple[str, ...] | list[str],
    seeds: tuple[int, ...] = ROBUSTNESS_SEEDS,
    fold_count: int = 5,
) -> LDAStudyResult:
    """Evaluate full and RFE LDA on identical repeated development OOF folds."""
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("Provide at least two unique LDA seeds")
    development, feature_sets = lda_development_feature_sets(
        events, selected_features
    )
    if fold_count < 2 or development["target"].value_counts().min() < fold_count:
        raise ValueError("Both classes must support the requested LDA folds")
    truth = development["target"].to_numpy(dtype=int)
    model_names = {
        "full": "Shrinkage LDA (full)",
        "reduced_rfe20": "Shrinkage LDA (RFE-20)",
    }
    rows = []
    representative_probabilities = {}
    representative_predictions = {}
    representative_discriminants = {}

    for seed in seeds:
        folds = list(StratifiedKFold(
            n_splits=fold_count, shuffle=True, random_state=seed
        ).split(development, truth))
        for feature_set, features in feature_sets.items():
            probabilities = np.empty(len(development), dtype=float)
            discriminants = np.empty(len(development), dtype=float)
            for fit_positions, holdout_positions in folds:
                model = lda_classifier()
                model.fit(
                    development.iloc[fit_positions][features],
                    truth[fit_positions],
                )
                holdout = development.iloc[holdout_positions][features]
                probabilities[holdout_positions] = model.predict_proba(holdout)[:, 1]
                discriminants[holdout_positions] = model.decision_function(holdout)
            predictions = (probabilities >= 0.5).astype(int)
            model_name = model_names[feature_set]
            rows.append({
                "model": model_name,
                "feature_set": feature_set,
                "seed": seed,
                "folds": fold_count,
                "evaluation": f"development_{fold_count}fold_oof",
                "rows": len(development),
                "features": len(features),
                **binary_metrics(truth, predictions, probabilities),
            })
            if seed == seeds[0]:
                representative_probabilities[model_name] = probabilities.copy()
                representative_predictions[model_name] = predictions.copy()
                representative_discriminants[model_name] = discriminants.copy()

    per_seed = pd.DataFrame(rows)
    return LDAStudyResult(
        per_seed=per_seed,
        summary=summarize_lda_seed_metrics(per_seed),
        truth=truth,
        probabilities=representative_probabilities,
        predictions=representative_predictions,
        discriminant_scores=representative_discriminants,
    )


def plot_lda_diagnostics(result: LDAStudyResult) -> plt.Figure:
    """Plot representative-seed ROC and confusion matrices for both LDA inputs."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for model_name, probability in result.probabilities.items():
        RocCurveDisplay.from_predictions(
            result.truth, probability, name=model_name, ax=axes[0]
        )
    axes[0].plot([0, 1], [0, 1], color="black", linestyle="--")
    axes[0].set_title("Representative-seed development OOF ROC")
    for axis, (model_name, prediction) in zip(
        axes[1:], result.predictions.items()
    ):
        ConfusionMatrixDisplay.from_predictions(
            result.truth,
            prediction,
            display_labels=["background", "signal"],
            colorbar=False,
            ax=axis,
        )
        axis.set_title(model_name)
    fig.tight_layout()
    return fig


def plot_lda_discriminants(result: LDAStudyResult) -> plt.Figure:
    """Show the single binary discriminant score by physical class."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for axis, (model_name, score) in zip(
        axes, result.discriminant_scores.items()
    ):
        lower, upper = np.quantile(score, (0.005, 0.995))
        axis.hist(
            score[result.truth == 0], bins=50, density=True, alpha=0.55,
            range=(lower, upper), label="background",
        )
        axis.hist(
            score[result.truth == 1], bins=50, density=True, alpha=0.55,
            range=(lower, upper), label="signal",
        )
        axis.axvline(0, color="black", linestyle="--", linewidth=1)
        axis.set_xlim(lower, upper)
        axis.set_title(model_name)
        axis.set_xlabel("OOF linear discriminant score")
        axis.set_ylabel("Density")
        axis.legend()
        axis.text(
            0.02, 0.96, "Central 99% shown", transform=axis.transAxes,
            va="top", fontsize=9,
        )
    fig.suptitle("Study D: one-dimensional LDA separation")
    fig.tight_layout()
    return fig
