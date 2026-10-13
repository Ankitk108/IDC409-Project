"""Development-only seed robustness for representative best models."""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from .dense_nn import evaluate_deep_dense_cv
from .eda import feature_columns
from .evaluate import binary_metrics
from .models import vishruti_models


METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc")
ROBUSTNESS_SEEDS = (42, 123, 2026, 7, 19, 73, 314, 2718, 4096, 10007)


@dataclass(frozen=True)
class SeedRobustnessResult:
    """Per-seed complete OOF metrics and across-seed summaries."""

    per_seed: pd.DataFrame
    summary: pd.DataFrame


def evaluate_full_bdt_seed(
    events: pd.DataFrame,
    seed: int,
    fold_count: int = 5,
    max_iter: int = 200,
) -> dict[str, float | int | str]:
    """Evaluate one full-feature BDT seed on development OOF predictions."""
    development = events.loc[events["split"].isin(("train", "validation"))].copy()
    if development.empty or set(development["target"].unique()) != {0, 1}:
        raise ValueError("Binary development rows must contain both classes")
    if fold_count < 2 or development["target"].value_counts().min() < fold_count:
        raise ValueError("Both classes must support the requested stratified folds")
    features = feature_columns(development)
    model = vishruti_models()["Histogram Gradient Boosting BDT"]
    model.set_params(random_state=seed, max_iter=max_iter)
    folds = StratifiedKFold(n_splits=fold_count, shuffle=True, random_state=seed)
    probability = cross_val_predict(
        model,
        development[features],
        development["target"],
        cv=folds,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]
    truth = development["target"].to_numpy()
    prediction = (probability >= 0.5).astype(int)
    return {
        "model": "Histogram Gradient Boosting BDT (full)",
        "feature_set": "full",
        "seed": seed,
        "folds": fold_count,
        "evaluation": f"development_{fold_count}fold_oof",
        "rows": len(development),
        "features": len(features),
        **binary_metrics(truth, prediction, probability),
    }


def summarize_seed_metrics(per_seed: pd.DataFrame) -> pd.DataFrame:
    """Summarize complete OOF repetitions without treating folds as replicates."""
    required = {"model", "seed", *METRICS}
    if not required <= set(per_seed.columns):
        raise ValueError(f"Per-seed table is missing: {sorted(required - set(per_seed))}")
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


def evaluate_seed_robustness(
    events: pd.DataFrame,
    selected_features: tuple[str, ...] | list[str],
    seeds: tuple[int, ...] = ROBUSTNESS_SEEDS,
    fold_count: int = 5,
) -> SeedRobustnessResult:
    """Run full BDT and reduced deep dense OOF evaluation for every seed."""
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("Provide at least two unique robustness seeds")
    rows = []
    for seed in seeds:
        rows.append(evaluate_full_bdt_seed(events, seed, fold_count))
        dense = evaluate_deep_dense_cv(
            events,
            selected_features,
            fold_count=fold_count,
            random_state=seed,
        )
        dense_row = dense.metrics.iloc[0].to_dict()
        dense_row.update({"seed": seed, "folds": fold_count})
        rows.append(dense_row)
    per_seed = pd.DataFrame(rows)
    return SeedRobustnessResult(per_seed, summarize_seed_metrics(per_seed))


def plot_seed_robustness(result: SeedRobustnessResult) -> plt.Figure:
    """Plot every seed plus mean and sample standard deviation by metric."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
    for axis, metric in zip(axes, ("roc_auc", "f1")):
        for model_name, group in result.per_seed.groupby("model", sort=False):
            short_name = "Full BDT" if "BDT" in model_name else "Deep dense RFE-20"
            axis.plot(group["seed"].astype(str), group[metric], "o-", label=short_name)
        axis.set_title(f"{metric.replace('_', ' ').upper()} by seed")
        axis.set_xlabel("Repeated-CV seed")
        axis.set_ylabel(metric.replace("_", " "))
        axis.grid(alpha=0.25)
    axes[0].legend()
    fig.suptitle("Study C: development-only 5-fold OOF robustness")
    fig.tight_layout()
    return fig
