"""Methodology-correction utilities for uncertainty and nested evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from .eda import feature_columns
from .evaluate import binary_metrics


METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc")


def hep_operating_points(
    truth: np.ndarray, score: np.ndarray, targets: tuple[float, ...] = (0.90, 0.95)
) -> pd.DataFrame:
    """Return background efficiency/rejection at requested signal efficiencies."""
    truth = np.asarray(truth, dtype=int)
    score = np.asarray(score, dtype=float)
    if set(np.unique(truth)) != {0, 1} or len(truth) != len(score):
        raise ValueError("Aligned binary truth and score arrays are required")
    fpr, tpr, thresholds = roc_curve(truth, score)
    rows = []
    for target in targets:
        if not 0 < target <= 1:
            raise ValueError("Signal-efficiency targets must be in (0, 1]")
        index = int(np.flatnonzero(tpr >= target)[0])
        background_efficiency = float(fpr[index])
        rows.append({
            "target_signal_efficiency": target,
            "signal_efficiency": float(tpr[index]),
            "background_efficiency": background_efficiency,
            "background_rejection": (
                float("inf") if background_efficiency == 0
                else 1.0 / background_efficiency
            ),
            "threshold": float(thresholds[index]),
        })
    return pd.DataFrame(rows)


def paired_stratified_auc_bootstrap(
    truth: np.ndarray,
    scores: dict[str, np.ndarray],
    reference: str,
    additional_pairs: tuple[tuple[str, str], ...] = (),
    iterations: int = 2000,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bootstrap model AUCs and paired differences on identical resamples."""
    truth = np.asarray(truth, dtype=int)
    if reference not in scores or iterations < 100:
        raise ValueError("A reference score and at least 100 iterations are required")
    class_positions = [np.flatnonzero(truth == label) for label in (0, 1)]
    if any(len(group) < 2 for group in class_positions):
        raise ValueError("Both classes need at least two observations")
    checked = {name: np.asarray(value, dtype=float) for name, value in scores.items()}
    if any(len(value) != len(truth) for value in checked.values()):
        raise ValueError("Every score array must align with truth")

    rng = np.random.default_rng(random_state)
    samples = {name: np.empty(iterations) for name in checked}
    for iteration in range(iterations):
        positions = np.concatenate([
            rng.choice(group, size=len(group), replace=True)
            for group in class_positions
        ])
        sampled_truth = truth[positions]
        for name, value in checked.items():
            samples[name][iteration] = roc_auc_score(sampled_truth, value[positions])

    def interval(values: np.ndarray) -> tuple[float, float]:
        low, high = np.quantile(values, (0.025, 0.975))
        return float(low), float(high)

    auc_rows, difference_rows = [], []
    compared_pairs = [(reference, name) for name in checked if name != reference]
    compared_pairs.extend(additional_pairs)
    if any(left not in checked or right not in checked or left == right
           for left, right in compared_pairs):
        raise ValueError("Every requested comparison must contain two known models")
    for name, values in samples.items():
        low, high = interval(values)
        observed = roc_auc_score(truth, checked[name])
        auc_rows.append({
            "model": name, "roc_auc": observed,
            "ci_95_low": low, "ci_95_high": high,
            "bootstrap_iterations": iterations,
        })
    for left, right in dict.fromkeys(compared_pairs):
        differences = samples[left] - samples[right]
        observed_difference = (
            roc_auc_score(truth, checked[left]) - roc_auc_score(truth, checked[right])
        )
        difference_rows.append({
            "reference": left, "comparison": right,
            "auc_difference": observed_difference,
            "ci_95_low": interval(differences)[0],
            "ci_95_high": interval(differences)[1],
            "bootstrap_iterations": iterations,
        })
    return pd.DataFrame(auc_rows), pd.DataFrame(difference_rows)


def attach_process_type(raw: pd.DataFrame, cleaned: pd.DataFrame) -> pd.DataFrame:
    """Attach raw six-way process truth after strict positional validation."""
    if "type" not in raw or "target" not in cleaned or len(raw) != len(cleaned):
        raise ValueError("Raw and cleaned rows must align and contain truth columns")
    process = raw["type"].to_numpy()
    expected = np.isin(process, (0, 1)).astype(np.int8)
    if not np.array_equal(expected, cleaned["target"].to_numpy(dtype=np.int8)):
        raise ValueError("Raw process order does not match cleaned binary targets")
    result = cleaned.copy()
    result["type"] = process
    return result


def process_efficiencies(
    process_type: np.ndarray, score: np.ndarray, thresholds: pd.DataFrame
) -> pd.DataFrame:
    """Report predicted-signal acceptance separately for each generator process."""
    labels = {
        0: "charged_BB", 1: "neutral_BB", 2: "ccbar",
        3: "uubar", 4: "ddbar", 5: "ssbar",
    }
    process_type = np.asarray(process_type, dtype=int)
    score = np.asarray(score, dtype=float)
    if len(process_type) != len(score) or not set(np.unique(process_type)) <= set(labels):
        raise ValueError("Known aligned process types and scores are required")
    rows = []
    for threshold_row in thresholds.itertuples(index=False):
        for process, label in labels.items():
            mask = process_type == process
            if mask.any():
                rows.append({
                    "target_signal_efficiency": threshold_row.target_signal_efficiency,
                    "type": process, "process": label, "rows": int(mask.sum()),
                    "signal_acceptance": float(np.mean(score[mask] >= threshold_row.threshold)),
                })
    return pd.DataFrame(rows)


def classical_search_spaces(random_state: int = 42) -> dict[str, tuple[Any, dict[str, list[Any]]]]:
    """Return four classical families with exactly four candidate settings each."""
    return {
        "Logistic Regression": (
            Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=2000))]),
            {"model__C": [0.03, 0.1, 0.3, 1.0]},
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=random_state),
            {"max_depth": [3, 5, 8, None], "min_samples_leaf": [50]},
        ),
        "Linear SVM": (
            Pipeline([("scale", StandardScaler()), ("model", LinearSVC(dual="auto", max_iter=20_000))]),
            {"model__C": [0.03, 0.1, 0.3, 1.0]},
        ),
        "Histogram Gradient Boosting BDT": (
            HistGradientBoostingClassifier(
                learning_rate=0.05, max_iter=200, early_stopping=False,
                random_state=random_state,
            ),
            {"max_leaf_nodes": [15, 31], "min_samples_leaf": [30, 70]},
        ),
    }


def _fold_selected_features(
    x: pd.DataFrame, y: pd.Series, count: int, random_state: int
) -> list[str]:
    """Select features using only outer-training rows and a private holdout."""
    fit_positions, importance_positions = train_test_split(
        np.arange(len(x)), test_size=0.2, stratify=y, random_state=random_state
    )
    selector = HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=150, max_leaf_nodes=31,
        min_samples_leaf=50, random_state=random_state,
    ).fit(x.iloc[fit_positions], y.iloc[fit_positions])
    importance = permutation_importance(
        selector, x.iloc[importance_positions], y.iloc[importance_positions],
        scoring="roc_auc", n_repeats=3, random_state=random_state, n_jobs=-1,
    ).importances_mean
    order = np.argsort(-importance, kind="stable")[:count]
    return x.columns[order].tolist()


@dataclass(frozen=True)
class NestedStudy:
    metrics: pd.DataFrame
    selected_features: pd.DataFrame
    tuning: pd.DataFrame


def nested_development_study(
    events: pd.DataFrame,
    outer_folds: int = 5,
    inner_folds: int = 3,
    selected_count: int = 20,
    random_state: int = 42,
) -> NestedStudy:
    """Run equal-budget classical tuning and fold-local BDT reduction/PCA."""
    development = events.loc[events["split"].isin(("train", "validation"))].copy()
    features = feature_columns(development)
    x, y = development[features], development["target"]
    outer = StratifiedKFold(outer_folds, shuffle=True, random_state=random_state)
    truth = y.to_numpy(dtype=int)
    score_store: dict[tuple[str, str], np.ndarray] = {}
    feature_rows: list[dict[str, Any]] = []
    tuning_rows: list[dict[str, Any]] = []
    searches = classical_search_spaces(random_state)

    for fold, (fit_positions, holdout_positions) in enumerate(outer.split(x, y), start=1):
        x_fit, y_fit = x.iloc[fit_positions], y.iloc[fit_positions]
        x_holdout = x.iloc[holdout_positions]
        inner = StratifiedKFold(inner_folds, shuffle=True, random_state=random_state + fold)
        for name, (estimator, grid) in searches.items():
            search = GridSearchCV(estimator, grid, scoring="roc_auc", cv=inner, n_jobs=-1)
            search.fit(x_fit, y_fit)
            tuning_rows.append({
                "fold": fold, "model": name, "feature_set": "full",
                "candidate_count": len(search.cv_results_["params"]),
                "best_inner_roc_auc": float(search.best_score_),
                "best_parameters": json.dumps(search.best_params_, sort_keys=True),
                "pca_components": pd.NA,
            })
            key = (name, "full")
            score_store.setdefault(key, np.empty(len(x), dtype=float))[holdout_positions] = (
                search.decision_function(x_holdout)
                if hasattr(search, "decision_function")
                else search.predict_proba(x_holdout)[:, 1]
            )

        selected = _fold_selected_features(x_fit, y_fit, selected_count, random_state + fold)
        feature_rows.extend({"fold": fold, "rank": rank, "feature": feature}
                            for rank, feature in enumerate(selected, start=1))
        bdt, bdt_grid = searches["Histogram Gradient Boosting BDT"]
        selected_search = GridSearchCV(clone(bdt), bdt_grid, scoring="roc_auc", cv=inner, n_jobs=-1)
        selected_search.fit(x_fit[selected], y_fit)
        tuning_rows.append({
            "fold": fold, "model": "Histogram Gradient Boosting BDT",
            "feature_set": "nested_permutation_20",
            "candidate_count": len(selected_search.cv_results_["params"]),
            "best_inner_roc_auc": float(selected_search.best_score_),
            "best_parameters": json.dumps(selected_search.best_params_, sort_keys=True),
            "pca_components": pd.NA,
        })
        score_store.setdefault(("Histogram Gradient Boosting BDT", "nested_permutation_20"),
                               np.empty(len(x), dtype=float))[holdout_positions] = (
            selected_search.predict_proba(x_holdout[selected])[:, 1]
        )

        pca_pipeline = Pipeline([
            ("scale", StandardScaler()), ("pca", PCA(n_components=0.95, svd_solver="full")),
            ("model", clone(bdt)),
        ])
        pca_grid = {f"model__{key}": value for key, value in bdt_grid.items()}
        pca_search = GridSearchCV(pca_pipeline, pca_grid, scoring="roc_auc", cv=inner, n_jobs=-1)
        pca_search.fit(x_fit, y_fit)
        tuning_rows.append({
            "fold": fold, "model": "Histogram Gradient Boosting BDT",
            "feature_set": "nested_pca95",
            "candidate_count": len(pca_search.cv_results_["params"]),
            "best_inner_roc_auc": float(pca_search.best_score_),
            "best_parameters": json.dumps(pca_search.best_params_, sort_keys=True),
            "pca_components": int(
                pca_search.best_estimator_.named_steps["pca"].n_components_
            ),
        })
        score_store.setdefault(("Histogram Gradient Boosting BDT", "nested_pca95"),
                               np.empty(len(x), dtype=float))[holdout_positions] = (
            pca_search.predict_proba(x_holdout)[:, 1]
        )

    rows = []
    for (name, representation), score in score_store.items():
        prediction = (score >= 0.5).astype(int)
        rows.append({
            "model": name, "feature_set": representation,
            "evaluation": f"nested_{outer_folds}x{inner_folds}_development_oof",
            "rows": len(x), "outer_folds": outer_folds,
            "inner_folds": inner_folds, "candidates": 4,
            **binary_metrics(truth, prediction, score),
        })
    return NestedStudy(
        pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True),
        pd.DataFrame(feature_rows),
        pd.DataFrame(tuning_rows),
    )
