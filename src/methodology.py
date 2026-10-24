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
    """Interpolate background efficiency at exact signal-efficiency targets."""
    truth = np.asarray(truth, dtype=int)
    score = np.asarray(score, dtype=float)
    if set(np.unique(truth)) != {0, 1} or len(truth) != len(score):
        raise ValueError("Aligned binary truth and score arrays are required")
    fpr, tpr, thresholds = roc_curve(truth, score)
    rows = []
    for target in targets:
        if not 0 < target <= 1:
            raise ValueError("Signal-efficiency targets must be in (0, 1]")
        background_efficiency = float(np.interp(target, tpr, fpr))
        threshold = float(np.interp(target, tpr, thresholds))
        rows.append({
            "target_signal_efficiency": target,
            "signal_efficiency": target,
            "background_efficiency": background_efficiency,
            "background_rejection_fraction": 1.0 - background_efficiency,
            "background_rejection_factor": (
                float("inf") if background_efficiency == 0
                else 1.0 / background_efficiency
            ),
            "threshold": threshold,
        })
    return pd.DataFrame(rows)


def bootstrap_operating_points(
    truth: np.ndarray,
    scores: dict[str, np.ndarray],
    targets: tuple[float, ...] = (0.90, 0.95),
    iterations: int = 2000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Add stratified-bootstrap intervals to fixed-efficiency rejection factors."""
    truth = np.asarray(truth, dtype=int)
    checked = {name: np.asarray(score, dtype=float) for name, score in scores.items()}
    if iterations < 100 or set(np.unique(truth)) != {0, 1}:
        raise ValueError("Binary truth and at least 100 bootstrap iterations are required")
    if any(len(score) != len(truth) for score in checked.values()):
        raise ValueError("Every score array must align with truth")
    class_positions = [np.flatnonzero(truth == label) for label in (0, 1)]
    rng = np.random.default_rng(random_state)
    samples = {(name, target): np.empty(iterations) for name in checked for target in targets}
    for iteration in range(iterations):
        positions = np.concatenate([
            rng.choice(group, size=len(group), replace=True) for group in class_positions
        ])
        for name, score in checked.items():
            points = hep_operating_points(truth[positions], score[positions], targets)
            for row in points.itertuples(index=False):
                samples[(name, row.target_signal_efficiency)][iteration] = (
                    row.background_rejection_factor
                )
    rows = []
    for name, score in checked.items():
        observed = hep_operating_points(truth, score, targets)
        for point in observed.itertuples(index=False):
            values = samples[(name, point.target_signal_efficiency)]
            low, high = np.quantile(values, (0.025, 0.975))
            rows.append({
                "model": name,
                **point._asdict(),
                "rejection_factor_ci_95_low": float(low),
                "rejection_factor_ci_95_high": float(high),
                "bootstrap_iterations": iterations,
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


def process_auc(process_type: np.ndarray, score: np.ndarray) -> pd.DataFrame:
    """Compare all BB signal rows with each continuum generator process."""
    names = {2: "ccbar", 3: "uubar", 4: "ddbar", 5: "ssbar"}
    process_type = np.asarray(process_type, dtype=int)
    score = np.asarray(score, dtype=float)
    if len(process_type) != len(score):
        raise ValueError("Process types and scores must align")
    signal = np.isin(process_type, (0, 1))
    rows = []
    for process, name in names.items():
        selected = signal | (process_type == process)
        truth = signal[selected].astype(int)
        rows.append({
            "type": process,
            "process": name,
            "signal_rows": int(signal.sum()),
            "background_rows": int(np.sum(process_type == process)),
            "roc_auc": float(roc_auc_score(truth, score[selected])),
        })
    return pd.DataFrame(rows)


def duplicate_feature_audit(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Report exact duplicate feature columns and zero fractions."""
    values = frame[features]
    duplicate = values.T.duplicated(keep="first")
    first_match: dict[str, str] = {}
    for column in values.columns[duplicate]:
        first_match[column] = next(
            previous for previous in values.columns
            if previous != column and values[previous].equals(values[column])
        )
    return pd.DataFrame({
        "feature": features,
        "duplicate_of": [first_match.get(feature, "") for feature in features],
        "zero_fraction": [(values[feature] == 0).mean() for feature in features],
    })


def selection_stability(selected_features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return selection frequencies and pairwise Jaccard overlap across folds."""
    required = {"fold", "feature"}
    if not required <= set(selected_features):
        raise ValueError("Fold and feature columns are required")
    folds = {
        int(fold): set(group["feature"])
        for fold, group in selected_features.groupby("fold", sort=True)
    }
    if len(folds) < 2 or any(not features for features in folds.values()):
        raise ValueError("At least two non-empty fold selections are required")
    frequency = (selected_features.groupby("feature")["fold"].nunique()
                 .rename("selected_folds").reset_index())
    frequency["selection_frequency"] = frequency["selected_folds"] / len(folds)
    pairs = []
    fold_ids = sorted(folds)
    for left_index, left in enumerate(fold_ids):
        for right in fold_ids[left_index + 1:]:
            union = folds[left] | folds[right]
            pairs.append({"fold_a": left, "fold_b": right,
                          "jaccard": len(folds[left] & folds[right]) / len(union)})
    return (frequency.sort_values(["selected_folds", "feature"], ascending=[False, True]),
            pd.DataFrame(pairs))


def unified_results_table(
    test_auc: pd.DataFrame,
    nested_metrics: pd.DataFrame,
    operating_points: pd.DataFrame,
) -> pd.DataFrame:
    """Join compatible headline evidence without filling unavailable protocols."""
    required_auc = {"model", "roc_auc", "ci_95_low", "ci_95_high"}
    required_nested = {"model", "feature_set", "fold_roc_auc_mean"}
    if not required_auc <= set(test_auc) or not required_nested <= set(nested_metrics):
        raise ValueError("Expected test-AUC and fold-level nested columns are missing")
    rows = []
    for test_row in test_auc.itertuples(index=False):
        model = test_row.model
        nested = nested_metrics.loc[
            (nested_metrics["model"] == model) & (nested_metrics["feature_set"] == "full")
        ]
        points = operating_points.loc[operating_points["model"] == model]
        result = {
            "model": model,
            "test_auc_ci": (f"{test_row.roc_auc:.3f} "
                            f"[{test_row.ci_95_low:.3f}, {test_row.ci_95_high:.3f}]"),
            "nested_dev_auc_mean": (float(nested.iloc[0]["fold_roc_auc_mean"])
                                    if len(nested) == 1 else np.nan),
        }
        for target in (0.90, 0.95):
            point = points.loc[np.isclose(points["target_signal_efficiency"], target)]
            label = f"rejection_factor_at_{int(target * 100)}"
            result[label] = (
                f"{point.iloc[0]['background_rejection_factor']:.1f} "
                f"[{point.iloc[0]['rejection_factor_ci_95_low']:.1f}, "
                f"{point.iloc[0]['rejection_factor_ci_95_high']:.1f}]"
                if len(point) == 1 and test_row.roc_auc > 0.5 + 1e-12 else pd.NA
            )
        rows.append(result)
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
    comparisons: pd.DataFrame


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
    fold_auc_rows: list[dict[str, Any]] = []
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
            fold_score = (
                search.decision_function(x_holdout)
                if hasattr(search, "decision_function")
                else search.predict_proba(x_holdout)[:, 1]
            )
            score_store.setdefault(key, np.empty(len(x), dtype=float))[holdout_positions] = fold_score
            fold_auc_rows.append({"fold": fold, "model": name, "feature_set": "full",
                                  "roc_auc": roc_auc_score(y.iloc[holdout_positions], fold_score)})

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
        selected_score = selected_search.predict_proba(x_holdout[selected])[:, 1]
        score_store.setdefault(("Histogram Gradient Boosting BDT", "nested_permutation_20"),
                               np.empty(len(x), dtype=float))[holdout_positions] = selected_score
        fold_auc_rows.append({"fold": fold, "model": "Histogram Gradient Boosting BDT",
                              "feature_set": "nested_permutation_20",
                              "roc_auc": roc_auc_score(y.iloc[holdout_positions], selected_score)})

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
        pca_score = pca_search.predict_proba(x_holdout)[:, 1]
        score_store.setdefault(("Histogram Gradient Boosting BDT", "nested_pca95"),
                               np.empty(len(x), dtype=float))[holdout_positions] = pca_score
        fold_auc_rows.append({"fold": fold, "model": "Histogram Gradient Boosting BDT",
                              "feature_set": "nested_pca95",
                              "roc_auc": roc_auc_score(y.iloc[holdout_positions], pca_score)})

    rows = []
    fold_aucs = pd.DataFrame(fold_auc_rows)
    for (name, representation), score in score_store.items():
        prediction = (score >= 0.5).astype(int)
        relevant = fold_aucs.loc[(fold_aucs["model"] == name) &
                                 (fold_aucs["feature_set"] == representation), "roc_auc"]
        rows.append({
            "model": name, "feature_set": representation,
            "evaluation": f"nested_{outer_folds}x{inner_folds}_development_oof",
            "rows": len(x), "outer_folds": outer_folds,
            "inner_folds": inner_folds, "candidates": 4,
            "fold_roc_auc_mean": float(relevant.mean()),
            "fold_roc_auc_std": float(relevant.std(ddof=1)),
            **binary_metrics(truth, prediction, score),
        })
    _, comparisons = paired_stratified_auc_bootstrap(
        truth,
        {
            "BDT full": score_store[("Histogram Gradient Boosting BDT", "full")],
            "BDT nested permutation-20": score_store[
                ("Histogram Gradient Boosting BDT", "nested_permutation_20")
            ],
        },
        reference="BDT full",
        iterations=2000,
        random_state=random_state,
    )
    comparisons.insert(0, "evaluation", f"nested_{outer_folds}x{inner_folds}_development_oof")
    return NestedStudy(
        pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True),
        pd.DataFrame(feature_rows),
        pd.DataFrame(tuning_rows),
        comparisons,
    )
