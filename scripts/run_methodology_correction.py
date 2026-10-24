"""Run the post-hoc uncertainty audit and nested development-only correction."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold, learning_curve

from src.config import DB_PATH
from src.db import read_table
from src.features import analyze_feature_reduction
from src.final_evaluation import evaluate_frozen_test
from src.eda import feature_columns
from src.models import vishruti_models
from src.methodology import (
    attach_process_type,
    bootstrap_operating_points,
    duplicate_feature_audit,
    nested_development_study,
    paired_stratified_auc_bootstrap,
    process_auc,
    process_efficiencies,
    selection_stability,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument(
        "--stage", choices=("audit", "nested", "diagnostics", "all"), default="all"
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--tables-dir", type=Path, default=Path("results/tables"))
    parser.add_argument("--figures-dir", type=Path, default=Path("results/figures"))
    return parser.parse_args()


def refuse_overwrite(paths: list[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite methodology artifacts: " + ", ".join(existing))


def run_audit(args: argparse.Namespace, events: pd.DataFrame) -> None:
    auc_path = args.tables_dir / "methodology_test_auc_bootstrap.csv"
    difference_path = args.tables_dir / "methodology_test_auc_differences.csv"
    operating_path = args.tables_dir / "methodology_test_operating_points.csv"
    process_path = args.tables_dir / "methodology_test_process_efficiencies.csv"
    process_auc_path = args.tables_dir / "methodology_test_process_auc.csv"
    feature_audit_path = args.tables_dir / "methodology_feature_audit.csv"
    figure_path = args.figures_dir / "methodology_test_log_roc.png"
    outputs = [auc_path, difference_path, operating_path, process_path,
               process_auc_path, feature_audit_path, figure_path]
    refuse_overwrite(outputs)

    reduction = analyze_feature_reduction(events)
    frozen = evaluate_frozen_test(events, reduction.selected_features, include_cnn=False)
    aucs, differences = paired_stratified_auc_bootstrap(
        frozen.y_test, frozen.signal_scores,
        reference="Histogram Gradient Boosting BDT",
        additional_pairs=(("MLP (full)", "MLP (RFE-20)"),),
        iterations=args.bootstrap_iterations,
    )
    operating = bootstrap_operating_points(
        frozen.y_test, frozen.signal_scores,
        iterations=args.bootstrap_iterations,
    )

    raw = read_table("raw_events", args.db)
    typed = attach_process_type(raw, events)
    test_types = typed.loc[typed["split"] == "test", "type"].to_numpy()
    bdt_name = "Histogram Gradient Boosting BDT"
    bdt_points = operating.loc[operating["model"] == bdt_name]
    process = process_efficiencies(test_types, frozen.signal_scores[bdt_name], bdt_points)
    process.insert(0, "model", bdt_name)
    process_ranking = process_auc(test_types, frozen.signal_scores[bdt_name])
    process_ranking.insert(0, "model", bdt_name)
    feature_audit = duplicate_feature_audit(events, feature_columns(events))

    args.tables_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    aucs.to_csv(auc_path, index=False)
    differences.to_csv(difference_path, index=False)
    operating.to_csv(operating_path, index=False)
    process.to_csv(process_path, index=False)
    process_ranking.to_csv(process_auc_path, index=False)
    feature_audit.to_csv(feature_audit_path, index=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    for name, score in frozen.signal_scores.items():
        fpr, tpr, _ = roc_curve(frozen.y_test, score)
        ax.semilogx(fpr.clip(min=1 / len(frozen.y_test)), tpr, label=name)
    ax.set(
        xlabel="Background efficiency (log scale)", ylabel="Signal efficiency",
        title="Frozen test ROC at physics operating points", xlim=(1e-4, 1), ylim=(0, 1),
    )
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(aucs.to_string(index=False))
    print("\nPaired AUC differences:\n", differences.to_string(index=False))
    print("\nBDT process efficiencies:\n", process.to_string(index=False))
    print("\nBDT process-versus-signal AUC:\n", process_ranking.to_string(index=False))
    duplicates = feature_audit.loc[feature_audit["duplicate_of"] != ""]
    print("\nExact duplicate feature columns:\n", duplicates.to_string(index=False))


def run_nested(args: argparse.Namespace, events: pd.DataFrame) -> None:
    metrics_path = args.tables_dir / "methodology_nested_metrics.csv"
    features_path = args.tables_dir / "methodology_nested_selected_features.csv"
    tuning_path = args.tables_dir / "methodology_nested_tuning.csv"
    comparison_path = args.tables_dir / "methodology_nested_auc_differences.csv"
    frequency_path = args.tables_dir / "methodology_nested_selection_frequency.csv"
    overlap_path = args.tables_dir / "methodology_nested_selection_jaccard.csv"
    refuse_overwrite([metrics_path, features_path, tuning_path, comparison_path,
                      frequency_path, overlap_path])
    result = nested_development_study(events)
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    result.metrics.to_csv(metrics_path, index=False)
    result.selected_features.to_csv(features_path, index=False)
    result.tuning.to_csv(tuning_path, index=False)
    result.comparisons.to_csv(comparison_path, index=False)
    frequency, overlap = selection_stability(result.selected_features)
    frequency.to_csv(frequency_path, index=False)
    overlap.to_csv(overlap_path, index=False)
    print(result.metrics.to_string(index=False))
    print("\nFold-local feature frequencies:\n")
    print(result.selected_features["feature"].value_counts().to_string())
    print("\nFold-level selected hyperparameters:\n")
    print(result.tuning.to_string(index=False))
    print("\nNested paired AUC difference:\n", result.comparisons.to_string(index=False))
    print("\nSelection frequency:\n", frequency.to_string(index=False))
    print("\nPairwise Jaccard overlap:\n", overlap.to_string(index=False))


def run_diagnostics(args: argparse.Namespace, events: pd.DataFrame) -> None:
    """Create BDT calibration and development learning-curve evidence."""
    calibration_path = args.tables_dir / "methodology_bdt_calibration.csv"
    learning_path = args.tables_dir / "methodology_bdt_learning_curve.csv"
    figure_path = args.figures_dir / "methodology_bdt_diagnostics.png"
    refuse_overwrite([calibration_path, learning_path, figure_path])

    development = events.loc[events["split"].isin(("train", "validation"))].copy()
    test = events.loc[events["split"] == "test"].copy()
    features = feature_columns(development)
    model = vishruti_models()["Histogram Gradient Boosting BDT"]
    model.fit(development[features], development["target"])
    probability = model.predict_proba(test[features])[:, 1]
    observed, predicted = calibration_curve(
        test["target"], probability, n_bins=10, strategy="quantile"
    )
    calibration = pd.DataFrame({
        "mean_predicted_probability": predicted,
        "observed_signal_fraction": observed,
    })

    train_sizes, train_scores, validation_scores = learning_curve(
        vishruti_models()["Histogram Gradient Boosting BDT"],
        development[features], development["target"],
        train_sizes=np.array([0.10, 0.25, 0.50, 0.75, 1.00]),
        cv=StratifiedKFold(3, shuffle=True, random_state=42),
        scoring="roc_auc", shuffle=True, random_state=42, n_jobs=-1,
    )
    learning = pd.DataFrame({
        "training_rows": train_sizes,
        "train_roc_auc_mean": train_scores.mean(axis=1),
        "train_roc_auc_std": train_scores.std(axis=1, ddof=1),
        "validation_roc_auc_mean": validation_scores.mean(axis=1),
        "validation_roc_auc_std": validation_scores.std(axis=1, ddof=1),
    })

    args.tables_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    calibration.to_csv(calibration_path, index=False)
    learning.to_csv(learning_path, index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(predicted, observed, "o-", label="BDT")
    axes[0].plot([0, 1], [0, 1], "--", color="black", label="Perfect calibration")
    axes[0].set(xlabel="Mean predicted signal probability",
                ylabel="Observed signal fraction", title="Frozen-test calibration")
    axes[0].legend()
    axes[1].errorbar(
        learning["training_rows"], learning["validation_roc_auc_mean"],
        yerr=learning["validation_roc_auc_std"], marker="o", label="Validation",
    )
    axes[1].errorbar(
        learning["training_rows"], learning["train_roc_auc_mean"],
        yerr=learning["train_roc_auc_std"], marker="o", label="Training",
    )
    axes[1].set(xlabel="Development training rows", ylabel="ROC-AUC",
                title="BDT learning curve")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("Calibration:\n", calibration.to_string(index=False))
    print("\nLearning curve:\n", learning.to_string(index=False))


def main() -> None:
    args = parse_args()
    events = read_table("cleaned_events", args.db)
    if args.stage in ("audit", "all"):
        run_audit(args, events)
    if args.stage in ("nested", "all"):
        run_nested(args, events)
    if args.stage in ("diagnostics", "all"):
        run_diagnostics(args, events)


if __name__ == "__main__":
    main()
