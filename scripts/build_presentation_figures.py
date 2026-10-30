"""Build the slide-ready presentation figure package from reviewed artifacts.

This command does not fit or score any model.  It reads committed result tables,
the private instructor CSV for two EDA distributions, and existing reviewed PNGs.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
SOURCE_FIGURES = ROOT / "results" / "figures"
DEFAULT_DATA = ROOT / "data" / "raw" / "data_hep - data_hep.csv"
DEFAULT_OUTPUT = SOURCE_FIGURES / "presentation"

NAVY = "#17324D"
BLUE = "#2F6B9A"
CYAN = "#4AA3A2"
ORANGE = "#E07A3F"
RED = "#B94A48"
GREEN = "#3C8D6F"
LIGHT = "#F3F6F8"
MID = "#DCE5EB"
TEXT = "#17212B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--force", action="store_true", help="Replace the existing package."
    )
    return parser.parse_args()


def load_csv(name: str) -> pd.DataFrame:
    path = TABLES / name
    if not path.exists():
        raise FileNotFoundError(f"Required evidence table is missing: {path}")
    return pd.read_csv(path)


def prepare_output(path: Path, force: bool) -> None:
    resolved = path.resolve()
    expected_parent = SOURCE_FIGURES.resolve()
    if expected_parent not in resolved.parents:
        raise ValueError("Presentation output must stay under results/figures")
    if resolved.exists() and any(resolved.iterdir()):
        if not force:
            raise FileExistsError(
                f"{resolved} is not empty; rerun with --force to replace it"
            )
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)


def new_figure(title: str, subtitle: str = "") -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(13.333, 7.5), dpi=180)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.04, 0.94, title, fontsize=25, weight="bold", color=NAVY, va="top")
    if subtitle:
        ax.text(0.04, 0.885, subtitle, fontsize=12.5, color=BLUE, va="top")
    return fig, ax


def box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    body: str,
    color: str = BLUE,
    title_size: float = 15,
    body_size: float = 11,
) -> None:
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor="white", edgecolor=color, linewidth=2,
    )
    ax.add_patch(patch)
    ax.text(x + 0.02, y + height - 0.045, title, fontsize=title_size,
            weight="bold", color=color, va="top")
    ax.text(x + 0.02, y + height - 0.105, body, fontsize=body_size,
            color=TEXT, va="top", linespacing=1.4)


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=18,
                                linewidth=2, color=BLUE))


def save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def exact_row(frame: pd.DataFrame, **filters: object) -> pd.Series:
    selected = frame
    for column, value in filters.items():
        selected = selected.loc[selected[column] == value]
    if len(selected) != 1:
        raise ValueError(f"Expected one row for {filters}; found {len(selected)}")
    return selected.iloc[0]


def headline_values() -> dict[str, float]:
    auc = load_csv("methodology_test_auc_bootstrap.csv")
    operating = load_csv("methodology_test_operating_points.csv")
    nested_diff = load_csv("methodology_nested_auc_differences.csv")
    bdt_auc = exact_row(auc, model="Histogram Gradient Boosting BDT")
    bdt_90 = exact_row(
        operating,
        model="Histogram Gradient Boosting BDT",
        target_signal_efficiency=0.9,
    )
    reduction = nested_diff.iloc[0]
    return {
        "auc": bdt_auc["roc_auc"],
        "auc_low": bdt_auc["ci_95_low"],
        "auc_high": bdt_auc["ci_95_high"],
        "rejection": bdt_90["background_rejection_factor"],
        "rejection_low": bdt_90["rejection_factor_ci_95_low"],
        "rejection_high": bdt_90["rejection_factor_ci_95_high"],
        "reduction": reduction["auc_difference"],
        "reduction_low": reduction["ci_95_low"],
        "reduction_high": reduction["ci_95_high"],
    }


def slide_01(output: Path, values: dict[str, float]) -> None:
    fig, ax = new_figure(
        "Continuum Suppression with Belle II Event-Shape Variables",
        "Can simulated B-pair events be separated from continuum background?",
    )
    cards = [
        ("Test ROC-AUC", f"{values['auc']:.6f}\n[{values['auc_low']:.6f}, {values['auc_high']:.6f}]", BLUE),
        ("Rejection at 90% signal", f"{values['rejection']:.6f}×\n[{values['rejection_low']:.6f}, {values['rejection_high']:.6f}]", GREEN),
        ("20-feature nested cost", f"{values['reduction']:.6f} AUC\n[{values['reduction_low']:.6f}, {values['reduction_high']:.6f}]", ORANGE),
    ]
    for index, (title, body, color) in enumerate(cards):
        box(ax, 0.05 + index * 0.315, 0.47, 0.285, 0.27, title, body, color, 15, 16)
    stages = ["Physics", "Data", "Models", "Representation\n& performance", "Conclusion"]
    for index, label in enumerate(stages):
        x = 0.055 + index * 0.185
        ax.text(x, 0.23, label, ha="center", va="center", fontsize=12,
                weight="bold", color=NAVY if index == 0 else TEXT)
        if index < len(stages) - 1:
            arrow(ax, (x + 0.065, 0.23), (x + 0.12, 0.23))
    ax.text(0.5, 0.08,
            "Simulated, unweighted, row-split data • test already observed",
            ha="center", fontsize=11, color=RED, weight="bold")
    save(fig, output / "01_headline_and_flow.png")


def slide_02(output: Path) -> None:
    fig, ax = new_figure("Why event shape separates signal and continuum")
    box(ax, 0.05, 0.32, 0.36, 0.43, "Signal: B-pair events",
        "Less jet-like topology\nProcess flags 0–1\nBinary target = 1", GREEN, 17, 15)
    box(ax, 0.59, 0.32, 0.36, 0.43, "Background: continuum",
        "More two-jet-like topology\nccbar, uubar, ddbar, ssbar\nProcess flags 2–5 • target = 0", ORANGE, 17, 15)
    arrow(ax, (0.43, 0.26), (0.57, 0.26))
    ax.text(0.5, 0.29, "59 event-shape measurements", ha="center",
            fontsize=13, weight="bold", color=BLUE)
    ax.text(0.5, 0.22, "R2 • thrust • CLEO cones • KSFW moments", ha="center",
            fontsize=11, color=TEXT)
    ax.text(0.5, 0.11, "Measurements → continuous signal score → physics threshold",
            ha="center", fontsize=17, weight="bold", color=NAVY)
    save(fig, output / "02_physics_topology.png")


def slide_03(output: Path) -> None:
    fig, ax = new_figure("Dataset and preparation", "Instructor-provided simulated Belle II analysis-level CSV")
    stages = [
        ("Raw data", "70,606 rows\n59 measured features\n6 process flags"),
        ("Audit", "Numeric\nFinite\nComplete\nPreserve raw type"),
        ("Model table", "Drop Unnamed: 0\nExclude truth type\nCreate binary target"),
        ("Persist split", "SQLite\n70% train\n15% validation\n15% test"),
    ]
    for index, (title, body) in enumerate(stages):
        x = 0.035 + index * 0.245
        box(ax, x, 0.38, 0.205, 0.32, title, body,
            [BLUE, CYAN, ORANGE, GREEN][index], 15, 10)
        if index < len(stages) - 1:
            arrow(ax, (x + 0.208, 0.54), (x + 0.238, 0.54))
    ax.text(0.5, 0.20,
            "Limitations: row-level split • unweighted data • non-physical process mixture",
            ha="center", fontsize=14, color=RED, weight="bold")
    save(fig, output / "03_data_preparation.png")


def slide_04(output: Path, data_path: Path) -> None:
    if not data_path.exists():
        raise FileNotFoundError(f"Private instructor CSV is missing: {data_path}")
    data = pd.read_csv(data_path, usecols=["R2", "thrust", "type"])
    data["class"] = np.where(data["type"].isin([0, 1]), "Signal", "Continuum")
    fig, axes = plt.subplots(1, 2, figsize=(13.333, 7.5), dpi=180)
    fig.suptitle("Event-shape separation before multivariate modeling", fontsize=24,
                 weight="bold", color=NAVY, y=0.97)
    for ax, feature in zip(axes, ["R2", "thrust"]):
        for label, color in [("Signal", GREEN), ("Continuum", ORANGE)]:
            values = data.loc[data["class"] == label, feature]
            ax.hist(values, bins=45, density=True, histtype="step", linewidth=2.2,
                    label=label, color=color)
        ax.set_xlabel(feature, fontsize=14)
        ax.set_ylabel("Normalized density", fontsize=13)
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
    fig.text(0.5, 0.03,
             "R2-only test AUC 0.896210 [0.890198, 0.902247] • distributions overlap, motivating multivariate models",
             ha="center", fontsize=12, color=TEXT)
    save(fig, output / "04_eda_distributions.png")


def slide_05(output: Path) -> None:
    fig, ax = new_figure("How model quality is judged")
    box(ax, 0.06, 0.42, 0.38, 0.32, "Historical fixed-test evidence",
        "10,591 rows\nROC-AUC + stratified bootstrap CI\nPaired AUC differences\n90% and 95% signal operating points", BLUE, 17, 13)
    box(ax, 0.56, 0.42, 0.38, 0.32, "Development-only evidence",
        "60,015 rows\nOut-of-fold or nested CV\nFold-local tuning and selection\nNo new test claims", GREEN, 17, 13)
    ax.plot([0.5, 0.5], [0.34, 0.79], color=RED, linewidth=3)
    ax.text(0.5, 0.27, "Do not merge these protocols into one leaderboard",
            ha="center", fontsize=16, weight="bold", color=RED)
    ax.text(0.5, 0.15, "Rejection factor = 1 / background efficiency",
            ha="center", fontsize=16, color=NAVY)
    save(fig, output / "05_evaluation_protocols.png")


def slide_06(output: Path) -> None:
    fig, ax = new_figure("Model hypotheses and rough structures")
    groups = [
        ("Physics baseline", "R2 cut", "One measured variable", BLUE),
        ("Linear boundary", "Logistic • SVM • LDA", "Scaled linear scores", CYAN),
        ("Tree nonlinearities", "Tree • RF • histogram BDT", "Thresholds and interactions", GREEN),
        ("Neural nonlinearities", "MLP • deep dense • grouped CNN", "Learned nonlinear representation", ORANGE),
        ("PCA representation", "PCA95 → BDT", "Variance-preserving projection", RED),
    ]
    for index, (title, models, body, color) in enumerate(groups):
        row, col = divmod(index, 3)
        x = 0.04 + col * 0.32
        y = 0.50 - row * 0.30
        box(ax, x, y, 0.285, 0.22, title, f"{models}\n{body}", color, 13, 11)
    save(fig, output / "06_model_hypotheses.png")


def slide_07(output: Path) -> None:
    test = load_csv("final_test_metrics.csv")
    dense = load_csv("posthoc_dense_rfe20_cv.csv")
    forest = load_csv("posthoc_reduced_random_forest_cv.csv")
    lda = load_csv("lda_summary.csv")
    fixed_names = [
        "Histogram Gradient Boosting BDT", "MLP (full)", "MLP (RFE-20)",
        "Physics-grouped 1D CNN", "Logistic Regression", "Linear SVM",
        "Decision Tree", "R2 Cut",
    ]
    fixed = test.set_index("model").loc[fixed_names, "roc_auc"]
    lda_auc = lda.loc[lda["metric"] == "roc_auc"].set_index("model")["mean"]
    development = pd.Series({
        "Dense NN (RFE-20)": dense.iloc[0]["roc_auc"],
        "RF (RFE-20)": forest.iloc[0]["roc_auc"],
        "LDA (full)": lda_auc["Shrinkage LDA (full)"],
        "LDA (RFE-20)": lda_auc["Shrinkage LDA (RFE-20)"],
    })
    fig, axes = plt.subplots(1, 2, figsize=(13.333, 7.5), dpi=180)
    fig.suptitle("Model-family evidence — protocols shown separately", fontsize=23,
                 weight="bold", color=NAVY, y=0.98)
    for ax, series, title, color in [
        (axes[0], fixed.sort_values(), "Fixed test • 10,591 rows", BLUE),
        (axes[1], development.sort_values(), "Development OOF • 60,015 rows", GREEN),
    ]:
        ax.barh(series.index, series.values, color=color, alpha=0.85)
        ax.set_xlim(0.88, 0.96)
        ax.set_xlabel("ROC-AUC")
        ax.set_title(title, weight="bold", color=color)
        ax.grid(axis="x", alpha=0.2)
        for y, value in enumerate(series.values):
            ax.text(value + 0.001, y, f"{value:.6f}", va="center", fontsize=9)
    fig.text(0.5, 0.025, "Different protocols: do not rank the two panels as one leaderboard",
             ha="center", color=RED, weight="bold", fontsize=12)
    fig.subplots_adjust(left=0.19, right=0.97, bottom=0.13, top=0.86, wspace=0.34)
    save(fig, output / "07_model_evidence.png")


def slide_08(output: Path) -> None:
    metrics = load_csv("final_test_metrics.csv")
    seed = load_csv("seed_robustness_summary.csv")
    bdt = exact_row(metrics, model="Histogram Gradient Boosting BDT")
    seed_auc = exact_row(seed, model="Histogram Gradient Boosting BDT (full)", metric="roc_auc")
    fig, axes = plt.subplots(1, 2, figsize=(13.333, 7.5), dpi=180)
    fig.suptitle("Why the histogram BDT is the primary model", fontsize=24,
                 weight="bold", color=NAVY, y=0.98)
    labels = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    values = [bdt["accuracy"], bdt["precision"], bdt["recall"], bdt["f1"], bdt["roc_auc"]]
    axes[0].bar(labels, values, color=[BLUE, CYAN, GREEN, ORANGE, NAVY])
    axes[0].set_ylim(0.82, 0.96)
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].set_ylabel("Metric value")
    axes[0].grid(axis="y", alpha=0.2)
    for index, value in enumerate(values):
        axes[0].text(index, value + 0.003, f"{value:.6f}", ha="center", fontsize=9)
    per_seed = load_csv("seed_robustness_per_seed.csv")
    bdt_seed = per_seed.loc[per_seed["model"] == "Histogram Gradient Boosting BDT (full)"]
    axes[1].scatter(bdt_seed["seed"].astype(str), bdt_seed["roc_auc"], color=GREEN, s=55)
    axes[1].axhline(seed_auc["mean"], color=NAVY, linewidth=2,
                    label=f"mean {seed_auc['mean']:.6f}")
    axes[1].fill_between([-0.5, len(bdt_seed) - 0.5], seed_auc["minimum"],
                         seed_auc["maximum"], color=GREEN, alpha=0.12,
                         label=f"range {seed_auc['minimum']:.6f}–{seed_auc['maximum']:.6f}")
    axes[1].set_title("10-seed development OOF sensitivity", weight="bold")
    axes[1].set_xlabel("Seed")
    axes[1].set_ylabel("ROC-AUC")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].legend(frameon=False, fontsize=9)
    axes[1].grid(axis="y", alpha=0.2)
    fig.text(0.75, 0.04, f"sample SD {seed_auc['std']:.6f} • not a confidence interval",
             ha="center", color=RED, weight="bold", fontsize=11)
    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.22, top=0.86, wspace=0.20)
    save(fig, output / "08_bdt_metrics_and_seed_robustness.png")


def slide_09(output: Path) -> None:
    nested = load_csv("methodology_nested_metrics.csv")
    nested = nested.loc[
        nested["model"] == "Histogram Gradient Boosting BDT"
    ].copy()
    order = ["full", "nested_permutation_20", "nested_pca95"]
    labels = ["Full 59", "Permutation 20", "PCA95"]
    selected = nested.set_index("feature_set").loc[order]
    fig, ax = plt.subplots(figsize=(13.333, 7.5), dpi=180)
    values = selected["fold_roc_auc_mean"].to_numpy()
    errors = selected["fold_roc_auc_std"].to_numpy()
    bars = ax.bar(labels, values, yerr=errors, capsize=8,
                  color=[NAVY, GREEN, ORANGE], alpha=0.9)
    ax.set_ylim(0.90, 0.955)
    ax.set_ylabel("Mean outer-fold ROC-AUC", fontsize=14)
    ax.set_title("Which feature representation preserves discrimination?",
                 fontsize=24, weight="bold", color=NAVY, pad=20)
    ax.grid(axis="y", alpha=0.2)
    for bar, value, error in zip(bars, values, errors):
        ax.text(bar.get_x() + bar.get_width() / 2, value + error + 0.001,
                f"{value:.6f} ± {error:.6f}", ha="center", fontsize=11)
    ax.text(0.5, 0.905,
            "Full − permutation-20 pooled AUC: 0.000639 [0.000286, 0.000966]\n"
            "16 features selected in every fold • Jaccard 0.739130–0.904762",
            ha="center", va="bottom", fontsize=11, color=TEXT,
            bbox={"facecolor": LIGHT, "edgecolor": MID, "boxstyle": "round,pad=0.5"})
    save(fig, output / "09_feature_representation.png")


def slide_10(output: Path) -> None:
    source = SOURCE_FIGURES / "methodology_test_log_roc.png"
    if not source.exists():
        raise FileNotFoundError(source)
    shutil.copy2(source, output / "10_physics_operating_points.png")


def slide_11(output: Path) -> None:
    process = load_csv("methodology_test_process_auc.csv")
    calibration = load_csv("methodology_bdt_calibration.csv")
    learning = load_csv("methodology_bdt_learning_curve.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13.333, 7.5), dpi=180)
    fig.suptitle("Process consistency, calibration, and data sufficiency",
                 fontsize=22, weight="bold", color=NAVY, y=0.98)
    axes[0].bar(process["process"], process["roc_auc"], color=GREEN)
    axes[0].set_ylim(0.93, 0.952)
    axes[0].set_ylabel("Test ROC-AUC")
    axes[0].set_title("Continuum flavour", weight="bold")
    for i, row in process.reset_index(drop=True).iterrows():
        axes[0].text(i, row["roc_auc"] + 0.0005,
                     f"{row['roc_auc']:.6f}\nn={int(row['background_rows']):,}",
                     ha="center", fontsize=8)
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="grey")
    axes[1].plot(calibration["mean_predicted_probability"],
                 calibration["observed_signal_fraction"], marker="o", color=BLUE)
    axes[1].set_xlabel("Mean predicted probability")
    axes[1].set_ylabel("Observed signal fraction")
    axes[1].set_title("Calibration", weight="bold")
    axes[2].errorbar(learning["training_rows"], learning["validation_roc_auc_mean"],
                     yerr=learning["validation_roc_auc_std"], marker="o",
                     capsize=4, color=ORANGE)
    axes[2].set_xlabel("Training rows")
    axes[2].set_ylabel("Validation ROC-AUC")
    axes[2].set_title("Learning curve", weight="bold")
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.text(0.5, 0.025,
             "Lowest calibration bin: predicted 0.009479, observed 0.002830 • ddbar/ssbar samples are small",
             ha="center", fontsize=10.5, color=RED, weight="bold")
    save(fig, output / "11_process_and_diagnostics.png")


def slide_12(output: Path) -> None:
    fig, ax = new_figure("Recommendation and limitations")
    box(ax, 0.04, 0.47, 0.285, 0.28, "Primary", "Full 59-feature histogram BDT\nStrongest fixed-test evidence", NAVY, 17, 13)
    box(ax, 0.355, 0.47, 0.285, 0.28, "Compact alternative",
        "Nested permutation-selected\n20-feature BDT\n39 fewer measurements",
        GREEN, 17, 11)
    box(ax, 0.67, 0.47, 0.285, 0.28, "Physics baseline", "R2 cut\nTest AUC 0.896210", ORANGE, 17, 13)
    limitations = (
        "Simulated and unweighted • non-physical process mixture • row-level split • "
        "already-observed test • unequal tuning budgets • no delta E or Mbc • "
        "deep dense model near epoch cap"
    )
    ax.text(0.5, 0.25, limitations, ha="center", va="center", wrap=True,
            fontsize=13, color=RED, weight="bold",
            bbox={"facecolor": "#FFF5F4", "edgecolor": RED, "boxstyle": "round,pad=0.7"})
    save(fig, output / "12_recommendation_and_limits.png")


def slide_13(output: Path, values: dict[str, float]) -> None:
    fig, ax = new_figure("Final answer")
    cards = [
        ("Discrimination", f"BDT AUC {values['auc']:.6f}\n[{values['auc_low']:.6f}, {values['auc_high']:.6f}]", BLUE),
        ("Physics operation", f"{values['rejection']:.6f}× rejection\nat 90% signal efficiency", GREEN),
        ("Compact representation", f"20 features cost\n{values['reduction']:.6f} AUC", ORANGE),
    ]
    for index, (title, body, color) in enumerate(cards):
        box(ax, 0.05 + index * 0.315, 0.46, 0.285, 0.28, title, body, color, 16, 15)
    ax.text(0.5, 0.23,
            "Next: independent event-grouped or collision-data validation → physical weights → data/MC checks → delta E and Mbc decorrelation",
            ha="center", va="center", wrap=True, fontsize=14, weight="bold", color=NAVY)
    save(fig, output / "13_conclusion.png")


def copy_supporting_figures(output: Path) -> list[str]:
    copies = {
        "08b_test_confusion_matrices.png": "final_test_confusion_matrices.png",
        "11b_bdt_calibration_and_learning.png": "methodology_bdt_diagnostics.png",
        "A1_seed_robustness_detail.png": "seed_robustness.png",
        "A2_random_forest_diagnostics.png": "posthoc_reduced_random_forest_diagnostics.png",
        "A3_deep_dense_diagnostics.png": "posthoc_dense_rfe20_diagnostics.png",
        "A4_lda_diagnostics.png": "lda_diagnostics.png",
        "A5_lda_discriminant_scores.png": "lda_discriminant_scores.png",
    }
    for destination, source_name in copies.items():
        source = SOURCE_FIGURES / source_name
        if not source.exists():
            raise FileNotFoundError(source)
        shutil.copy2(source, output / destination)
    return list(copies)


def write_manifest(output: Path, supporting: list[str]) -> None:
    rows = [
        (1, "01_headline_and_flow.png", "Bootstrap, operating-point, nested-difference CSVs", "Opening and roadmap"),
        (2, "02_physics_topology.png", "Physics contract", "Conceptual diagram"),
        (3, "03_data_preparation.png", "Instructor dataset contract", "Data pipeline"),
        (4, "04_eda_distributions.png", "Private instructor CSV", "Normalized R2 and thrust distributions"),
        (5, "05_evaluation_protocols.png", "Methodology contract", "Protocol separation"),
        (6, "06_model_hypotheses.png", "Implemented model contracts", "Model-family map"),
        (7, "07_model_evidence.png", "Final, extension, and LDA result CSVs", "Protocol-separated results"),
        (8, "08_bdt_metrics_and_seed_robustness.png", "Final metrics and seed robustness CSVs", "Primary-model evidence"),
        (9, "09_feature_representation.png", "Nested metrics and differences CSVs", "Nested representation study"),
        (10, "10_physics_operating_points.png", "Reviewed methodology ROC artifact", "Fixed-test operating curve"),
        (11, "11_process_and_diagnostics.png", "Process, calibration, and learning CSVs", "Diagnostics"),
        (12, "12_recommendation_and_limits.png", "Reviewed project evidence", "Recommendation"),
        (13, "13_conclusion.png", "Headline evidence CSVs", "Conclusion"),
    ]
    manifest = pd.DataFrame(rows, columns=["slide", "file", "source", "purpose"])
    manifest.to_csv(output / "manifest.csv", index=False)
    (output / "supporting_files.txt").write_text("\n".join(supporting) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    data_path = args.data if args.data.is_absolute() else ROOT / args.data
    prepare_output(output, args.force)
    values = headline_values()
    slide_01(output, values)
    slide_02(output)
    slide_03(output)
    slide_04(output, data_path)
    slide_05(output)
    slide_06(output)
    slide_07(output)
    slide_08(output)
    slide_09(output)
    slide_10(output)
    slide_11(output)
    slide_12(output)
    slide_13(output, values)
    supporting = copy_supporting_figures(output)
    write_manifest(output, supporting)
    print(f"Presentation package written to: {output}")
    print("Central slide figures: 13")
    print(f"Supporting figures: {len(supporting)}")


if __name__ == "__main__":
    main()
