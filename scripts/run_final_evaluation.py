"""Run the frozen held-out protocol once and persist reviewable artifacts."""

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import pandas as pd

from src.config import DB_PATH
from src.db import read_table
from src.features import analyze_feature_reduction
from src.final_evaluation import (
    evaluate_frozen_test, plot_selected_confusion_matrices, plot_test_roc,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--tables-dir", type=Path, default=Path("results/tables"))
    parser.add_argument("--figures-dir", type=Path, default=Path("results/figures"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics_path = args.tables_dir / "final_test_metrics.csv"
    features_path = args.tables_dir / "final_rfe20_features.csv"
    roc_path = args.figures_dir / "final_test_roc.png"
    confusion_path = args.figures_dir / "final_test_confusion_matrices.png"
    outputs = (metrics_path, features_path, roc_path, confusion_path)
    existing = [path for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(
            "Final evaluation is one-shot; refusing to overwrite: "
            + ", ".join(map(str, existing))
        )

    events = read_table("cleaned_events", args.db)
    reduction = analyze_feature_reduction(events)
    result = evaluate_frozen_test(events, reduction.selected_features)
    roc = plot_test_roc(result)
    confusion = plot_selected_confusion_matrices(result)

    args.tables_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    result.metrics.to_csv(metrics_path, index=False)
    pd.DataFrame({"feature": reduction.selected_features}).to_csv(
        features_path, index=False
    )
    roc.savefig(roc_path, dpi=180, bbox_inches="tight")
    confusion.savefig(confusion_path, dpi=180, bbox_inches="tight")
    plt.close(roc)
    plt.close(confusion)

    print(result.metrics.to_string(index=False))
    print(f"Saved frozen features: {features_path}")
    print(f"Saved final metrics: {metrics_path}")
    print(f"Saved ROC plot: {roc_path}")
    print(f"Saved confusion matrices: {confusion_path}")


if __name__ == "__main__":
    main()
