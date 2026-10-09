"""Run the reduced-feature Random Forest extension on development rows."""

from pathlib import Path
import argparse

import matplotlib.pyplot as plt

from src.config import DB_PATH
from src.db import read_table
from src.extensions import (
    evaluate_reduced_random_forest_cv,
    plot_reduced_random_forest_diagnostics,
)
from src.features import analyze_feature_reduction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--tables-dir", type=Path, default=Path("results/tables"))
    parser.add_argument("--figures-dir", type=Path, default=Path("results/figures"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics_path = args.tables_dir / "posthoc_reduced_random_forest_cv.csv"
    diagnostics_path = args.figures_dir / "posthoc_reduced_random_forest_diagnostics.png"
    outputs = (metrics_path, diagnostics_path)
    existing = [path for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(
            "Reduced Random Forest artifacts already exist; refusing to overwrite: "
            + ", ".join(map(str, existing))
        )

    events = read_table("cleaned_events", args.db)
    reduction = analyze_feature_reduction(events)
    result = evaluate_reduced_random_forest_cv(events, reduction.selected_features)
    diagnostics = plot_reduced_random_forest_diagnostics(result)

    args.tables_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    result.metrics.to_csv(metrics_path, index=False)
    diagnostics.savefig(diagnostics_path, dpi=180, bbox_inches="tight")
    plt.close(diagnostics)

    print(result.metrics.to_string(index=False))
    print(f"Frozen reduced features ({len(reduction.selected_features)}):")
    print("\n".join(reduction.selected_features))
    for path in outputs:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
