"""Run matched full-versus-reduced BDT evaluation on development rows."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import DB_PATH
from src.db import read_table
from src.extensions import (
    evaluate_reduced_bdt_comparison_cv,
    plot_reduced_bdt_comparison,
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
    metrics_path = args.tables_dir / "posthoc_bdt_feature_comparison_cv.csv"
    diagnostics_path = args.figures_dir / "posthoc_bdt_feature_comparison.png"
    outputs = (metrics_path, diagnostics_path)
    existing = [path for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(
            "Post-hoc BDT artifacts already exist; refusing to overwrite: "
            + ", ".join(map(str, existing))
        )

    events = read_table("cleaned_events", args.db)
    reduction = analyze_feature_reduction(events)
    result = evaluate_reduced_bdt_comparison_cv(
        events, reduction.selected_features
    )
    diagnostics = plot_reduced_bdt_comparison(result)

    args.tables_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    result.metrics.to_csv(metrics_path, index=False)
    diagnostics.savefig(diagnostics_path, dpi=180, bbox_inches="tight")
    plt.close(diagnostics)

    print(result.metrics.to_string(index=False))
    print(f"Frozen reduced features: {len(reduction.selected_features)}")
    for path in outputs:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
