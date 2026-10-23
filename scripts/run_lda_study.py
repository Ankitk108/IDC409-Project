"""Run Study D: ten-seed full-versus-reduced shrinkage LDA."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import DB_PATH
from src.db import read_table
from src.features import analyze_feature_reduction
from src.lda_evaluation import (
    evaluate_lda_study,
    plot_lda_diagnostics,
    plot_lda_discriminants,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--tables-dir", type=Path, default=Path("results/tables"))
    parser.add_argument("--figures-dir", type=Path, default=Path("results/figures"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    per_seed_path = args.tables_dir / "lda_per_seed.csv"
    summary_path = args.tables_dir / "lda_summary.csv"
    diagnostics_path = args.figures_dir / "lda_diagnostics.png"
    discriminants_path = args.figures_dir / "lda_discriminant_scores.png"
    outputs = (
        per_seed_path, summary_path, diagnostics_path, discriminants_path
    )
    existing = [path for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(
            "Study D artifacts exist; refusing to overwrite: "
            + ", ".join(map(str, existing))
        )

    events = read_table("cleaned_events", args.db)
    reduction = analyze_feature_reduction(events)
    result = evaluate_lda_study(events, reduction.selected_features)
    diagnostics = plot_lda_diagnostics(result)
    discriminants = plot_lda_discriminants(result)

    args.tables_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    result.per_seed.to_csv(per_seed_path, index=False)
    result.summary.to_csv(summary_path, index=False)
    diagnostics.savefig(diagnostics_path, dpi=180, bbox_inches="tight")
    discriminants.savefig(discriminants_path, dpi=180, bbox_inches="tight")
    plt.close(diagnostics)
    plt.close(discriminants)

    print("LDA seed-level OOF metrics:")
    print(result.per_seed.to_string(index=False))
    print("\nLDA across-seed summary:")
    print(result.summary.to_string(index=False))
    for path in outputs:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
