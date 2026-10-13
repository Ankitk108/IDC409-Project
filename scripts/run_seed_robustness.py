"""Run ten-seed development-only robustness for the representative best models."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config import DB_PATH
from src.db import read_table
from src.features import analyze_feature_reduction
from src.dense_nn import evaluate_deep_dense_cv
from src.robustness import (
    ROBUSTNESS_SEEDS,
    SeedRobustnessResult,
    evaluate_full_bdt_seed,
    plot_seed_robustness,
    summarize_seed_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--tables-dir", type=Path, default=Path("results/tables"))
    parser.add_argument("--figures-dir", type=Path, default=Path("results/figures"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("data/interim/seed_robustness_checkpoint.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    per_seed_path = args.tables_dir / "seed_robustness_per_seed.csv"
    summary_path = args.tables_dir / "seed_robustness_summary.csv"
    figure_path = args.figures_dir / "seed_robustness.png"
    outputs = (per_seed_path, summary_path, figure_path)
    existing = [path for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(
            "Seed-robustness artifacts exist; refusing to overwrite: "
            + ", ".join(map(str, existing))
        )

    events = read_table("cleaned_events", args.db)
    reduction = analyze_feature_reduction(events)
    expected_models = {
        "Histogram Gradient Boosting BDT (full)",
        "Deep Dense NN (RFE-20, post-hoc)",
    }
    if args.checkpoint.is_file():
        per_seed = pd.read_csv(args.checkpoint)
        if not {"model", "seed"} <= set(per_seed.columns):
            raise ValueError("Checkpoint is missing model/seed identity columns")
        observed_models = set(per_seed.get("model", []))
        observed_seeds = set(per_seed.get("seed", []))
        if (
            not observed_models <= expected_models
            or not observed_seeds <= set(ROBUSTNESS_SEEDS)
            or per_seed.duplicated(["model", "seed"]).any()
        ):
            raise ValueError("Checkpoint does not match the fixed Study C protocol")
        print(f"Resuming {len(per_seed)} completed model-seed results")
    else:
        per_seed = pd.DataFrame()
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)

    for seed in ROBUSTNESS_SEEDS:
        completed = set()
        if not per_seed.empty:
            completed = set(per_seed.loc[per_seed["seed"] == seed, "model"])
        bdt_name = "Histogram Gradient Boosting BDT (full)"
        if bdt_name not in completed:
            row = evaluate_full_bdt_seed(events, seed)
            per_seed = pd.concat([per_seed, pd.DataFrame([row])], ignore_index=True)
            per_seed.to_csv(args.checkpoint, index=False)
            print(f"Checkpointed {bdt_name}, seed={seed}")
        dense_name = "Deep Dense NN (RFE-20, post-hoc)"
        if dense_name not in completed:
            dense = evaluate_deep_dense_cv(
                events, reduction.selected_features, random_state=seed
            )
            row = dense.metrics.iloc[0].to_dict()
            row.update({"seed": seed, "folds": 5})
            per_seed = pd.concat([per_seed, pd.DataFrame([row])], ignore_index=True)
            per_seed.to_csv(args.checkpoint, index=False)
            print(f"Checkpointed {dense_name}, seed={seed}")

    if len(per_seed) != len(ROBUSTNESS_SEEDS) * len(expected_models):
        raise RuntimeError("Study C checkpoint is incomplete after evaluation")
    per_seed = per_seed.sort_values(["model", "seed"]).reset_index(drop=True)
    result = SeedRobustnessResult(per_seed, summarize_seed_metrics(per_seed))
    figure = plot_seed_robustness(result)

    args.tables_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    result.per_seed.to_csv(per_seed_path, index=False)
    result.summary.to_csv(summary_path, index=False)
    figure.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    print("Per-seed OOF metrics:")
    print(result.per_seed.to_string(index=False))
    print("\nAcross-seed summary:")
    print(result.summary.to_string(index=False))
    for path in outputs:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
