"""Fast checks for the development-only seed robustness study."""

import unittest

import pandas as pd

from src.robustness import (
    ROBUSTNESS_SEEDS,
    evaluate_full_bdt_seed,
    summarize_seed_metrics,
)


class SeedRobustnessTests(unittest.TestCase):
    def setUp(self):
        labels = [0, 1] * 30
        self.events = pd.DataFrame({
            "R2": [0.8 if label == 0 else 0.2 for label in labels],
            "thrust": [0.9 if label == 0 else 0.5 for label in labels],
            "target": labels,
            "split": ["train"] * 40 + ["validation"] * 10 + ["test"] * 10,
        })

    def test_bdt_seed_excludes_original_test_rows(self):
        result = evaluate_full_bdt_seed(
            self.events, seed=42, fold_count=2, max_iter=2
        )
        self.assertEqual(result["rows"], 50)
        self.assertEqual(result["features"], 2)
        self.assertEqual(result["seed"], 42)

    def test_summary_uses_seed_level_values(self):
        per_seed = pd.DataFrame({
            "model": ["A", "A", "A"],
            "seed": [42, 123, 2026],
            "accuracy": [0.8, 0.9, 1.0],
            "precision": [0.8, 0.9, 1.0],
            "recall": [0.8, 0.9, 1.0],
            "f1": [0.8, 0.9, 1.0],
            "roc_auc": [0.8, 0.9, 1.0],
        })
        summary = summarize_seed_metrics(per_seed)
        auc = summary.loc[summary["metric"] == "roc_auc"].iloc[0]
        self.assertAlmostEqual(auc["mean"], 0.9)
        self.assertAlmostEqual(auc["std"], 0.1)
        self.assertEqual(auc["seeds"], 3)

    def test_summary_rejects_incomplete_metrics(self):
        with self.assertRaises(ValueError):
            summarize_seed_metrics(pd.DataFrame({"model": ["A"], "seed": [42]}))

    def test_protocol_contains_ten_unique_seeds(self):
        self.assertEqual(len(ROBUSTNESS_SEEDS), 10)
        self.assertEqual(len(set(ROBUSTNESS_SEEDS)), 10)


if __name__ == "__main__":
    unittest.main()
