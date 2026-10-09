"""Fast checks for the post-hoc reduced Random Forest extension."""

import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.extensions import (
    evaluate_reduced_bdt_comparison_cv,
    evaluate_reduced_random_forest_cv,
    plot_reduced_bdt_comparison,
    plot_reduced_random_forest_diagnostics,
)


class ReducedRandomForestTests(unittest.TestCase):
    def setUp(self):
        labels = [0, 1] * 60
        self.events = pd.DataFrame({
            "R2": [0.8 if label == 0 else 0.2 for label in labels],
            "thrust": [0.9 if label == 0 else 0.5 for label in labels],
            "target": labels,
            "split": ["train"] * 80 + ["validation"] * 20 + ["test"] * 20,
        })

    def test_reduced_forest_ignores_original_test(self):
        original = evaluate_reduced_random_forest_cv(
            self.events, ["R2"], fold_count=2, n_estimators=5
        )
        changed = self.events.copy()
        changed.loc[changed["split"] == "test", ["R2", "target"]] = [999, 0]
        repeated = evaluate_reduced_random_forest_cv(
            changed, ["R2"], fold_count=2, n_estimators=5
        )
        pd.testing.assert_frame_equal(original.metrics, repeated.metrics)
        self.assertEqual(original.metrics.loc[0, "rows"], 100)
        self.assertEqual(original.metrics.loc[0, "features"], 1)

    def test_diagnostics_use_oof_predictions(self):
        result = evaluate_reduced_random_forest_cv(
            self.events, ["R2"], fold_count=2, n_estimators=5
        )
        figure = plot_reduced_random_forest_diagnostics(result)
        self.assertEqual(len(figure.axes), 2)
        plt.close(figure)

    def test_unknown_reduced_feature_is_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_reduced_random_forest_cv(
                self.events, ["type"], fold_count=2, n_estimators=5
            )

    def test_bdt_comparison_ignores_test_and_matches_feature_sets(self):
        original = evaluate_reduced_bdt_comparison_cv(
            self.events, ["R2"], fold_count=2
        )
        changed = self.events.copy()
        changed.loc[changed["split"] == "test", ["R2", "target"]] = [999, 0]
        repeated = evaluate_reduced_bdt_comparison_cv(
            changed, ["R2"], fold_count=2
        )
        pd.testing.assert_frame_equal(original.metrics, repeated.metrics)
        self.assertEqual(original.metrics["features"].tolist(), [2, 1])
        self.assertEqual(original.metrics["rows"].tolist(), [100, 100])

    def test_bdt_comparison_plot_contains_roc_and_two_matrices(self):
        result = evaluate_reduced_bdt_comparison_cv(
            self.events, ["R2"], fold_count=2
        )
        figure = plot_reduced_bdt_comparison(result)
        self.assertEqual(len(figure.axes), 3)
        plt.close(figure)

    def test_bdt_comparison_rejects_duplicate_reduced_features(self):
        with self.assertRaises(ValueError):
            evaluate_reduced_bdt_comparison_cv(
                self.events, ["R2", "R2"], fold_count=2
            )


if __name__ == "__main__":
    unittest.main()
