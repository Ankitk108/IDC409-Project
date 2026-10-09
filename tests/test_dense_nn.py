"""Fast checks for the post-hoc deep dense network."""

import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.dense_nn import evaluate_deep_dense_cv, plot_deep_dense_diagnostics


class DeepDenseTests(unittest.TestCase):
    def setUp(self):
        labels = [0, 1] * 30
        self.events = pd.DataFrame({
            "R2": [0.8 if label == 0 else 0.2 for label in labels],
            "thrust": [0.9 if label == 0 else 0.5 for label in labels],
            "target": labels,
            "split": ["train"] * 40 + ["validation"] * 10 + ["test"] * 10,
        })

    def test_oof_result_excludes_original_test_rows(self):
        result = evaluate_deep_dense_cv(
            self.events,
            ["R2"],
            fold_count=2,
            max_epochs=2,
            patience=1,
            batch_size=32,
        )
        self.assertEqual(result.metrics.loc[0, "rows"], 50)
        self.assertEqual(result.metrics.loc[0, "features"], 1)
        self.assertEqual(len(result.truth), 50)
        self.assertEqual(len(result.fitted_epochs), 2)

    def test_diagnostics_contain_roc_and_confusion_matrix(self):
        result = evaluate_deep_dense_cv(
            self.events,
            ["R2"],
            fold_count=2,
            max_epochs=1,
            patience=1,
            batch_size=32,
        )
        figure = plot_deep_dense_diagnostics(result)
        self.assertEqual(len(figure.axes), 2)
        plt.close(figure)

    def test_duplicate_features_are_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_deep_dense_cv(self.events, ["R2", "R2"], fold_count=2)


if __name__ == "__main__":
    unittest.main()
