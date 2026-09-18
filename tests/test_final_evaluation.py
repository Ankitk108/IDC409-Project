"""Checks for the frozen final evaluator without the optional CNN runtime."""

import unittest

import matplotlib.pyplot as plt
import pandas as pd

from src.final_evaluation import evaluate_frozen_test, plot_test_roc


class FinalEvaluationTests(unittest.TestCase):
    def setUp(self):
        labels = [0, 1] * 240
        self.events = pd.DataFrame({
            "R2": [0.8 if y == 0 else 0.2 for y in labels],
            "thrust": [0.9 if y == 0 else 0.5 for y in labels],
            "target": labels,
            "split": ["train"] * 320 + ["validation"] * 80 + ["test"] * 80,
        })

    def test_all_non_cnn_protocol_rows_use_test(self):
        result = evaluate_frozen_test(self.events, ["R2"], include_cnn=False)
        self.assertEqual(len(result.metrics), 8)
        self.assertEqual(set(result.metrics["split"]), {"test"})
        self.assertFalse(result.metrics.isna().any().any())
        self.assertEqual(set(result.metrics.columns), {
            "model", "feature_set", "split", "fit_rows", "test_rows", "features",
            "accuracy", "precision", "recall", "f1", "roc_auc",
        })
        self.assertTrue((result.metrics["fit_rows"] == 400).all())
        self.assertTrue((result.metrics["test_rows"] == 80).all())
        self.assertEqual(result.metrics.loc[
            result.metrics["model"] == "MLP (RFE-20)", "features"
        ].item(), 1)

    def test_roc_plot_uses_stored_test_scores(self):
        result = evaluate_frozen_test(self.events, ["R2"], include_cnn=False)
        figure = plot_test_roc(result)
        self.assertEqual(len(figure.axes), 1)
        plt.close(figure)


if __name__ == "__main__":
    unittest.main()
