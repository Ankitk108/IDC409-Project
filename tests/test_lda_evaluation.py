"""Fast checks for the repeated development-only LDA study."""

import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.lda_evaluation import (
    evaluate_lda_study,
    plot_lda_diagnostics,
    plot_lda_discriminants,
)


class LDAEvaluationTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        labels = np.array([0, 1] * 30)
        self.events = pd.DataFrame({
            "R2": 0.8 - 0.6 * labels + rng.normal(0, 0.03, len(labels)),
            "thrust": 0.9 - 0.4 * labels + rng.normal(0, 0.03, len(labels)),
            "target": labels,
            "split": ["train"] * 40 + ["validation"] * 10 + ["test"] * 10,
        })

    def test_study_excludes_test_and_compares_both_feature_sets(self):
        result = evaluate_lda_study(
            self.events, ["R2"], seeds=(42, 123), fold_count=2
        )
        self.assertEqual(len(result.per_seed), 4)
        self.assertEqual(set(result.per_seed["rows"]), {50})
        self.assertEqual(set(result.per_seed["features"]), {1, 2})
        self.assertEqual(set(result.per_seed["seed"]), {42, 123})

    def test_plots_include_two_feature_sets(self):
        result = evaluate_lda_study(
            self.events, ["R2"], seeds=(42, 123), fold_count=2
        )
        diagnostics = plot_lda_diagnostics(result)
        discriminants = plot_lda_discriminants(result)
        self.assertEqual(len(diagnostics.axes), 3)
        self.assertEqual(len(discriminants.axes), 2)
        plt.close(diagnostics)
        plt.close(discriminants)

    def test_study_requires_unique_seeds(self):
        with self.assertRaises(ValueError):
            evaluate_lda_study(
                self.events, ["R2"], seeds=(42, 42), fold_count=2
            )


if __name__ == "__main__":
    unittest.main()
