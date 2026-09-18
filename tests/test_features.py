"""Synthetic checks for train-only feature-reduction analysis."""

import unittest

import pandas as pd

from src.features import analyze_feature_reduction


class FeatureReductionTests(unittest.TestCase):
    def setUp(self):
        labels = [0, 1] * 60
        self.events = pd.DataFrame({
            "strong": [0.9 if y == 0 else 0.1 for y in labels],
            "duplicate": [0.9 if y == 0 else 0.1 for y in labels],
            "weak": [i % 7 for i in range(120)],
            "extra": [i % 5 for i in range(120)],
            "target": labels,
            "split": ["train"] * 80 + ["validation"] * 20 + ["test"] * 20,
        })

    def test_analysis_prunes_duplicate_and_selects_requested_count(self):
        result = analyze_feature_reduction(
            self.events, correlation_threshold=0.95, selected_count=2
        )
        self.assertEqual(len(result.correlation_dropped), 1)
        self.assertEqual(
            set(result.correlation_dropped) & {"strong", "duplicate"},
            set(result.correlation_dropped),
        )
        self.assertEqual(len(result.selected_features), 2)
        self.assertTrue(result.pca_variance["cumulative_variance"].iloc[-1] > 0.99)

    def test_held_out_values_do_not_change_reduction(self):
        original = analyze_feature_reduction(self.events, selected_count=2)
        changed = self.events.copy()
        held_out = changed["split"] != "train"
        changed.loc[held_out, ["strong", "duplicate", "weak", "extra"]] = 999
        repeated = analyze_feature_reduction(changed, selected_count=2)
        self.assertEqual(original.correlation_kept, repeated.correlation_kept)
        self.assertEqual(original.correlation_dropped, repeated.correlation_dropped)
        self.assertEqual(original.selected_features, repeated.selected_features)
        pd.testing.assert_frame_equal(original.pca_variance, repeated.pca_variance)


if __name__ == "__main__":
    unittest.main()
