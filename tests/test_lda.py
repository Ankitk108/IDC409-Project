"""Checks for the shrinkage-LDA model and feature contract."""

import unittest

import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from src.lda import lda_classifier, lda_development_feature_sets


class LDAModelTests(unittest.TestCase):
    def setUp(self):
        labels = [0, 1] * 10
        self.events = pd.DataFrame({
            "R2": [0.8 if label == 0 else 0.2 for label in labels],
            "thrust": [0.9 if label == 0 else 0.5 for label in labels],
            "target": labels,
            "split": ["train"] * 12 + ["validation"] * 4 + ["test"] * 4,
        })

    def test_classifier_uses_scaled_automatic_shrinkage(self):
        pipeline = lda_classifier()
        classifier = pipeline.named_steps["classifier"]
        self.assertIsInstance(classifier, LinearDiscriminantAnalysis)
        self.assertEqual(classifier.solver, "lsqr")
        self.assertEqual(classifier.shrinkage, "auto")

    def test_feature_sets_exclude_original_test_rows(self):
        development, feature_sets = lda_development_feature_sets(
            self.events, ["R2"]
        )
        self.assertEqual(len(development), 16)
        self.assertEqual(feature_sets["full"], ["R2", "thrust"])
        self.assertEqual(feature_sets["reduced_rfe20"], ["R2"])

    def test_duplicate_reduced_features_are_rejected(self):
        with self.assertRaises(ValueError):
            lda_development_feature_sets(self.events, ["R2", "R2"])


if __name__ == "__main__":
    unittest.main()
