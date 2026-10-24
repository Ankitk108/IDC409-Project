"""Small synthetic checks for training-only EDA behavior."""

import unittest

import pandas as pd

from src.eda import class_balance, feature_columns, feature_separation, training_events


class EDATests(unittest.TestCase):
    def setUp(self):
        self.events = pd.DataFrame({
            "R2": [0.1, 0.2, 0.8, 0.9, 99.0, -99.0],
            "thrust": [0.2, 0.3, 0.7, 0.8, 99.0, -99.0],
            "target": [1, 1, 0, 0, 1, 0],
            "split": ["train"] * 4 + ["validation", "test"],
        })

    def test_only_training_rows_enter_eda(self):
        train = training_events(self.events)
        self.assertEqual(len(train), 4)
        self.assertEqual(feature_columns(train), ["R2", "thrust"])
        self.assertEqual(class_balance(train)["rows"].tolist(), [2, 2])

    def test_auc_and_medians_ignore_held_out_outliers(self):
        rank = feature_separation(training_events(self.events))
        self.assertEqual(set(rank["feature"]), {"R2", "thrust"})
        self.assertTrue((rank["separation_auc"] == 1).all())
        self.assertTrue((rank["higher_for_signal"] == 0).all())
        self.assertTrue((rank["ks_statistic"] == 1).all())
        self.assertAlmostEqual(
            rank.loc[rank["feature"] == "R2", "signal_median"].item(), 0.15
        )

    def test_raw_truth_and_exported_number_are_not_eda_features(self):
        train = training_events(self.events)
        for reserved in ("type", "Unnamed: 0"):
            with self.subTest(reserved=reserved):
                with self.assertRaises(ValueError):
                    feature_columns(train.assign(**{reserved: 0}))


if __name__ == "__main__":
    unittest.main()
