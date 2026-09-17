"""Synthetic checks for training-defined, validation-only baselines."""

import unittest

import pandas as pd

from src.baselines import define_baselines, evaluate_baselines


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.events = pd.DataFrame({
            "R2": [0.8, 0.7, 0.2, 0.1] * 30,
            "target": [0, 0, 1, 1] * 30,
            "split": ["train"] * 80 + ["validation"] * 20 + ["test"] * 20,
        })

    def test_definitions_and_metrics_use_expected_splits(self):
        definition, results = evaluate_baselines(self.events)
        self.assertIn(definition.majority_class, (0, 1))
        self.assertGreaterEqual(definition.r2_cut, 0.1)
        self.assertLessEqual(definition.r2_cut, 0.8)
        self.assertEqual(set(results["model"]), {"Training Majority", "R2 Cut"})
        self.assertEqual(set(results["split"]), {"validation"})
        self.assertTrue((results["train_rows"] == 80).all())
        self.assertTrue((results["validation_rows"] == 20).all())

    def test_test_rows_cannot_change_definitions_or_validation_metrics(self):
        original_definition, original_results = evaluate_baselines(self.events)
        changed = self.events.copy()
        changed.loc[changed["split"] == "test", ["R2", "target"]] = [999, 0]
        repeated_definition, repeated_results = evaluate_baselines(changed)
        self.assertEqual(original_definition, repeated_definition)
        pd.testing.assert_frame_equal(original_results, repeated_results)

    def test_training_rows_alone_set_r2_cut(self):
        original = define_baselines(self.events)
        changed = self.events.copy()
        changed.loc[changed["split"] == "validation", "R2"] = -999
        self.assertEqual(original, define_baselines(changed))


if __name__ == "__main__":
    unittest.main()
