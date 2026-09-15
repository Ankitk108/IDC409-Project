"""Fast synthetic checks for Ankit's full-feature validation models."""

import unittest

import pandas as pd

from src.models import fit_ankit_validation


class AnkitModelTests(unittest.TestCase):
    def setUp(self):
        labels = [0, 1] * 120
        self.events = pd.DataFrame({
            "R2": [0.8 if label == 0 else 0.2 for label in labels],
            "thrust": [0.9 if label == 0 else 0.5 for label in labels],
            "target": labels,
            "split": ["train"] * 200 + ["validation"] * 20 + ["test"] * 20,
        })

    def test_two_models_report_validation_only(self):
        models, results = fit_ankit_validation(self.events)
        self.assertEqual(set(models), {"Logistic Regression", "Decision Tree"})
        self.assertEqual(set(results["split"]), {"validation"})
        self.assertTrue((results["train_rows"] == 200).all())
        self.assertTrue((results["validation_rows"] == 20).all())
        self.assertTrue((results["features"] == 2).all())
        for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            self.assertTrue(results[metric].between(0, 1).all())

    def test_test_rows_do_not_change_validation_result(self):
        _, original = fit_ankit_validation(self.events)
        changed = self.events.copy()
        changed.loc[changed["split"] == "test", "R2"] = 999
        changed.loc[changed["split"] == "test", "target"] = 0
        _, repeated = fit_ankit_validation(changed)
        pd.testing.assert_frame_equal(original, repeated)

    def test_original_six_way_truth_is_rejected(self):
        contaminated = self.events.assign(type=0)
        with self.assertRaises(ValueError):
            fit_ankit_validation(contaminated)


if __name__ == "__main__":
    unittest.main()
