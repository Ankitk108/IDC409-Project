"""Tests for methodology-correction statistics and leakage guards."""

import unittest

import numpy as np
import pandas as pd

from src.methodology import (
    attach_process_type,
    bootstrap_operating_points,
    classical_search_spaces,
    duplicate_feature_audit,
    hep_operating_points,
    paired_stratified_auc_bootstrap,
    process_auc,
    process_efficiencies,
    selection_stability,
    unified_results_table,
)


class MethodologyTests(unittest.TestCase):
    def test_operating_points_report_background_rejection(self) -> None:
        truth = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        score = np.array([0.1, 0.2, 0.7, 0.8, 0.6, 0.75, 0.9, 0.95])
        table = hep_operating_points(truth, score, targets=(0.75,))
        self.assertEqual(table.loc[0, "signal_efficiency"], 0.75)
        self.assertEqual(
            table.loc[0, "background_rejection_factor"],
            1 / table.loc[0, "background_efficiency"],
        )
        self.assertEqual(
            table.loc[0, "background_rejection_fraction"],
            1 - table.loc[0, "background_efficiency"],
        )

    def test_operating_point_bootstrap_reports_intervals(self) -> None:
        truth = np.array([0] * 20 + [1] * 20)
        score = np.r_[np.linspace(0, 0.7, 20), np.linspace(0.3, 1, 20)]
        table = bootstrap_operating_points(
            truth, {"model": score}, targets=(0.8,), iterations=100, random_state=2
        )
        self.assertLessEqual(table.loc[0, "rejection_factor_ci_95_low"],
                             table.loc[0, "rejection_factor_ci_95_high"])
        self.assertTrue(np.isfinite(table.loc[0, "background_rejection_factor"]))

    def test_paired_bootstrap_preserves_score_pairing(self) -> None:
        truth = np.array([0] * 20 + [1] * 20)
        perfect = np.r_[np.linspace(0, 0.4, 20), np.linspace(0.6, 1, 20)]
        reversed_score = 1 - perfect
        aucs, differences = paired_stratified_auc_bootstrap(
            truth, {"best": perfect, "worst": reversed_score},
            reference="best", iterations=100, random_state=3,
        )
        self.assertEqual(aucs.loc[aucs["model"] == "best", "roc_auc"].item(), 1.0)
        self.assertGreater(differences.loc[0, "ci_95_low"], 0.9)

    def test_process_join_rejects_order_mismatch(self) -> None:
        raw = pd.DataFrame({"type": [0, 2, 1]})
        cleaned = pd.DataFrame({"target": [1, 1, 0], "split": ["train"] * 3})
        with self.assertRaises(ValueError):
            attach_process_type(raw, cleaned)

    def test_process_efficiency_keeps_six_way_truth(self) -> None:
        thresholds = pd.DataFrame({
            "target_signal_efficiency": [0.9], "threshold": [0.5],
        })
        table = process_efficiencies(
            np.arange(6), np.array([0.9, 0.8, 0.7, 0.4, 0.3, 0.2]), thresholds
        )
        self.assertEqual(set(table["type"]), set(range(6)))
        self.assertEqual(table.loc[table["type"] == 2, "signal_acceptance"].item(), 1.0)

    def test_process_auc_compares_each_background_with_all_signal(self) -> None:
        process = np.array([0, 1, 2, 2, 3, 3, 4, 4, 5, 5])
        score = np.array([0.9, 0.8, 0.1, 0.2, 0.2, 0.3, 0.3, 0.4, 0.4, 0.5])
        table = process_auc(process, score)
        self.assertEqual(set(table["process"]), {"ccbar", "uubar", "ddbar", "ssbar"})
        self.assertTrue((table["roc_auc"] == 1.0).all())

    def test_feature_audit_finds_duplicates_and_zeros(self) -> None:
        frame = pd.DataFrame({"a": [0, 1, 0], "b": [0, 1, 0], "c": [1, 2, 3]})
        table = duplicate_feature_audit(frame, ["a", "b", "c"])
        self.assertEqual(table.loc[table["feature"] == "b", "duplicate_of"].item(), "a")
        self.assertAlmostEqual(table.loc[table["feature"] == "a", "zero_fraction"].item(), 2 / 3)

    def test_selection_stability_reports_jaccard(self) -> None:
        selected = pd.DataFrame({"fold": [1, 1, 2, 2], "feature": ["a", "b", "a", "c"]})
        frequency, overlap = selection_stability(selected)
        self.assertEqual(frequency.loc[frequency["feature"] == "a", "selected_folds"].item(), 2)
        self.assertAlmostEqual(overlap.loc[0, "jaccard"], 1 / 3)

    def test_unified_table_keeps_unavailable_nested_cells(self) -> None:
        auc = pd.DataFrame({"model": ["A", "B"], "roc_auc": [0.9, 0.8],
                            "ci_95_low": [0.8, 0.7], "ci_95_high": [1.0, 0.9]})
        nested = pd.DataFrame({"model": ["A"], "feature_set": ["full"],
                               "fold_roc_auc_mean": [0.88]})
        operating = pd.DataFrame({
            "model": ["A", "A"], "target_signal_efficiency": [0.9, 0.95],
            "background_rejection_factor": [5.0, 3.0],
            "rejection_factor_ci_95_low": [4.0, 2.0],
            "rejection_factor_ci_95_high": [6.0, 4.0],
        })
        table = unified_results_table(auc, nested, operating)
        self.assertEqual(table.loc[0, "nested_dev_auc_mean"], 0.88)
        self.assertTrue(pd.isna(table.loc[1, "nested_dev_auc_mean"]))
        self.assertEqual(table.loc[0, "rejection_factor_at_90"], "5.0 [4.0, 6.0]")

    def test_classical_families_receive_four_candidates(self) -> None:
        for _, grid in classical_search_spaces().values():
            candidates = 1
            for values in grid.values():
                candidates *= len(values)
            self.assertEqual(candidates, 4)


if __name__ == "__main__":
    unittest.main()
