"""Tests for methodology-correction statistics and leakage guards."""

import unittest

import numpy as np
import pandas as pd

from src.methodology import (
    attach_process_type,
    classical_search_spaces,
    hep_operating_points,
    paired_stratified_auc_bootstrap,
    process_efficiencies,
)


class MethodologyTests(unittest.TestCase):
    def test_operating_points_report_background_rejection(self) -> None:
        truth = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        score = np.array([0.1, 0.2, 0.7, 0.8, 0.6, 0.75, 0.9, 0.95])
        table = hep_operating_points(truth, score, targets=(0.75,))
        self.assertGreaterEqual(table.loc[0, "signal_efficiency"], 0.75)
        self.assertEqual(
            table.loc[0, "background_rejection"],
            1 / table.loc[0, "background_efficiency"],
        )

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

    def test_classical_families_receive_four_candidates(self) -> None:
        for _, grid in classical_search_spaces().values():
            candidates = 1
            for values in grid.values():
                candidates *= len(values)
            self.assertEqual(candidates, 4)


if __name__ == "__main__":
    unittest.main()
