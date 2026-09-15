"""Fast tests for the ingestion and split contracts."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.data_io import (
    assign_split, audit_frame, load_raw_to_db, prepare_professor_binary
)
from src.db import read_table


class DataIoTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = pd.DataFrame(
            {
                "feature": list(range(20)),
                "constant": [1] * 20,
                "label": [0, 1] * 10,
            }
        )

    def test_audit_reports_structure_without_mutating_frame(self) -> None:
        original = self.frame.copy()
        audit = audit_frame(self.frame)
        self.assertEqual(audit["rows"], 20)
        self.assertEqual(audit["columns"], 3)
        self.assertEqual(audit["constant_columns"], ["constant"])
        pd.testing.assert_frame_equal(self.frame, original)

    def test_load_raw_creates_table_and_refuses_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            csv_path = directory_path / "events.csv"
            db_path = directory_path / "events.db"
            self.frame.to_csv(csv_path, index=False)

            loaded = load_raw_to_db(csv_path, db_path=db_path)
            pd.testing.assert_frame_equal(loaded, read_table("raw_events", db_path))
            with self.assertRaises(FileExistsError):
                load_raw_to_db(csv_path, db_path=db_path)

    def test_assign_split_is_reproducible_and_stratified(self) -> None:
        frame = pd.DataFrame(
            {"feature": range(100), "label": [0, 1] * 50}
        )
        first = assign_split(
            frame, "label", validation_size=0.2, test_size=0.2
        )
        second = assign_split(
            frame, "label", validation_size=0.2, test_size=0.2
        )
        pd.testing.assert_series_equal(first["split"], second["split"])

        train_rows = first[first["split"] == "train"]
        validation_rows = first[first["split"] == "validation"]
        test_rows = first[first["split"] == "test"]
        self.assertEqual(len(train_rows), 60)
        self.assertEqual(len(validation_rows), 20)
        self.assertEqual(len(test_rows), 20)
        for split_rows in (train_rows, validation_rows, test_rows):
            self.assertEqual(split_rows["label"].value_counts().nunique(), 1)

    def test_assign_split_rejects_invalid_fractions(self) -> None:
        with self.assertRaises(ValueError):
            assign_split(self.frame, "label", validation_size=0.5, test_size=0.5)

    def test_assign_split_keeps_duplicate_index_rows_disjoint(self) -> None:
        frame = pd.DataFrame(
            {"feature": range(100), "label": [0, 1] * 50},
            index=[position // 2 for position in range(100)],
        )
        result = assign_split(
            frame, "label", validation_size=0.2, test_size=0.2
        )
        self.assertEqual(result.index.tolist(), frame.index.tolist())
        self.assertEqual(result["split"].value_counts().to_dict(),
                         {"train": 60, "validation": 20, "test": 20})
        for split in ("train", "validation", "test"):
            self.assertEqual(result.loc[result["split"] == split, "label"]
                             .value_counts().to_dict(),
                             {0: len(result.loc[result["split"] == split]) // 2,
                              1: len(result.loc[result["split"] == split]) // 2})
        repeated = assign_split(
            frame, "label", validation_size=0.2, test_size=0.2
        )
        pd.testing.assert_series_equal(result["split"], repeated["split"])

    def test_professor_mapping_excludes_truth_and_exported_number(self) -> None:
        raw = pd.DataFrame({
            "Unnamed: 0": [0, 1, 1, 3, 4, 5],
            "R2": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "type": [0, 1, 2, 3, 4, 5],
        })
        prepared = prepare_professor_binary(raw)
        self.assertEqual(prepared.columns.tolist(), ["R2", "target"])
        self.assertEqual(prepared["target"].tolist(), [1, 1, 0, 0, 0, 0])
        self.assertEqual(raw.columns.tolist(), ["Unnamed: 0", "R2", "type"])
        self.assertEqual(len(prepared), len(raw))

    def test_professor_mapping_rejects_unknown_type(self) -> None:
        raw = pd.DataFrame({"Unnamed: 0": [0], "R2": [0.1], "type": [6]})
        with self.assertRaises(ValueError):
            prepare_professor_binary(raw)


if __name__ == "__main__":
    unittest.main()
