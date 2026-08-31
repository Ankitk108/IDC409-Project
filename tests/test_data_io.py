"""Fast tests for the ingestion and split contracts."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.data_io import assign_split, audit_frame, load_raw_to_db
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
        first = assign_split(self.frame, "label", test_size=0.2)
        second = assign_split(self.frame, "label", test_size=0.2)
        pd.testing.assert_series_equal(first["split"], second["split"])
        test_rows = first[first["split"] == "test"]
        self.assertEqual(len(test_rows), 4)
        self.assertEqual(test_rows["label"].value_counts().to_dict(), {0: 2, 1: 2})


if __name__ == "__main__":
    unittest.main()

