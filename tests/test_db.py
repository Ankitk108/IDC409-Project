"""SQLite update tests on a disposable database, never the project DB."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.db import read_table, update_cell, write_table


class UpdateCellTests(unittest.TestCase):
    def test_update_derived_table_only(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "test.db"
            write_table(pd.DataFrame({"feature": ["R2"], "status": ["computed"]}),
                        "eda_feature_summary", db_path=db_path)
            changed = update_cell("eda_feature_summary", "status", "candidate_for_review",
                                  "feature", "R2", db_path=db_path)
            self.assertEqual(changed, 1)
            self.assertEqual(read_table("eda_feature_summary", db_path)["status"].item(),
                             "candidate_for_review")
            with self.assertRaises(ValueError):
                update_cell("raw_events", "status", "bad", "feature", "R2", db_path)
            with self.assertRaises(ValueError):
                update_cell("cleaned_events", "status", "bad", "feature", "R2", db_path)


if __name__ == "__main__":
    unittest.main()
