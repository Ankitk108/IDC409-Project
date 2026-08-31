"""CSV ingestion, raw-data auditing, and reproducible split assignment."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DB_PATH, RANDOM_STATE, TEST_SIZE
from .db import table_exists, write_table


def load_raw_to_db(
    csv_path: Path,
    db_path: Path = DB_PATH,
    table_name: str = "raw_events",
) -> pd.DataFrame:
    """Load a CSV into a new immutable raw table and return its DataFrame.

    Refusing to replace an existing raw table prevents accidental loss of the
    original database copy. Re-ingestion should use a new database after the source
    file has been deliberately verified.
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Raw CSV not found: {csv_path}")
    if table_exists(table_name, db_path):
        raise FileExistsError(
            f"Table {table_name!r} already exists in {db_path}; refusing to replace it"
        )

    frame = pd.read_csv(csv_path)
    if frame.empty:
        raise ValueError(f"Raw CSV contains no event rows: {csv_path}")
    write_table(frame, table_name, if_exists="fail", db_path=db_path)
    return frame


def audit_frame(frame: pd.DataFrame) -> dict[str, Any]:
    """Return non-mutating structural checks needed before cleaning decisions."""
    numeric = frame.select_dtypes(include=np.number)
    non_finite_counts = {
        column: int((~np.isfinite(numeric[column].to_numpy())).sum())
        for column in numeric.columns
    }
    return {
        "rows": int(frame.shape[0]),
        "columns": int(frame.shape[1]),
        "column_names": frame.columns.tolist(),
        "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "missing_counts": frame.isna().sum().astype(int).to_dict(),
        "non_finite_counts": non_finite_counts,
        "duplicate_rows": int(frame.duplicated().sum()),
        "constant_columns": [
            column for column in frame.columns if frame[column].nunique(dropna=False) <= 1
        ],
    }


def assign_split(
    frame: pd.DataFrame,
    label_column: str,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Add one reproducible, label-stratified train/test assignment."""
    if label_column not in frame.columns:
        raise KeyError(f"Label column not found: {label_column}")
    if "split" in frame.columns:
        raise ValueError("Frame already contains a 'split' column")
    if frame[label_column].isna().any():
        raise ValueError("Labels contain missing values; resolve them before splitting")
    if frame[label_column].nunique() != 2:
        raise ValueError("Binary classification requires exactly two label values")

    train_indices, test_indices = train_test_split(
        frame.index,
        test_size=test_size,
        random_state=random_state,
        stratify=frame[label_column],
    )
    result = frame.copy()
    result["split"] = ""
    result.loc[train_indices, "split"] = "train"
    result.loc[test_indices, "split"] = "test"
    return result

