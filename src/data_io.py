"""CSV ingestion, raw-data auditing, and reproducible split assignment."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DB_PATH, RANDOM_STATE, TEST_SIZE, VALIDATION_SIZE
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


def prepare_professor_binary(frame: pd.DataFrame) -> pd.DataFrame:
    """Map the instructor's six process flags to the binary project target.

    Original ``type`` flags describe the generated event process:
        0: e+ e- -> Upsilon(4S) -> B+ B- (charged B-meson pair)
        1: e+ e- -> Upsilon(4S) -> B0 anti-B0 (neutral B-meson pair)
        2: e+ e- -> c anti-c (charm continuum)
        3: e+ e- -> u anti-u (up-quark continuum)
        4: e+ e- -> d anti-d (down-quark continuum)
        5: e+ e- -> s anti-s (strange-quark continuum)

    The professor's binary instruction combines flags 0 and 1 as signal
    (``target=1``), and flags 2, 3, 4, and 5 as continuum background
    (``target=0``). The target is a label, never a model input feature.

    Keep all physics measurements and all rows. The exported pandas row number
    and six-way truth flag are not model features.
    """
    if "type" not in frame.columns or "Unnamed: 0" not in frame.columns:
        raise KeyError("Expected professor CSV columns 'type' and 'Unnamed: 0'")
    if "target" in frame.columns or "split" in frame.columns:
        raise ValueError("Source already contains a target or split assignment")
    if frame["type"].isna().any() or not frame["type"].isin(range(6)).all():
        raise ValueError("Process flags must be complete and in 0, 1, 2, 3, 4, 5")

    # The original six-way truth remains in the immutable raw_events table.
    result = frame.drop(columns=["Unnamed: 0", "type"]).copy()
    if result.isna().any().any():
        raise ValueError("Missing feature values require an explicit cleaning decision")
    numeric = result.select_dtypes(include=np.number)
    if numeric.shape[1] != result.shape[1] or not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Features must be numeric and finite before split assignment")
    result["target"] = frame["type"].isin((0, 1)).astype("int8")
    return result


def assign_split(
    frame: pd.DataFrame,
    label_column: str,
    validation_size: float = VALIDATION_SIZE,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Add one reproducible, stratified train/validation/test assignment.

    ``validation_size`` and ``test_size`` are fractions of the complete dataset,
    not fractions of the intermediate remainder.
    """
    if label_column not in frame.columns:
        raise KeyError(f"Label column not found: {label_column}")
    if "split" in frame.columns:
        raise ValueError("Frame already contains a 'split' column")
    if frame[label_column].isna().any():
        raise ValueError("Labels contain missing values; resolve them before splitting")
    if frame[label_column].nunique() != 2:
        raise ValueError("Binary classification requires exactly two label values")
    if validation_size <= 0 or test_size <= 0:
        raise ValueError("Validation and test sizes must both be greater than zero")
    if validation_size + test_size >= 1:
        raise ValueError("Validation and test sizes must leave a non-empty train split")

    # Index labels can repeat, so split unique row positions rather than labels.
    row_positions = np.arange(len(frame))
    development_positions, test_positions = train_test_split(
        row_positions,
        test_size=test_size,
        random_state=random_state,
        stratify=frame[label_column].to_numpy(),
    )
    validation_fraction_of_remainder = validation_size / (1 - test_size)
    train_positions, validation_positions = train_test_split(
        development_positions,
        test_size=validation_fraction_of_remainder,
        random_state=random_state,
        stratify=frame[label_column].iloc[development_positions].to_numpy(),
    )
    result = frame.copy()
    result["split"] = ""
    split_column_position = result.columns.get_loc("split")
    result.iloc[train_positions, split_column_position] = "train"
    result.iloc[validation_positions, split_column_position] = "validation"
    result.iloc[test_positions, split_column_position] = "test"
    return result
