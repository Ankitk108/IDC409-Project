"""Small, centralized pandas/SQLite access helpers."""

import sqlite3
import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from .config import DB_PATH

_SQL_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_table_name(name: str) -> str:
    """Reject names that cannot safely be used as SQLite identifiers."""
    if not _SQL_IDENTIFIER.fullmatch(name):
        raise ValueError(f"Invalid SQLite table name: {name!r}")
    return name


@contextmanager
def get_connection(db_path: Path = DB_PATH) -> Iterator[sqlite3.Connection]:
    """Open a SQLite connection and commit or roll back atomically."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def read_table(name: str, db_path: Path = DB_PATH) -> pd.DataFrame:
    """Read a complete SQLite table into a DataFrame."""
    name = _validate_table_name(name)
    with get_connection(db_path) as connection:
        return pd.read_sql_query(f'SELECT * FROM "{name}"', connection)


def write_table(
    frame: pd.DataFrame,
    name: str,
    if_exists: str = "fail",
    db_path: Path = DB_PATH,
) -> None:
    """Write a DataFrame to SQLite through the shared connection contract."""
    name = _validate_table_name(name)
    with get_connection(db_path) as connection:
        frame.to_sql(name, connection, if_exists=if_exists, index=False)


def table_exists(name: str, db_path: Path = DB_PATH) -> bool:
    """Return whether a table exists without modifying the database."""
    name = _validate_table_name(name)
    query = "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?"
    with get_connection(db_path) as connection:
        return connection.execute(query, (name,)).fetchone() is not None


def update_cell(
    table: str,
    column: str,
    value: object,
    key_column: str,
    key_value: object,
    db_path: Path = DB_PATH,
) -> int:
    """Update matching rows using checked identifiers and bound SQL values."""
    table = _validate_table_name(table)
    column = _validate_table_name(column)
    key_column = _validate_table_name(key_column)
    if table in {"raw_events", "cleaned_events"}:
        raise ValueError("Source and split tables are immutable through update_cell")
    with get_connection(db_path) as connection:
        columns = {
            row[1] for row in connection.execute(f'PRAGMA table_info("{table}")')
        }
        if column not in columns or key_column not in columns:
            raise KeyError("Update or key column does not exist")
        cursor = connection.execute(
            f'UPDATE "{table}" SET "{column}" = ? WHERE "{key_column}" = ?',
            (value, key_value),
        )
        return cursor.rowcount
