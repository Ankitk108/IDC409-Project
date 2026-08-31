"""Small, centralized pandas/SQLite access helpers."""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from .config import DB_PATH


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
    with get_connection(db_path) as connection:
        return pd.read_sql_query(f'SELECT * FROM "{name}"', connection)


def write_table(
    frame: pd.DataFrame,
    name: str,
    if_exists: str = "fail",
    db_path: Path = DB_PATH,
) -> None:
    """Write a DataFrame to SQLite through the shared connection contract."""
    with get_connection(db_path) as connection:
        frame.to_sql(name, connection, if_exists=if_exists, index=False)

