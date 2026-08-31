"""Project-wide reproducibility and path settings."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
DB_PATH = DATA_DIR / "continuum.db"
RESULTS_DIR = PROJECT_ROOT / "results"

RANDOM_STATE = 42
TEST_SIZE = 0.25

