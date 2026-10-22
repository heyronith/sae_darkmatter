"""Repository path helpers."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
DATA_DIR = REPO_ROOT / "data"


def repo_root() -> Path:
    return REPO_ROOT


def results_dir() -> Path:
    return RESULTS_DIR


def data_dir() -> Path:
    return DATA_DIR
