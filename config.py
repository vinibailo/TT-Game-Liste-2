"""Application-wide configuration helpers and constants."""
from __future__ import annotations

import os
from pathlib import Path


def _coerce_positive_float(value: str | None, default: float) -> float:
    """Return ``value`` as a positive float or ``default`` when invalid."""

    try:
        numeric = float(value) if value is not None else default
    except (TypeError, ValueError):
        return default
    return numeric if numeric > 0 else default


def _coerce_truthy_env(value: str | None) -> bool:
    """Return ``True`` when ``value`` represents an affirmative flag."""

    if value is None:
        return False
    text = value.strip().lower()
    return text in {"1", "true", "yes", "on"}


BASE_DIR = Path(__file__).resolve().parent
INPUT_XLSX = "igdb_all_games.xlsx"
PROCESSED_DB = "processed_games.db"
UPLOAD_DIR = "uploaded_sources"
PROCESSED_DIR = "processed_covers"
COVERS_DIR = "covers_out"

DEFAULT_IGDB_USER_AGENT = "TT-Game-Liste/1.0 (support@example.com)"

SQLITE_TIMEOUT_SECONDS = _coerce_positive_float(os.environ.get("SQLITE_TIMEOUT"), 120.0)
RUN_DB_MIGRATIONS = _coerce_truthy_env(os.environ.get("RUN_DB_MIGRATIONS"))
APP_SECRET_KEY = os.environ.get("APP_SECRET_KEY", "dev-secret")
APP_PASSWORD = os.environ.get("APP_PASSWORD", "password")
IGDB_USER_AGENT = os.environ.get("IGDB_USER_AGENT") or DEFAULT_IGDB_USER_AGENT
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")


def get_lookup_data_dir() -> Path:
    """Return the directory containing lookup workbook files."""

    return Path(os.environ.get("LOOKUP_DATA_DIR", BASE_DIR))

__all__ = [
    "APP_PASSWORD",
    "APP_SECRET_KEY",
    "BASE_DIR",
    "COVERS_DIR",
    "DEFAULT_IGDB_USER_AGENT",
    "IGDB_USER_AGENT",
    "INPUT_XLSX",
    "get_lookup_data_dir",
    "OPENAI_API_KEY",
    "PROCESSED_DB",
    "PROCESSED_DIR",
    "RUN_DB_MIGRATIONS",
    "SQLITE_TIMEOUT_SECONDS",
    "UPLOAD_DIR",
]
