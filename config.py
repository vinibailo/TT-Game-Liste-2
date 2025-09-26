"""Application-wide configuration helpers and constants."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Final

BASE_DIR: Final[Path] = Path(__file__).resolve().parent

try:  # pragma: no cover - optional dependency for local development
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - python-dotenv is optional
    load_dotenv = None  # type: ignore[assignment]

if load_dotenv is not None:
    load_dotenv(BASE_DIR / ".env")


def _clean_text(value: str | None) -> str:
    """Return ``value`` stripped of surrounding whitespace."""

    if value is None:
        return ""
    return value.strip()


def _path_from(env_value: str | None, default: str | Path) -> Path:
    """Resolve a filesystem path using an environment override when provided."""

    text = _clean_text(env_value)
    candidate = Path(text) if text else Path(default)
    candidate = candidate.expanduser()
    if candidate.is_absolute():
        try:
            return candidate.resolve()
        except (OSError, RuntimeError):  # pragma: no cover - fallback for exotic paths
            return candidate
    return candidate


def _coerce_positive_float(value: str | None, default: float) -> float:
    """Return ``value`` coerced to a positive float or ``default`` when invalid."""

    text = _clean_text(value)
    if not text:
        return default
    try:
        numeric = float(text)
    except (TypeError, ValueError):
        return default
    return numeric if numeric > 0 else default


def _coerce_positive_int(value: str | None, default: int) -> int:
    """Return ``value`` coerced to a positive integer or ``default`` when invalid."""

    text = _clean_text(value)
    if not text:
        return default
    try:
        numeric = int(float(text))
    except (TypeError, ValueError):
        return default
    return numeric if numeric > 0 else default


def _coerce_truthy_env(value: str | None) -> bool:
    """Return ``True`` when ``value`` represents an affirmative flag."""

    if value is None:
        return False
    text = value.strip().lower()
    return text in {"1", "true", "yes", "on"}


INPUT_XLSX_PATH: Final[Path] = _path_from(
    os.environ.get("INPUT_XLSX"), "igdb_all_games.xlsx"
)
PROCESSED_DB_PATH: Final[Path] = _path_from(
    os.environ.get("PROCESSED_DB"), "processed_games.db"
)
UPLOAD_DIR_PATH: Final[Path] = _path_from(
    os.environ.get("UPLOAD_DIR"), "uploaded_sources"
)
PROCESSED_DIR_PATH: Final[Path] = _path_from(
    os.environ.get("PROCESSED_DIR"), "processed_covers"
)
COVERS_DIR_PATH: Final[Path] = _path_from(os.environ.get("COVERS_DIR"), "covers_out")

LOG_DIR_PATH: Final[Path] = _path_from(os.environ.get("LOG_DIR"), BASE_DIR / "logs")
LOG_DIR: Final[str] = os.fspath(LOG_DIR_PATH)
LOG_FILE_PATH: Final[Path] = _path_from(
    os.environ.get("LOG_FILE"), LOG_DIR_PATH / "app.log"
)
LOG_FILE: Final[str] = os.fspath(LOG_FILE_PATH)

INPUT_XLSX: Final[str] = os.fspath(INPUT_XLSX_PATH)
PROCESSED_DB: Final[str] = os.fspath(PROCESSED_DB_PATH)
UPLOAD_DIR: Final[str] = os.fspath(UPLOAD_DIR_PATH)
PROCESSED_DIR: Final[str] = os.fspath(PROCESSED_DIR_PATH)
COVERS_DIR: Final[str] = os.fspath(COVERS_DIR_PATH)

DEFAULT_IGDB_USER_AGENT: Final[str] = "TT-Game-Liste/1.0 (support@example.com)"
IGDB_USER_AGENT: Final[str] = (
    _clean_text(os.environ.get("IGDB_USER_AGENT")) or DEFAULT_IGDB_USER_AGENT
)

SQLITE_TIMEOUT_SECONDS: Final[float] = _coerce_positive_float(
    os.environ.get("SQLITE_TIMEOUT"), 120.0
)
RUN_DB_MIGRATIONS: Final[bool] = _coerce_truthy_env(
    os.environ.get("RUN_DB_MIGRATIONS")
)

APP_SECRET_KEY: Final[str] = _clean_text(os.environ.get("APP_SECRET_KEY")) or "dev-secret"
APP_PASSWORD: Final[str] = _clean_text(os.environ.get("APP_PASSWORD")) or "password"

_OPENAI_RAW = os.environ.get("OPENAI_API_KEY")
OPENAI_API_KEY: Final[str] = _clean_text(_OPENAI_RAW)
if _OPENAI_RAW is not None and not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is set but empty; provide a value or unset the variable."
    )
OPENAI_SUMMARY_ENABLED: Final[bool] = bool(OPENAI_API_KEY)

IGDB_BATCH_SIZE: Final[int] = _coerce_positive_int(
    os.environ.get("IGDB_BATCH_SIZE"), 500
)
FIX_NAMES_BATCH_LIMIT: Final[int] = _coerce_positive_int(
    os.environ.get("FIX_NAMES_BATCH_LIMIT"), 50
)

DEFAULT_LOOKUP_DATA_DIR: Final[Path] = BASE_DIR


def get_lookup_data_dir() -> Path:
    """Return the directory containing lookup workbook files."""

    return _path_from(os.environ.get("LOOKUP_DATA_DIR"), DEFAULT_LOOKUP_DATA_DIR)


def _validate_settings() -> None:
    """Sanity-check critical configuration values."""

    if not APP_SECRET_KEY:
        raise RuntimeError("APP_SECRET_KEY must not be empty")
    if not APP_PASSWORD:
        raise RuntimeError("APP_PASSWORD must not be empty")


_validate_settings()


__all__ = [
    "APP_PASSWORD",
    "APP_SECRET_KEY",
    "BASE_DIR",
    "COVERS_DIR",
    "COVERS_DIR_PATH",
    "DEFAULT_IGDB_USER_AGENT",
    "DEFAULT_LOOKUP_DATA_DIR",
    "FIX_NAMES_BATCH_LIMIT",
    "IGDB_BATCH_SIZE",
    "IGDB_USER_AGENT",
    "INPUT_XLSX",
    "INPUT_XLSX_PATH",
    "LOG_DIR",
    "LOG_DIR_PATH",
    "LOG_FILE",
    "LOG_FILE_PATH",
    "OPENAI_API_KEY",
    "OPENAI_SUMMARY_ENABLED",
    "PROCESSED_DB",
    "PROCESSED_DB_PATH",
    "PROCESSED_DIR",
    "PROCESSED_DIR_PATH",
    "RUN_DB_MIGRATIONS",
    "SQLITE_TIMEOUT_SECONDS",
    "UPLOAD_DIR",
    "UPLOAD_DIR_PATH",
    "get_lookup_data_dir",
]
