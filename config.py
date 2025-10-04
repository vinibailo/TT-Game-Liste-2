"""Application-wide configuration helpers and constants."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Final
from urllib.parse import quote_plus

BASE_DIR: Final[Path] = Path(__file__).resolve().parent

try:  # pragma: no cover - optional dependency for local development
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - python-dotenv is optional
    load_dotenv = None  # type: ignore[assignment]

if load_dotenv is not None:
    load_dotenv(BASE_DIR / ".env")


logger = logging.getLogger(__name__)


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
UPLOAD_DIR: Final[str] = os.fspath(UPLOAD_DIR_PATH)
PROCESSED_DIR: Final[str] = os.fspath(PROCESSED_DIR_PATH)
COVERS_DIR: Final[str] = os.fspath(COVERS_DIR_PATH)

DB_HOST: Final[str] = _clean_text(os.environ.get("DB_HOST")) or "localhost"
DB_PORT: Final[int] = _coerce_positive_int(os.environ.get("DB_PORT"), 3306)
DB_NAME: Final[str] = _clean_text(os.environ.get("DB_NAME")) or "tt_game_liste"
DB_USER: Final[str] = _clean_text(os.environ.get("DB_USER"))
DB_PASSWORD: Final[str] = _clean_text(os.environ.get("DB_PASSWORD"))
DB_SSL_CA_PATH: Final[Path | None] = (
    _path_from(os.environ.get("DB_SSL_CA"), "") if os.environ.get("DB_SSL_CA") else None
)
DB_SSL_CA: Final[str] = os.fspath(DB_SSL_CA_PATH) if DB_SSL_CA_PATH is not None else ""


def _build_db_dsn() -> str:
    """Return a database DSN constructed from environment configuration."""

    maria_overrides = {
        key: _clean_text(os.environ.get(key))
        for key in ("DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD")
    }
    if any(value for value in maria_overrides.values()):
        auth = ""
        if DB_USER:
            password = quote_plus(DB_PASSWORD) if DB_PASSWORD else ""
            auth = DB_USER
            if password:
                auth = f"{auth}:{password}"
            auth = f"{auth}@"

        query_params = []
        if DB_SSL_CA:
            query_params.append(f"ssl_ca={quote_plus(DB_SSL_CA)}")

        query_string = f"?{'&'.join(query_params)}" if query_params else ""
        return f"mariadb://{auth}{DB_HOST}:{DB_PORT}/{DB_NAME}{query_string}"

    sqlite_path = _path_from(None, BASE_DIR / "processed_games.db").resolve()
    return f"sqlite:///{sqlite_path.as_posix()}"


DB_DSN: Final[str] = _build_db_dsn()

DEFAULT_IGDB_USER_AGENT: Final[str] = "TT-Game-Liste/1.0 (support@example.com)"
IGDB_USER_AGENT: Final[str] = (
    _clean_text(os.environ.get("IGDB_USER_AGENT")) or DEFAULT_IGDB_USER_AGENT
)

IGDB_CLIENT_ID: Final[str] = _clean_text(os.environ.get("IGDB_CLIENT_ID"))
IGDB_CLIENT_SECRET: Final[str] = _clean_text(os.environ.get("IGDB_CLIENT_SECRET"))
IGDB_ENABLED: bool = True

DB_CONNECT_TIMEOUT_SECONDS: Final[float] = _coerce_positive_float(
    os.environ.get("DB_CONNECT_TIMEOUT"), 10.0
)
DB_READ_TIMEOUT_SECONDS: Final[float] = _coerce_positive_float(
    os.environ.get("DB_READ_TIMEOUT"), 30.0
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

DEFAULT_LOOKUP_DATA_DIR: Final[Path] = BASE_DIR


def get_lookup_data_dir() -> Path:
    """Return the directory containing lookup workbook files."""

    return _path_from(os.environ.get("LOOKUP_DATA_DIR"), DEFAULT_LOOKUP_DATA_DIR)


def validate_igdb_credentials() -> bool:
    """Ensure IGDB credentials are configured and update ``IGDB_ENABLED``."""

    global IGDB_ENABLED

    missing = [
        name
        for name, value in (
            ("IGDB_CLIENT_ID", IGDB_CLIENT_ID),
            ("IGDB_CLIENT_SECRET", IGDB_CLIENT_SECRET),
        )
        if not value
    ]

    IGDB_ENABLED = not missing
    if missing:
        logger.error(
            "Missing required IGDB credentials; set %s.", " and ".join(missing)
        )

    return IGDB_ENABLED


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
    "IGDB_BATCH_SIZE",
    "IGDB_CLIENT_ID",
    "IGDB_CLIENT_SECRET",
    "IGDB_ENABLED",
    "IGDB_USER_AGENT",
    "INPUT_XLSX",
    "INPUT_XLSX_PATH",
    "LOG_DIR",
    "LOG_DIR_PATH",
    "LOG_FILE",
    "LOG_FILE_PATH",
    "OPENAI_API_KEY",
    "OPENAI_SUMMARY_ENABLED",
    "DB_CONNECT_TIMEOUT_SECONDS",
    "DB_DSN",
    "DB_HOST",
    "DB_NAME",
    "DB_PASSWORD",
    "DB_PORT",
    "DB_READ_TIMEOUT_SECONDS",
    "DB_SSL_CA",
    "DB_SSL_CA_PATH",
    "DB_USER",
    "PROCESSED_DIR",
    "PROCESSED_DIR_PATH",
    "RUN_DB_MIGRATIONS",
    "UPLOAD_DIR",
    "UPLOAD_DIR_PATH",
    "validate_igdb_credentials",
    "get_lookup_data_dir",
]
