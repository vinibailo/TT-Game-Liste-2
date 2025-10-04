"""Shared helpers for working with the processed-games database."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from threading import Lock
from typing import Callable

from flask import g, has_app_context
from urllib.parse import urlparse, unquote

db_lock = Lock()
"""Module-level lock to guard write access to the processed-games database."""

_fallback_connection: sqlite3.Connection | None = None
_processed_games_columns_cache: set[str] | None = None


def set_fallback_connection(conn: sqlite3.Connection | None) -> None:
    """Configure the connection returned when no Flask app context is active."""

    global _fallback_connection
    _fallback_connection = conn


def clear_processed_games_columns_cache() -> None:
    """Reset the cached ``processed_games`` column names."""

    global _processed_games_columns_cache
    _processed_games_columns_cache = None


def _configure_sqlite_connection(
    conn: sqlite3.Connection,
    *,
    busy_timeout: float | None = None,
) -> sqlite3.Connection:
    """Apply standard pragmas and timeouts to SQLite connections."""

    busy_timeout_ms = None
    if busy_timeout is not None:
        busy_timeout_ms = int(max(busy_timeout, 0) * 1000)
        if busy_timeout_ms <= 0:
            busy_timeout_ms = None

    pragmas: tuple[tuple[str, str | int | float | None, bool], ...] = (
        ("busy_timeout", busy_timeout_ms, False),
        ("journal_mode", "WAL", True),
        ("mmap_size", 268_435_456, False),
        ("cache_size", -200_000, False),
    )

    for name, value, fetch_result in pragmas:
        if value is None:
            continue
        try:
            cursor = conn.execute(f"PRAGMA {name}={value}")
            if fetch_result:
                # Some pragmas (journal_mode) require fetching a row to apply.
                cursor.fetchone()
        except sqlite3.OperationalError:
            continue

    return conn


def _create_sqlite_connection(
    db_path: str,
    *,
    timeout: float | None = None,
) -> sqlite3.Connection:
    """Create a SQLite connection with the project's standard configuration."""

    effective_timeout = timeout if timeout is not None else 5.0
    conn = sqlite3.connect(db_path, timeout=effective_timeout)
    conn.row_factory = sqlite3.Row
    return _configure_sqlite_connection(conn, busy_timeout=effective_timeout)


def _resolve_sqlite_path_from_dsn(dsn: str) -> str:
    """Extract a filesystem path from a ``sqlite:///`` DSN string."""

    parsed = urlparse(dsn)
    if parsed.scheme != "sqlite":
        raise ValueError(f"Unsupported DSN scheme for SQLite resolver: {parsed.scheme}")

    path = unquote(parsed.path or "")
    if parsed.netloc and parsed.netloc not in {"", "localhost"}:
        # Support UNC-like hosts by prefixing them to the path component.
        path = f"//{parsed.netloc}{path}"

    if not path:
        raise ValueError("SQLite DSN must include a filesystem path")

    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = candidate.resolve()
    return os.fspath(candidate)


def create_connection_from_dsn(
    dsn: str,
    *,
    timeout: float | None = None,
) -> sqlite3.Connection:
    """Create a database connection using a DSN string.

    Currently only ``sqlite:///`` DSNs are supported. MariaDB support will be
    introduced in a future iteration once the application migrates drivers.
    """

    parsed = urlparse(dsn)
    if parsed.scheme == "sqlite":
        sqlite_path = _resolve_sqlite_path_from_dsn(dsn)
        return _create_sqlite_connection(sqlite_path, timeout=timeout)

    raise ValueError(f"Unsupported DSN scheme: {parsed.scheme}")


def get_db(
    connection_factory: Callable[[], sqlite3.Connection] | None = None,
    *,
    context_key: str = 'db',
    use_global_fallback: bool = True,
) -> sqlite3.Connection:
    """Return the active SQLite connection, creating one if necessary."""

    global _fallback_connection

    if has_app_context():
        if not hasattr(g, context_key):
            if connection_factory is not None:
                setattr(g, context_key, connection_factory())
            elif _fallback_connection is not None:
                setattr(g, context_key, _fallback_connection)
            else:
                raise RuntimeError('Database connection is not configured')
        return getattr(g, context_key)

    if not use_global_fallback:
        if connection_factory is None:
            raise RuntimeError(
                'connection_factory is required when no Flask application context is active'
            )
        return connection_factory()

    if _fallback_connection is None:
        if connection_factory is None:
            raise RuntimeError('Database connection is not configured')
        _fallback_connection = connection_factory()
    return _fallback_connection


def get_processed_games_columns(
    conn: sqlite3.Connection | None = None,
    *,
    connection_factory: Callable[[], sqlite3.Connection] | None = None,
) -> set[str]:
    """Return the cached ``processed_games`` column names."""

    global _processed_games_columns_cache
    if _processed_games_columns_cache is not None:
        return _processed_games_columns_cache

    if conn is None:
        conn = get_db(connection_factory)

    cur = conn.execute('PRAGMA table_info(processed_games)')
    _processed_games_columns_cache = {row['name'] for row in cur.fetchall()}
    return _processed_games_columns_cache


def _quote_identifier(identifier: str) -> str:
    """Return the SQLite-safe quoted version of ``identifier``."""

    return '"' + str(identifier).replace('"', '""') + '"'
