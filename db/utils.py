"""Shared helpers for working with the processed-games database."""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from typing import Callable, Iterator

from flask import g, has_app_context
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.engine.default import DefaultDialect
from sqlalchemy.orm import Session, sessionmaker
from urllib.parse import unquote, urlparse

db_lock = Lock()
"""Module-level lock to guard write access to the processed-games database."""


class DatabaseEngine:
    """Wrapper exposing context-managed SQLAlchemy connections and sessions."""

    def __init__(self, engine: Engine):
        self._engine = engine
        self._session_factory = sessionmaker(bind=engine, future=True)

    @property
    def engine(self) -> Engine:
        """Return the underlying SQLAlchemy :class:`~sqlalchemy.engine.Engine`."""

        return self._engine

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """Yield a DBAPI connection configured by the engine's pool."""

        raw = self._engine.raw_connection()
        try:
            yield raw
        finally:
            raw.close()

    @contextmanager
    def sa_connection(self) -> Iterator[Connection]:
        """Yield a SQLAlchemy :class:`~sqlalchemy.engine.Connection`."""

        with self._engine.connect() as conn:
            yield conn

    @contextmanager
    def session(self) -> Iterator[Session]:
        """Yield a SQLAlchemy :class:`~sqlalchemy.orm.Session`."""

        with self._session_factory() as session:
            yield session

    def dispose(self) -> None:
        """Dispose the underlying engine's connection pool."""

        self._engine.dispose()


class DatabaseHandle:
    """Compatibility proxy exposing both DBAPI and SQLAlchemy access patterns."""

    def __init__(self, engine: DatabaseEngine):
        self._engine_wrapper = engine
        self._connection: sqlite3.Connection | None = None

    @property
    def engine(self) -> Engine:
        return self._engine_wrapper.engine

    def _get_connection(self) -> sqlite3.Connection:
        if self._connection is None:
            self._connection = self._engine_wrapper.engine.raw_connection()
        return self._connection

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def dispose(self) -> None:
        self._engine_wrapper.dispose()

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        with self._engine_wrapper.connection() as conn:
            yield conn

    @contextmanager
    def sa_connection(self) -> Iterator[Connection]:
        with self._engine_wrapper.sa_connection() as conn:
            yield conn

    @contextmanager
    def session(self) -> Iterator[Session]:
        with self._engine_wrapper.session() as session:
            yield session

    def __getattr__(self, item):
        return getattr(self._get_connection(), item)

    def __enter__(self):  # pragma: no cover - passthrough to DBAPI connection
        return self._get_connection().__enter__()

    def __exit__(self, exc_type, exc, tb):  # pragma: no cover - passthrough to DBAPI
        return self._get_connection().__exit__(exc_type, exc, tb)


_fallback_connection: DatabaseHandle | None = None
_processed_games_columns_cache: set[str] | None = None


def set_fallback_connection(conn: DatabaseHandle | None) -> None:
    """Configure the engine returned when no Flask app context is active."""

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


def build_engine_from_dsn(
    dsn: str,
    *,
    timeout: float | None = None,
    pool_size: int = 5,
    pool_recycle: int = 1_800,
    pool_pre_ping: bool = True,
) -> DatabaseEngine:
    """Return a :class:`DatabaseEngine` configured from ``dsn``."""

    parsed = urlparse(dsn)
    connect_args: dict[str, object] = {}
    effective_timeout = timeout if timeout is not None else 5.0

    if parsed.scheme == "sqlite":
        sqlite_path = _resolve_sqlite_path_from_dsn(dsn)
        normalized_dsn = f"sqlite:///{sqlite_path}"
        connect_args["check_same_thread"] = False
    else:
        normalized_dsn = dsn

    engine = create_engine(
        normalized_dsn,
        future=True,
        pool_size=pool_size,
        pool_recycle=pool_recycle,
        pool_pre_ping=pool_pre_ping,
        connect_args=connect_args,
    )

    if parsed.scheme == "sqlite":

        @event.listens_for(engine, "connect")
        def _on_connect(dbapi_conn, connection_record):  # type: ignore[override]
            dbapi_conn.row_factory = sqlite3.Row
            _configure_sqlite_connection(dbapi_conn, busy_timeout=effective_timeout)

    return DatabaseEngine(engine)


def get_db(
    connection_factory: Callable[[], DatabaseHandle | DatabaseEngine] | None = None,
    *,
    context_key: str = 'db',
    use_global_fallback: bool = True,
) -> DatabaseHandle:
    """Return the active :class:`DatabaseHandle`, creating one if necessary."""

    global _fallback_connection

    def _coerce_handle(value: DatabaseHandle | DatabaseEngine) -> DatabaseHandle:
        if isinstance(value, DatabaseHandle):
            return value
        if isinstance(value, DatabaseEngine):
            return DatabaseHandle(value)
        raise TypeError('connection_factory must return DatabaseHandle or DatabaseEngine')

    if has_app_context():
        if not hasattr(g, context_key):
            if connection_factory is not None:
                setattr(g, context_key, _coerce_handle(connection_factory()))
            elif _fallback_connection is not None:
                setattr(g, context_key, _fallback_connection)
            else:
                raise RuntimeError('Database connection is not configured')
        value = getattr(g, context_key)
        if isinstance(value, DatabaseHandle):
            return value
        if isinstance(value, DatabaseEngine):
            handle = DatabaseHandle(value)
            setattr(g, context_key, handle)
            return handle
        raise RuntimeError('Database connection is not configured correctly')

    if not use_global_fallback:
        if connection_factory is None:
            raise RuntimeError(
                'connection_factory is required when no Flask application context is active'
            )
        return _coerce_handle(connection_factory())

    if _fallback_connection is None:
        if connection_factory is None:
            raise RuntimeError('Database connection is not configured')
        _fallback_connection = _coerce_handle(connection_factory())
    return _fallback_connection


def get_processed_games_columns(
    conn: sqlite3.Connection | None = None,
    *,
    handle: DatabaseHandle | None = None,
    connection_factory: Callable[[], DatabaseHandle | DatabaseEngine] | None = None,
) -> set[str]:
    """Return the cached ``processed_games`` column names."""

    global _processed_games_columns_cache
    if _processed_games_columns_cache is not None:
        return _processed_games_columns_cache

    if conn is not None:
        cursor = conn.execute('PRAGMA table_info(processed_games)')
    else:
        if handle is None:
            handle = get_db(connection_factory)
        with handle.connection() as dbapi_conn:
            cursor = dbapi_conn.execute('PRAGMA table_info(processed_games)')
            rows = cursor.fetchall()
        _processed_games_columns_cache = {row['name'] for row in rows}
        return _processed_games_columns_cache

    _processed_games_columns_cache = {row['name'] for row in cursor.fetchall()}
    return _processed_games_columns_cache


def _quote_identifier(identifier: str) -> str:
    """Return the SQL dialect-safe quoted version of ``identifier``."""

    engine: Engine | None = None

    if has_app_context():
        context_value = getattr(g, 'db', None)
        if isinstance(context_value, DatabaseHandle):
            engine = context_value.engine
        elif isinstance(context_value, DatabaseEngine):
            engine = context_value.engine

    if engine is None and _fallback_connection is not None:
        engine = _fallback_connection.engine

    preparer = (engine.dialect.identifier_preparer if engine is not None else DefaultDialect().identifier_preparer)
    return preparer.quote(identifier)
