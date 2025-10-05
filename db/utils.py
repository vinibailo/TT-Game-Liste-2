"""Shared helpers for working with the processed-games database."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from collections.abc import Mapping
from typing import Any, Callable, Iterator, Sequence

try:  # pragma: no cover - optional dependency
    import sqlite3
except ImportError:  # pragma: no cover - environments without sqlite bindings
    sqlite3 = None  # type: ignore[assignment]

from flask import g, has_app_context
from sqlalchemy import create_engine, event, inspect
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.engine.default import DefaultDialect
from sqlalchemy.orm import Session, sessionmaker
from urllib.parse import unquote, urlparse

db_lock = Lock()
"""Module-level lock to guard write access to the processed-games database."""


class _DBRow(Mapping[str, Any]):
    """Lightweight row wrapper supporting mapping-style access."""

    __slots__ = ("_columns", "_values", "_mapping")

    def __init__(self, columns: Sequence[str], values: Sequence[Any]):
        self._columns = list(columns)
        self._values = list(values)
        self._mapping = dict(zip(self._columns, self._values))

    def __getitem__(self, key: int | str) -> Any:
        if isinstance(key, int):
            return self._values[key]
        return self._mapping[key]

    def get(self, key: str, default: Any | None = None) -> Any | None:
        return self._mapping.get(key, default)

    def keys(self) -> Sequence[str]:  # pragma: no cover - helper for debugging
        return list(self._columns)

    def values(self) -> Sequence[Any]:  # pragma: no cover - helper for debugging
        return list(self._values)

    def items(self):  # pragma: no cover - helper for debugging
        return self._mapping.items()

    def __contains__(self, item: object) -> bool:  # pragma: no cover - rarely used
        return item in self._mapping

    def __iter__(self):
        return iter(self._columns)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._columns)

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"_DBRow({self._mapping!r})"


class _CursorWrapper:
    """Thin wrapper normalizing DB-API cursor behaviour."""

    def __init__(self, cursor: Any):
        self._cursor = cursor
        description = cursor.description or []
        self._columns = [col[0] for col in description]

    def _wrap_row(self, values: Sequence[Any]) -> _DBRow:
        return _DBRow(self._columns, values)

    def fetchone(self) -> _DBRow | None:
        row = self._cursor.fetchone()
        if row is None:
            return None
        return self._wrap_row(row)

    def fetchmany(self, size: int | None = None) -> list[_DBRow]:
        rows = self._cursor.fetchmany(size) if size is not None else self._cursor.fetchmany()
        return [self._wrap_row(row) for row in rows]

    def fetchall(self) -> list[_DBRow]:
        rows = self._cursor.fetchall()
        return [self._wrap_row(row) for row in rows]

    def close(self) -> None:
        try:
            self._cursor.close()
        except Exception:  # pragma: no cover - DBAPI edge cases
            pass

    @property
    def rowcount(self) -> int:  # pragma: no cover - passthrough
        return getattr(self._cursor, "rowcount", -1)

    @property
    def lastrowid(self) -> Any:  # pragma: no cover - passthrough
        return getattr(self._cursor, "lastrowid", None)

    def __iter__(self):  # pragma: no cover - rarely used
        for row in self._cursor:
            yield self._wrap_row(row)

    def __getattr__(self, item):  # pragma: no cover - passthrough
        return getattr(self._cursor, item)


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
    def connection(self) -> Iterator[Any]:
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
        self._connection: Any | None = None

    @property
    def engine(self) -> Engine:
        return self._engine_wrapper.engine

    def _get_connection(self) -> Any:
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
    def connection(self) -> Iterator[Any]:
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

    def _normalize_sql(self, sql: str) -> str:
        dialect = self.engine.dialect
        paramstyle = getattr(dialect, "paramstyle", "qmark")
        if paramstyle in {"format", "pyformat"} and "?" in sql:
            return sql.replace("?", "%s")
        return sql

    def _wrap_cursor(self, cursor: Any) -> _CursorWrapper:
        try:
            description = cursor.description
        except AttributeError:  # pragma: no cover - DBAPI without description attribute
            return cursor
        if description is None:
            return cursor
        return _CursorWrapper(cursor)

    def execute(
        self,
        sql: str,
        parameters: Sequence[Any] | Mapping[str, Any] | None = None,
    ):
        conn = self._get_connection()
        if hasattr(conn, "execute"):
            return conn.execute(sql, parameters or ())

        cursor = conn.cursor()
        try:
            cursor.execute(self._normalize_sql(sql), parameters or ())
        except Exception:
            cursor.close()
            raise
        return self._wrap_cursor(cursor)

    def executemany(
        self,
        sql: str,
        seq_of_parameters: Sequence[Sequence[Any] | Mapping[str, Any]],
    ):
        conn = self._get_connection()
        if hasattr(conn, "executemany"):
            return conn.executemany(sql, seq_of_parameters)

        cursor = conn.cursor()
        try:
            cursor.executemany(self._normalize_sql(sql), seq_of_parameters)
        except Exception:
            cursor.close()
            raise
        return cursor

    def __getattr__(self, item):
        return getattr(self._get_connection(), item)

    def __enter__(self):  # pragma: no cover - passthrough to DBAPI connection
        return self._get_connection().__enter__()

    def __exit__(self, exc_type, exc, tb):  # pragma: no cover - passthrough to DBAPI
        return self._get_connection().__exit__(exc_type, exc, tb)


_fallback_connection: DatabaseHandle | DatabaseEngine | None = None
_fallback_handle_cache: DatabaseHandle | None = None
_processed_games_columns_cache: set[str] | None = None


def set_fallback_connection(conn: DatabaseHandle | DatabaseEngine | None) -> None:
    """Configure the engine returned when no Flask app context is active."""

    global _fallback_connection
    global _fallback_handle_cache

    _fallback_connection = conn
    if conn is None:
        _fallback_handle_cache = None
    elif isinstance(conn, DatabaseHandle):
        _fallback_handle_cache = conn
    else:
        _fallback_handle_cache = None


def clear_processed_games_columns_cache() -> None:
    """Reset the cached ``processed_games`` column names."""

    global _processed_games_columns_cache
    _processed_games_columns_cache = None


def _configure_sqlite_connection(conn: Any, *, busy_timeout: float | None = None) -> Any:
    """Apply timeout tuning to SQLite connections when available."""

    if sqlite3 is None or not isinstance(conn, sqlite3.Connection):
        return conn

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
                cursor.fetchone()
        except sqlite3.OperationalError:  # pragma: no cover - best effort only
            continue

    return conn


def _configure_mariadb_connection(conn: Any, *, lock_timeout: float | None = None) -> Any:
    """Apply session-level settings for MariaDB connections."""

    if lock_timeout is None:
        return conn

    timeout_value = max(int(lock_timeout), 1)

    try:
        cursor = conn.cursor()
    except AttributeError:  # pragma: no cover - DBAPI without cursor helper
        return conn

    try:
        try:
            cursor.execute("SET SESSION innodb_lock_wait_timeout = %s", (timeout_value,))
        except Exception:  # pragma: no cover - unavailable variable
            pass
        try:
            cursor.execute("SET SESSION lock_wait_timeout = %s", (timeout_value,))
        except Exception:  # pragma: no cover - unavailable variable
            pass
        try:
            cursor.execute("SET SESSION wait_timeout = %s", (timeout_value,))
        except Exception:  # pragma: no cover - unavailable variable
            pass
    finally:
        try:
            cursor.close()
        except Exception:  # pragma: no cover - DBAPI edge case
            pass

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

    dialect_name = parsed.scheme.split("+", 1)[0]

    engine = create_engine(
        normalized_dsn,
        future=True,
        pool_size=pool_size,
        pool_recycle=pool_recycle,
        pool_pre_ping=pool_pre_ping,
        connect_args=connect_args,
    )

    if parsed.scheme == "sqlite" and sqlite3 is not None:

        @event.listens_for(engine, "connect")
        def _on_connect(dbapi_conn, connection_record):  # type: ignore[override]
            try:
                dbapi_conn.row_factory = sqlite3.Row
            except AttributeError:  # pragma: no cover - unexpected DB-API
                pass
            _configure_sqlite_connection(dbapi_conn, busy_timeout=effective_timeout)
    elif dialect_name in {"mysql", "mariadb"}:

        @event.listens_for(engine, "connect")
        def _on_connect(dbapi_conn, connection_record):  # type: ignore[override]
            _configure_mariadb_connection(dbapi_conn, lock_timeout=effective_timeout)

    return DatabaseEngine(engine)


def get_db(
    connection_factory: Callable[[], DatabaseHandle | DatabaseEngine] | None = None,
    *,
    context_key: str = 'db',
    use_global_fallback: bool = True,
) -> DatabaseHandle:
    """Return the active :class:`DatabaseHandle`, creating one if necessary."""

    global _fallback_connection
    global _fallback_handle_cache

    def _coerce_handle(value: DatabaseHandle | DatabaseEngine) -> DatabaseHandle:
        if isinstance(value, DatabaseHandle):
            return value
        if isinstance(value, DatabaseEngine):
            return DatabaseHandle(value)
        raise TypeError('connection_factory must return DatabaseHandle or DatabaseEngine')

    def _fallback_handle() -> DatabaseHandle:
        global _fallback_handle_cache

        if _fallback_connection is None:
            raise RuntimeError('Database connection is not configured')
        if isinstance(_fallback_connection, DatabaseHandle):
            _fallback_handle_cache = _fallback_connection
            return _fallback_connection
        if _fallback_handle_cache is None:
            _fallback_handle_cache = DatabaseHandle(_fallback_connection)
        return _fallback_handle_cache

    if has_app_context():
        if not hasattr(g, context_key):
            if connection_factory is not None:
                setattr(g, context_key, _coerce_handle(connection_factory()))
            elif _fallback_connection is not None:
                setattr(g, context_key, _fallback_handle())
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
        _fallback_connection = connection_factory()
        if isinstance(_fallback_connection, DatabaseHandle):
            _fallback_handle_cache = _fallback_connection
        else:
            _fallback_handle_cache = None
    return _fallback_handle()


@contextmanager
def get_db_connection(
    connection_factory: Callable[[], DatabaseHandle | DatabaseEngine] | None = None,
    *,
    context_key: str = 'db',
    use_global_fallback: bool = True,
) -> Iterator[Any]:
    """Yield a DBAPI connection from the active database handle."""

    handle = get_db(
        connection_factory,
        context_key=context_key,
        use_global_fallback=use_global_fallback,
    )
    with handle.connection() as conn:
        yield conn


@contextmanager
def get_db_sa_connection(
    connection_factory: Callable[[], DatabaseHandle | DatabaseEngine] | None = None,
    *,
    context_key: str = 'db',
    use_global_fallback: bool = True,
) -> Iterator[Connection]:
    """Yield a SQLAlchemy :class:`~sqlalchemy.engine.Connection`."""

    handle = get_db(
        connection_factory,
        context_key=context_key,
        use_global_fallback=use_global_fallback,
    )
    with handle.sa_connection() as conn:
        yield conn


@contextmanager
def get_db_session(
    connection_factory: Callable[[], DatabaseHandle | DatabaseEngine] | None = None,
    *,
    context_key: str = 'db',
    use_global_fallback: bool = True,
) -> Iterator[Session]:
    """Yield a SQLAlchemy :class:`~sqlalchemy.orm.Session`."""

    handle = get_db(
        connection_factory,
        context_key=context_key,
        use_global_fallback=use_global_fallback,
    )
    with handle.session() as session:
        yield session


def get_processed_games_columns(
    conn: Connection | DatabaseHandle | None = None,
    *,
    handle: DatabaseHandle | None = None,
    connection_factory: Callable[[], DatabaseHandle | DatabaseEngine] | None = None,
) -> set[str]:
    """Return the cached ``processed_games`` column names."""

    global _processed_games_columns_cache
    if _processed_games_columns_cache is not None:
        return _processed_games_columns_cache

    if isinstance(conn, DatabaseHandle):
        handle = conn
        conn = None

    if isinstance(conn, Connection):
        inspector = inspect(conn)
        _processed_games_columns_cache = {
            col['name'] for col in inspector.get_columns('processed_games')
        }
        return _processed_games_columns_cache

    if handle is None:
        handle = get_db(connection_factory)

    with handle.sa_connection() as sa_conn:
        inspector = inspect(sa_conn)
        _processed_games_columns_cache = {col['name'] for col in inspector.get_columns('processed_games')}
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
