"""IGDB cache management utilities."""
from __future__ import annotations

import json
import numbers
from contextlib import contextmanager, suppress
from datetime import datetime, timezone
from typing import Any, Iterable, Iterator, Mapping

from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import SQLAlchemyError

from db import utils as db_utils
from igdb.client import IGDBClient

IGDB_CACHE_TABLE = "igdb_games"
IGDB_CACHE_STATE_TABLE = "igdb_cache_state"

_igdb_client = IGDBClient()


CacheConnection = (
    db_utils.DatabaseHandle
    | db_utils.DatabaseEngine
    | Connection
    | Engine
)


def _quote(name: str) -> str:
    """Return a safely quoted identifier for SQL statements."""

    return db_utils._quote_identifier(name)


@contextmanager
def _sa_connection(conn: CacheConnection) -> Iterator[tuple[Connection, bool]]:
    """Yield a SQLAlchemy connection derived from ``conn``.

    The context manager also reports whether the helper is responsible for
    committing the transaction. Connections that are created inside this helper
    need an explicit commit to persist changes, while externally managed
    connections (for example those obtained from ``engine.begin()``) should be
    left untouched so surrounding code can manage the transaction lifecycle.
    """

    if isinstance(conn, db_utils.DatabaseHandle):
        with conn.sa_connection() as sa_conn:
            yield sa_conn, True
        return
    if isinstance(conn, db_utils.DatabaseEngine):
        with conn.sa_connection() as sa_conn:
            yield sa_conn, True
        return
    if isinstance(conn, Engine):
        with conn.connect() as sa_conn:
            yield sa_conn, True
        return
    if isinstance(conn, Connection):
        yield conn, False
        return
    raise TypeError(f"Unsupported connection type: {type(conn)!r}")


def _dialect_name(conn: CacheConnection) -> str | None:
    """Return the SQLAlchemy dialect name for ``conn`` if available."""

    if isinstance(conn, db_utils.DatabaseHandle):
        return conn.engine.dialect.name
    if isinstance(conn, db_utils.DatabaseEngine):
        return conn.engine.dialect.name
    if isinstance(conn, (Engine, Connection)):
        return conn.dialect.name
    return None


def _ensure_cache_state_table(conn: CacheConnection) -> None:
    dialect = _dialect_name(conn)
    if dialect in {"mysql", "mariadb"}:
        definition = """
            id BIGINT PRIMARY KEY,
            total_count BIGINT,
            last_synced_at VARCHAR(64)
        """
    else:
        definition = """
            id BIGINT PRIMARY KEY CHECK (id = 1),
            total_count BIGINT,
            last_synced_at TEXT
        """

    create_statement = text(
        f"""
        CREATE TABLE IF NOT EXISTS {_quote(IGDB_CACHE_STATE_TABLE)} (
            {definition}
        )
        """
    )

    with _sa_connection(conn) as (sa_conn, should_commit):
        sa_conn.execute(create_statement)
        if should_commit and sa_conn.in_transaction():
            sa_conn.commit()


def _ensure_games_table(conn: CacheConnection) -> None:
    dialect = _dialect_name(conn)
    if dialect in {"mysql", "mariadb"}:
        column_definition = """
            igdb_id BIGINT PRIMARY KEY,
            name VARCHAR(255),
            summary LONGTEXT,
            updated_at BIGINT,
            first_release_date BIGINT,
            category INT,
            cover_image_id VARCHAR(255),
            rating_count INT,
            developers LONGTEXT,
            publishers LONGTEXT,
            genres LONGTEXT,
            platforms LONGTEXT,
            game_modes LONGTEXT,
            cached_at VARCHAR(64)
        """
        index_statement = text(
            f"""
            CREATE INDEX {_quote(f'{IGDB_CACHE_TABLE}_updated_at_id_idx')}
            ON {_quote(IGDB_CACHE_TABLE)} (updated_at, igdb_id)
            """
        )
    else:
        column_definition = """
            igdb_id BIGINT PRIMARY KEY,
            name TEXT,
            summary TEXT,
            updated_at BIGINT,
            first_release_date BIGINT,
            category INTEGER,
            cover_image_id TEXT,
            rating_count INTEGER,
            developers TEXT,
            publishers TEXT,
            genres TEXT,
            platforms TEXT,
            game_modes TEXT,
            cached_at TEXT
        """
        index_statement = text(
            f"""
            CREATE INDEX IF NOT EXISTS {_quote(f'{IGDB_CACHE_TABLE}_updated_at_id_idx')}
            ON {_quote(IGDB_CACHE_TABLE)} (updated_at, igdb_id)
            """
        )

    create_statement = text(
        f"""
        CREATE TABLE IF NOT EXISTS {_quote(IGDB_CACHE_TABLE)} (
            {column_definition}
        )
        """
    )

    with _sa_connection(conn) as (sa_conn, should_commit):
        sa_conn.execute(create_statement)
        if dialect in {"mysql", "mariadb"}:
            with suppress(SQLAlchemyError):
                sa_conn.execute(index_statement)
        else:
            sa_conn.execute(index_statement)
        if should_commit and sa_conn.in_transaction():
            sa_conn.commit()

    _ensure_postgres_cache_index(conn)


def _ensure_postgres_cache_index(conn: CacheConnection) -> None:
    """Ensure the Postgres cache index exists when ``conn`` targets PostgreSQL."""

    if _dialect_name(conn) != "postgresql":
        return

    index_statement = text(
        """
        CREATE INDEX IF NOT EXISTS igdb_games_updated_at_id_desc_idx
        ON igdb_games (updated_at DESC, id)
        """
    )
    analyze_statement = text("ANALYZE igdb_games")

    with _sa_connection(conn) as (sa_conn, should_commit):
        try:
            sa_conn.execute(index_statement)
            sa_conn.execute(analyze_statement)
        except SQLAlchemyError:
            return
        if should_commit and sa_conn.in_transaction():
            sa_conn.commit()


def _cache_state_upsert_statement(dialect: str | None):
    """Return an ``INSERT`` statement suited for the active SQL dialect."""

    if dialect in {"mysql", "mariadb"}:
        return text(
            f"""
            INSERT INTO {_quote(IGDB_CACHE_STATE_TABLE)}
                (id, total_count, last_synced_at)
            VALUES (:id_value, :total_value, :synced_value)
            ON DUPLICATE KEY UPDATE
                total_count = VALUES(total_count),
                last_synced_at = VALUES(last_synced_at)
            """
        )
    return text(
        f"""
        INSERT INTO {_quote(IGDB_CACHE_STATE_TABLE)}
            (id, total_count, last_synced_at)
        VALUES (:id_value, :total_value, :synced_value)
        ON CONFLICT(id) DO UPDATE SET
            total_count = excluded.total_count,
            last_synced_at = excluded.last_synced_at
        """
    )


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_cached_total(conn: Any) -> int | None:
    """Return the cached IGDB total game count, if present."""

    _ensure_cache_state_table(conn)
    row = conn.execute(
        f"SELECT total_count FROM {_quote(IGDB_CACHE_STATE_TABLE)} WHERE id = 1"
    ).fetchone()
    if row is None:
        return None
    value = None
    if hasattr(row, "keys"):
        try:
            value = row["total_count"]
        except (KeyError, IndexError, TypeError):
            pass
    if value is None:
        try:
            value = row[0]
        except (IndexError, TypeError):
            value = None
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def set_cached_total(
    conn: Any,
    total: int | None,
    *,
    synced_at: str | None = None,
) -> None:
    """Persist the IGDB total game count in the cache state table."""

    _ensure_cache_state_table(conn)
    timestamp = synced_at or _now_utc_iso()
    statement = _cache_state_upsert_statement(_dialect_name(conn))
    parameters = {
        "id_value": 1,
        "total_value": total,
        "synced_value": timestamp,
    }

    with _sa_connection(conn) as (sa_conn, should_commit):
        sa_conn.execute(statement, parameters)
        if should_commit and sa_conn.in_transaction():
            sa_conn.commit()


def get_cache_status(conn: Any) -> dict[str, Any]:
    """Return aggregate information about the local IGDB cache."""

    _ensure_cache_state_table(conn)
    _ensure_games_table(conn)

    total_row = conn.execute(
        f"SELECT COUNT(*) AS count FROM {_quote(IGDB_CACHE_TABLE)}"
    ).fetchone()
    cached_entries = _row_lookup(total_row, 'count', 0)
    try:
        cached_total = max(int(cached_entries), 0)
    except (TypeError, ValueError):
        cached_total = 0

    state_row = conn.execute(
        f"SELECT total_count, last_synced_at FROM {_quote(IGDB_CACHE_STATE_TABLE)} WHERE id = 1"
    ).fetchone()
    remote_total = _row_lookup(state_row, 'total_count', 0)
    last_synced = _row_lookup(state_row, 'last_synced_at', 1)

    result: dict[str, Any] = {
        'cached_entries': cached_total,
        'remote_total': None,
        'last_synced_at': str(last_synced) if last_synced else None,
    }

    try:
        if remote_total is not None:
            result['remote_total'] = max(int(remote_total), 0)
    except (TypeError, ValueError):
        result['remote_total'] = None

    return result


def _serialize_cache_list(values: Iterable[Any]) -> str:
    items: list[str] = []
    for value in values or []:
        text = str(value).strip()
        if text:
            items.append(text)
    return json.dumps(items, ensure_ascii=False)


def _normalize_payload(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    if "id" in payload:
        return dict(payload)
    normalized = _igdb_client.normalize_game(payload)
    if normalized is None:
        return None
    return normalized


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = value.strip() if isinstance(value, str) else str(value).strip()
    return text or None


def _build_row_from_payload(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    normalized = _normalize_payload(payload)
    if not normalized:
        return None

    igdb_id = normalized.get("id")
    try:
        numeric_id = int(igdb_id)
    except (TypeError, ValueError):
        return None

    cover = normalized.get("cover")
    cover_image_id: str | None = None
    if isinstance(cover, Mapping):
        cover_value = cover.get("image_id") or cover.get("imageId")
        if cover_value is not None:
            cover_image_id = str(cover_value).strip() or None
    elif isinstance(cover, str):
        cover_image_id = cover.strip() or None

    rating_value = normalized.get("rating_count")
    if isinstance(rating_value, numbers.Number):
        rating_count = int(rating_value)
    else:
        try:
            rating_count = int(str(rating_value).strip())
        except (TypeError, ValueError):
            rating_count = None

    return {
        "igdb_id": numeric_id,
        "name": _clean_text(normalized.get("name")),
        "summary": _clean_text(normalized.get("summary")),
        "updated_at": normalized.get("updated_at"),
        "first_release_date": normalized.get("first_release_date"),
        "category": normalized.get("category"),
        "cover_image_id": cover_image_id,
        "rating_count": rating_count,
        "developers_json": _serialize_cache_list(normalized.get("developers") or []),
        "publishers_json": _serialize_cache_list(normalized.get("publishers") or []),
        "genres_json": _serialize_cache_list(normalized.get("genres") or []),
        "platforms_json": _serialize_cache_list(normalized.get("platforms") or []),
        "game_modes_json": _serialize_cache_list(normalized.get("game_modes") or []),
    }


def _row_lookup(row: Any, key: str, index: int) -> Any:
    if row is None:
        return None
    if hasattr(row, "keys"):
        try:
            return row[key]
        except (KeyError, IndexError, TypeError):
            pass
    try:
        return row[index]
    except (IndexError, TypeError):
        return None


def upsert_igdb_games(
    conn: Any,
    games: Iterable[Mapping[str, Any]],
) -> tuple[int, int, int]:
    """Insert or update IGDB cache entries based on payloads."""

    _ensure_games_table(conn)
    inserted = updated = unchanged = 0
    columns = (
        "name",
        "summary",
        "updated_at",
        "first_release_date",
        "category",
        "cover_image_id",
        "rating_count",
        "developers",
        "publishers",
        "genres",
        "platforms",
        "game_modes",
    )

    for payload in games:
        row = _build_row_from_payload(payload)
        if row is None:
            continue
        igdb_id = row["igdb_id"]
        existing = conn.execute(
            f"""
            SELECT {', '.join(columns)}
            FROM {_quote(IGDB_CACHE_TABLE)}
            WHERE igdb_id = ?
            """,
            (igdb_id,),
        ).fetchone()
        cached_at = _now_utc_iso()
        params = (
            row["name"],
            row["summary"],
            row["updated_at"],
            row["first_release_date"],
            row["category"],
            row["cover_image_id"],
            row["rating_count"],
            row["developers_json"],
            row["publishers_json"],
            row["genres_json"],
            row["platforms_json"],
            row["game_modes_json"],
        )
        if existing is None:
            conn.execute(
                f"""
                INSERT INTO {_quote(IGDB_CACHE_TABLE)} (
                    igdb_id, {', '.join(columns)}, cached_at
                ) VALUES (?, {', '.join('?' for _ in columns)}, ?)
                """,
                (igdb_id, *params, cached_at),
            )
            inserted += 1
            continue

        existing_values = tuple(
            _row_lookup(existing, column, idx) for idx, column in enumerate(columns)
        )
        if existing_values == params:
            unchanged += 1
            continue

        conn.execute(
            f"""
            UPDATE {_quote(IGDB_CACHE_TABLE)}
               SET name=?, summary=?, updated_at=?, first_release_date=?,
                   category=?, cover_image_id=?, rating_count=?,
                   developers=?, publishers=?, genres=?, platforms=?,
                   game_modes=?, cached_at=?
             WHERE igdb_id=?
            """,
            (*params, cached_at, igdb_id),
        )
        updated += 1

    return inserted, updated, unchanged
