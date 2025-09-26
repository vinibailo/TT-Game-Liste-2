"""IGDB cache management utilities."""
from __future__ import annotations

import json
import numbers
import sqlite3
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

from db import utils as db_utils
from igdb.client import IGDBClient

IGDB_CACHE_TABLE = "igdb_games"
IGDB_CACHE_STATE_TABLE = "igdb_cache_state"

_igdb_client = IGDBClient()


def _quote(name: str) -> str:
    """Return a safely quoted identifier for SQLite statements."""

    return db_utils._quote_identifier(name)  # type: ignore[attr-defined]


def _ensure_cache_state_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {_quote(IGDB_CACHE_STATE_TABLE)} (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            total_count INTEGER,
            last_synced_at TEXT
        )
        """
    )


def _ensure_games_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {_quote(IGDB_CACHE_TABLE)} (
            igdb_id INTEGER PRIMARY KEY,
            name TEXT,
            summary TEXT,
            updated_at INTEGER,
            first_release_date INTEGER,
            category INTEGER,
            cover_image_id TEXT,
            rating_count INTEGER,
            developers TEXT,
            publishers TEXT,
            genres TEXT,
            platforms TEXT,
            game_modes TEXT,
            cached_at TEXT
        )
        """
    )


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_cached_total(conn: sqlite3.Connection) -> int | None:
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
    conn: sqlite3.Connection, total: int | None, *, synced_at: str | None = None
) -> None:
    """Persist the IGDB total game count in the cache state table."""

    _ensure_cache_state_table(conn)
    timestamp = synced_at or _now_utc_iso()
    conn.execute(
        f"""
        INSERT INTO {_quote(IGDB_CACHE_STATE_TABLE)} (id, total_count, last_synced_at)
        VALUES (1, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            total_count=excluded.total_count,
            last_synced_at=excluded.last_synced_at
        """,
        (total, timestamp),
    )


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
    conn: sqlite3.Connection, games: Iterable[Mapping[str, Any]]
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
