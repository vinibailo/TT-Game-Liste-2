"""Lookup-table persistence and service-layer helpers."""

from __future__ import annotations

import logging
import numbers
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence

import sqlite3

import pandas as pd

from db import utils as db_utils

logger = logging.getLogger(__name__)

SQLALCHEMY_AVAILABLE = True
try:  # pragma: no cover - import guard for optional dependency during bootstrap
    from sqlalchemy import (
        MetaData,
        Table,
        create_engine,
        func,
        insert,
        select,
        update as sa_update,
        delete as sa_delete,
    )
    from sqlalchemy.engine import Engine
    from sqlalchemy.exc import SQLAlchemyError
    from sqlalchemy.pool import StaticPool
except Exception:  # pragma: no cover - surfaced by tests when dependency missing
    SQLALCHEMY_AVAILABLE = False
    MetaData = Table = Engine = object  # type: ignore[assignment]
    create_engine = StaticPool = None  # type: ignore[assignment]
    func = insert = select = sa_update = sa_delete = None  # type: ignore[assignment]


class LookupServiceError(RuntimeError):
    """Base class for lookup service errors."""


class LookupConflictError(LookupServiceError):
    """Raised when attempting to rename a lookup to an existing value."""


class LookupNotFoundError(LookupServiceError):
    """Raised when a lookup entry cannot be located."""


_ENGINE_CACHE: dict[int, Engine] = {}
_TABLE_CACHE: dict[tuple[int, str], Table] = {}


def _engine_from_connection(conn: sqlite3.Connection) -> Engine:
    """Return a SQLAlchemy ``Engine`` bound to ``conn``."""

    if not SQLALCHEMY_AVAILABLE:
        raise LookupServiceError("SQLAlchemy engine is unavailable")

    key = id(conn)
    if key not in _ENGINE_CACHE:
        _ENGINE_CACHE[key] = create_engine(
            "sqlite://",
            creator=lambda: conn,
            poolclass=StaticPool,
            future=True,
        )
    return _ENGINE_CACHE[key]


def _table_from_connection(conn: sqlite3.Connection, table_name: str) -> Table:
    """Reflect ``table_name`` using SQLAlchemy metadata."""

    if not SQLALCHEMY_AVAILABLE:
        raise LookupServiceError("SQLAlchemy metadata reflection is unavailable")

    cache_key = (id(conn), table_name)
    if cache_key in _TABLE_CACHE:
        return _TABLE_CACHE[cache_key]

    engine = _engine_from_connection(conn)
    metadata = MetaData()
    try:
        table = Table(table_name, metadata, autoload_with=engine)
    except (SQLAlchemyError, TypeError):
        raise LookupNotFoundError(f"lookup table not available: {table_name}")
    _TABLE_CACHE[cache_key] = table
    return table


def list_lookup_entries(
    conn: sqlite3.Connection,
    table_name: str,
    *,
    normalize_lookup_name: Callable[[Any], str],
    limit: int = 200,
    offset: int = 0,
) -> tuple[list[dict[str, Any]], int]:
    """Return paginated lookup entries ordered by name for ``table_name``."""

    if SQLALCHEMY_AVAILABLE:
        try:
            table = _table_from_connection(conn, table_name)
        except LookupNotFoundError:
            return ([], 0)

        engine = _engine_from_connection(conn)
        try:
            with engine.connect() as sa_conn:
                result = sa_conn.execute(
                    select(table.c.id, table.c.name).order_by(
                        func.lower(table.c.name), table.c.id
                    )
                )
                rows = list(result)
        except SQLAlchemyError:
            return ([], 0)
    else:
        try:
            cur = conn.execute(
                f'SELECT id, name FROM {table_name} ORDER BY name COLLATE NOCASE, id'
            )
        except sqlite3.OperationalError:
            return ([], 0)
        rows = cur.fetchall()

    items: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    for row in rows:
        raw_id = row[0]
        lookup_id = _coerce_int(raw_id)
        if lookup_id is None or lookup_id in seen_ids:
            continue
        name = normalize_lookup_name(row[1])
        if not name:
            continue
        seen_ids.add(lookup_id)
        items.append({"id": lookup_id, "name": name})

    total = len(items)
    if offset < 0:
        offset = 0
    if limit <= 0:
        paginated_items: list[dict[str, Any]] = []
    else:
        start = min(offset, total)
        end = start + limit
        paginated_items = items[start:end]

    return paginated_items, total


def get_lookup_entry(
    conn: sqlite3.Connection,
    table_name: str,
    lookup_id: int,
    *,
    normalize_lookup_name: Callable[[Any], str],
) -> dict[str, Any] | None:
    """Return the lookup entry identified by ``lookup_id``."""

    if SQLALCHEMY_AVAILABLE:
        try:
            table = _table_from_connection(conn, table_name)
        except LookupNotFoundError:
            return None

        engine = _engine_from_connection(conn)
        try:
            with engine.connect() as sa_conn:
                row = sa_conn.execute(
                    select(table.c.id, table.c.name).where(table.c.id == lookup_id)
                ).first()
        except SQLAlchemyError:
            return None
    else:
        try:
            cur = conn.execute(
                f'SELECT id, name FROM {table_name} WHERE id = ?', (lookup_id,)
            )
        except sqlite3.OperationalError:
            return None
        row = cur.fetchone()

    if row is None:
        return None

    coerced = _coerce_int(row[0])
    if coerced is None:
        return None
    name = normalize_lookup_name(row[1])
    if not name:
        return None
    return {"id": coerced, "name": name}


def _lookup_by_name(
    conn: sqlite3.Connection,
    table_name: str,
    normalized_name: str,
) -> tuple[int, str] | None:
    """Return ``(id, name)`` for an entry matching ``normalized_name``."""

    if SQLALCHEMY_AVAILABLE:
        try:
            table = _table_from_connection(conn, table_name)
        except LookupNotFoundError:
            return None

        engine = _engine_from_connection(conn)
        try:
            with engine.connect() as sa_conn:
                row = sa_conn.execute(
                    select(table.c.id, table.c.name).where(
                        func.lower(table.c.name) == normalized_name.casefold()
                    )
                ).first()
        except SQLAlchemyError:
            return None
    else:
        try:
            cur = conn.execute(
                f'SELECT id, name FROM {table_name} WHERE name = ? COLLATE NOCASE',
                (normalized_name,),
            )
        except sqlite3.OperationalError:
            return None
        row = cur.fetchone()

    if row is None:
        return None

    lookup_id = _coerce_int(row[0])
    if lookup_id is None:
        return None
    name_value = row[1]
    return lookup_id, str(name_value) if name_value is not None else ""


def get_or_create_lookup_id(
    conn: sqlite3.Connection,
    table_name: str,
    raw_name: str,
    *,
    normalize_lookup_name: Callable[[Any], str],
) -> int | None:
    """Return the identifier for ``raw_name``, creating a lookup when needed."""

    name = normalize_lookup_name(raw_name)
    if not name:
        return None

    existing = _lookup_by_name(conn, table_name, name)
    if existing is not None:
        return existing[0]

    if SQLALCHEMY_AVAILABLE:
        try:
            table = _table_from_connection(conn, table_name)
        except LookupNotFoundError:
            return None

        engine = _engine_from_connection(conn)
        try:
            with engine.begin() as sa_conn:
                result = sa_conn.execute(insert(table).values(name=name))
                inserted_id = result.inserted_primary_key[0] if result.inserted_primary_key else None
                if inserted_id is None:
                    inserted_id = sa_conn.execute(select(func.max(table.c.id))).scalar_one()
        except SQLAlchemyError:
            return None
        return _coerce_int(inserted_id)

    try:
        cur = conn.execute(
            f'INSERT INTO {table_name} (name) VALUES (?)',
            (name,),
        )
    except sqlite3.OperationalError:
        return None
    return _coerce_int(cur.lastrowid)


def lookup_name_for_id(
    conn: sqlite3.Connection,
    table_name: str,
    lookup_id: int,
    *,
    normalize_lookup_name: Callable[[Any], str],
) -> str | None:
    """Return the normalized name for ``lookup_id`` in ``table_name``."""

    entry = get_lookup_entry(
        conn, table_name, lookup_id, normalize_lookup_name=normalize_lookup_name
    )
    if entry is None:
        return None
    return entry["name"]


def iter_lookup_payload(
    raw_value: Any,
    *,
    normalize_lookup_name: Callable[[Any], str],
) -> list[dict[str, Any]]:
    """Normalize lookup selections from user-provided payloads."""

    if raw_value is None:
        return []

    if isinstance(raw_value, Mapping):
        if "selected" in raw_value:
            return iter_lookup_payload(
                raw_value["selected"], normalize_lookup_name=normalize_lookup_name
            )
        if "entries" in raw_value:
            return iter_lookup_payload(
                raw_value["entries"], normalize_lookup_name=normalize_lookup_name
            )

        ids_value = raw_value.get("ids")
        names_value = raw_value.get("names")
        results: list[dict[str, Any]] = []

        if isinstance(ids_value, (list, tuple)):
            for idx, entry_id in enumerate(ids_value):
                entry_name = None
                if isinstance(names_value, (list, tuple)) and idx < len(names_value):
                    entry_name = names_value[idx]
                results.append({"id": entry_id, "name": entry_name})
            return results

        if isinstance(names_value, (list, tuple)):
            for name in names_value:
                results.append({"id": None, "name": name})
            return results

        entry_id = raw_value.get("id")
        entry_name = raw_value.get("name")
        if entry_id is not None or entry_name is not None:
            return [{"id": entry_id, "name": entry_name}]
        return []

    if isinstance(raw_value, str):
        text = normalize_lookup_name(raw_value)
        return [{"id": None, "name": text}] if text else []

    if isinstance(raw_value, numbers.Number) and not isinstance(raw_value, bool):
        try:
            return [{"id": int(raw_value), "name": None}]
        except (TypeError, ValueError):
            return []

    try:
        iterator = iter(raw_value)
    except TypeError:
        text = normalize_lookup_name(raw_value)
        return [{"id": None, "name": text}] if text else []

    results: list[dict[str, Any]] = []
    for item in iterator:
        results.extend(
            iter_lookup_payload(item, normalize_lookup_name=normalize_lookup_name)
        )
    return results


def format_lookup_response(
    entries: list[dict[str, Any]],
    *,
    normalize_lookup_name: Callable[[Any], str],
) -> dict[str, Any]:
    """Return a formatted payload for lookup selections."""

    formatted: list[dict[str, Any]] = []
    names: list[str] = []
    ids: list[int] = []
    seen_ids: set[int] = set()
    seen_names: set[str] = set()

    for entry in entries:
        entry_id = entry.get("id")
        if entry_id is not None:
            try:
                entry_id = int(entry_id)
            except (TypeError, ValueError):
                entry_id = None

        entry_name = normalize_lookup_name(entry.get("name"))
        if entry_id is None and not entry_name:
            continue

        if entry_id is not None:
            if entry_id in seen_ids:
                continue
            seen_ids.add(entry_id)
            ids.append(entry_id)
        else:
            fingerprint = entry_name.casefold()
            if fingerprint in seen_names:
                continue
            seen_names.add(fingerprint)

        if entry_name:
            names.append(entry_name)
        formatted.append({"id": entry_id, "name": entry_name})

    return {
        "selected": formatted,
        "names": names,
        "ids": ids,
    }


def resolve_lookup_selection(
    conn: sqlite3.Connection,
    relation: Mapping[str, Any],
    raw_value: Any,
    *,
    normalize_lookup_name: Callable[[Any], str],
) -> dict[str, Any]:
    """Resolve selection payload into lookup IDs and names."""

    lookup_table = relation["lookup_table"]
    entries = iter_lookup_payload(raw_value, normalize_lookup_name=normalize_lookup_name)
    resolved: list[dict[str, Any]] = []

    for entry in entries:
        entry_id = entry.get("id")
        entry_name = normalize_lookup_name(entry.get("name"))
        coerced_id = _coerce_int(entry_id)

        if coerced_id is not None:
            lookup_name = lookup_name_for_id(
                conn,
                lookup_table,
                coerced_id,
                normalize_lookup_name=normalize_lookup_name,
            )
            if lookup_name:
                resolved.append({"id": coerced_id, "name": lookup_name})
                continue
            coerced_id = None

        if not entry_name:
            continue

        lookup_id = get_or_create_lookup_id(
            conn,
            lookup_table,
            entry_name,
            normalize_lookup_name=normalize_lookup_name,
        )
        if lookup_id is None:
            continue

        lookup_name = (
            lookup_name_for_id(
                conn,
                lookup_table,
                lookup_id,
                normalize_lookup_name=normalize_lookup_name,
            )
            or entry_name
        )
        resolved.append({"id": lookup_id, "name": lookup_name})

    deduped: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    for entry in resolved:
        entry_id = entry["id"]
        if entry_id in seen_ids:
            continue
        seen_ids.add(entry_id)
        deduped.append(entry)

    names = [entry["name"] for entry in deduped if entry["name"]]
    ids: list[int] = []
    for entry in deduped:
        coerced = _coerce_int(entry.get("id"))
        if coerced is None:
            continue
        ids.append(coerced)

    return {"entries": deduped, "names": names, "ids": ids}


def create_lookup_entry(
    conn: sqlite3.Connection,
    table_name: str,
    raw_name: str,
    *,
    normalize_lookup_name: Callable[[Any], str],
) -> tuple[str, dict[str, Any] | None]:
    """Create a lookup entry when missing and return its payload."""

    name = normalize_lookup_name(raw_name)
    if not name:
        return "invalid", None

    existing = _lookup_by_name(conn, table_name, name)
    if existing is not None:
        lookup_id, stored_name = existing
        normalized = normalize_lookup_name(stored_name)
        return "exists", {"id": lookup_id, "name": normalized or name}

    if SQLALCHEMY_AVAILABLE:
        try:
            table = _table_from_connection(conn, table_name)
        except LookupNotFoundError:
            return "invalid", None

        engine = _engine_from_connection(conn)
        try:
            with engine.begin() as sa_conn:
                result = sa_conn.execute(insert(table).values(name=name))
                inserted_id = result.inserted_primary_key[0] if result.inserted_primary_key else None
                if inserted_id is None:
                    inserted_id = sa_conn.execute(select(func.max(table.c.id))).scalar_one()
                row = sa_conn.execute(
                    select(table.c.id, table.c.name).where(table.c.id == inserted_id)
                ).first()
        except SQLAlchemyError:
            return "invalid", None

        if row is None:
            return "invalid", None

        lookup_id = _coerce_int(row[0])
        if lookup_id is None:
            return "invalid", None
        normalized = normalize_lookup_name(row[1])
        return "created", {"id": lookup_id, "name": normalized or name}

    try:
        cur = conn.execute(
            f'INSERT INTO {table_name} (name) VALUES (?)',
            (name,),
        )
    except sqlite3.OperationalError:
        return "invalid", None
    lookup_id = _coerce_int(cur.lastrowid)
    if lookup_id is None:
        return "invalid", None
    try:
        row = conn.execute(
            f'SELECT id, name FROM {table_name} WHERE id = ?', (lookup_id,)
        ).fetchone()
    except sqlite3.OperationalError:
        row = None
    normalized = normalize_lookup_name(row[1]) if row is not None else name
    return "created", {"id": lookup_id, "name": normalized or name}


def update_lookup_entry(
    conn: sqlite3.Connection,
    table_name: str,
    lookup_id: int,
    new_name: str,
    *,
    normalize_lookup_name: Callable[[Any], str],
) -> tuple[str, dict[str, Any] | None]:
    """Update ``lookup_id`` to ``new_name`` when possible."""

    name = normalize_lookup_name(new_name)
    if not name:
        return "invalid", None

    if SQLALCHEMY_AVAILABLE:
        try:
            table = _table_from_connection(conn, table_name)
        except LookupNotFoundError:
            return "invalid", None

        engine = _engine_from_connection(conn)
        try:
            with engine.begin() as sa_conn:
                row = sa_conn.execute(
                    select(table.c.id, table.c.name).where(table.c.id == lookup_id)
                ).first()
                if row is None:
                    return "not_found", None
                existing_name = normalize_lookup_name(row[1])
                if existing_name != name:
                    conflict = sa_conn.execute(
                        select(table.c.id).where(
                            func.lower(table.c.name) == name.casefold(), table.c.id != lookup_id
                        )
                    ).first()
                    if conflict is not None:
                        return "conflict", None
                    sa_conn.execute(
                        sa_update(table)
                        .where(table.c.id == lookup_id)
                        .values(name=name)
                    )
                refreshed = sa_conn.execute(
                    select(table.c.id, table.c.name).where(table.c.id == lookup_id)
                ).first()
        except SQLAlchemyError:
            return "invalid", None

        if refreshed is None:
            return "invalid", None

        final_id = _coerce_int(refreshed[0])
        if final_id is None:
            return "invalid", None
        final_name = normalize_lookup_name(refreshed[1]) or name
        return "updated", {"id": final_id, "name": final_name}

    try:
        row = conn.execute(
            f'SELECT id, name FROM {table_name} WHERE id = ?', (lookup_id,)
        ).fetchone()
    except sqlite3.OperationalError:
        return "invalid", None
    if row is None:
        return "not_found", None
    existing_name = normalize_lookup_name(row[1])
    if existing_name != name:
        try:
            conflict = conn.execute(
                f'SELECT id FROM {table_name} WHERE name = ? COLLATE NOCASE AND id != ?',
                (name, lookup_id),
            ).fetchone()
        except sqlite3.OperationalError:
            conflict = None
        if conflict is not None:
            return "conflict", None
        try:
            conn.execute(
                f'UPDATE {table_name} SET name = ? WHERE id = ?',
                (name, lookup_id),
            )
        except sqlite3.OperationalError:
            return "invalid", None
    try:
        refreshed = conn.execute(
            f'SELECT id, name FROM {table_name} WHERE id = ?', (lookup_id,)
        ).fetchone()
    except sqlite3.OperationalError:
        return "invalid", None
    if refreshed is None:
        return "invalid", None
    final_id = _coerce_int(refreshed[0])
    if final_id is None:
        return "invalid", None
    final_name = normalize_lookup_name(refreshed[1]) or name
    return "updated", {"id": final_id, "name": final_name}


def delete_lookup_entry(
    conn: sqlite3.Connection,
    table_name: str,
    lookup_id: int,
) -> bool:
    """Remove ``lookup_id`` from ``table_name``."""

    if SQLALCHEMY_AVAILABLE:
        try:
            table = _table_from_connection(conn, table_name)
        except LookupNotFoundError:
            return False

        engine = _engine_from_connection(conn)
        try:
            with engine.begin() as sa_conn:
                row = sa_conn.execute(
                    select(table.c.id).where(table.c.id == lookup_id)
                ).first()
                if row is None:
                    return False
                sa_conn.execute(sa_delete(table).where(table.c.id == lookup_id))
        except SQLAlchemyError:
            return False
        return True

    try:
        row = conn.execute(
            f'SELECT id FROM {table_name} WHERE id = ?', (lookup_id,)
        ).fetchone()
    except sqlite3.OperationalError:
        return False
    if row is None:
        return False
    try:
        conn.execute(f'DELETE FROM {table_name} WHERE id = ?', (lookup_id,))
    except sqlite3.OperationalError:
        return False
    return True


def get_related_processed_game_ids(
    conn: sqlite3.Connection,
    relation: Mapping[str, Any],
    lookup_id: int,
) -> list[int]:
    """Return processed game identifiers linked to ``lookup_id``."""

    join_table = relation.get("join_table")
    join_column = relation.get("join_column")
    if not join_table or not join_column:
        return []

    if SQLALCHEMY_AVAILABLE:
        try:
            table = _table_from_connection(conn, join_table)
        except LookupNotFoundError:
            return []

        processed_column = table.c.get("processed_game_id")
        lookup_column = table.c.get(join_column)
        if processed_column is None or lookup_column is None:
            return []

        engine = _engine_from_connection(conn)
        try:
            with engine.connect() as sa_conn:
                result = sa_conn.execute(
                    select(processed_column)
                    .where(lookup_column == lookup_id)
                    .distinct()
                    .order_by(processed_column)
                )
                rows = list(result)
        except SQLAlchemyError:
            return []
    else:
        try:
            cur = conn.execute(
                f'SELECT DISTINCT processed_game_id FROM {join_table} '
                f'WHERE {join_column} = ? ORDER BY processed_game_id',
                (lookup_id,),
            )
        except sqlite3.OperationalError:
            return []
        rows = cur.fetchall()

    ids: list[int] = []
    for row in rows:
        coerced = _coerce_int(row[0])
        if coerced is None:
            continue
        ids.append(coerced)
    return ids


def _coerce_int(value: Any) -> int | None:
    """Best-effort conversion of ``value`` to ``int``."""

    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _replace_relations(
    conn: sqlite3.Connection,
    join_table: str,
    join_column: str,
    processed_game_id: int,
    lookup_ids: Iterable[int],
) -> None:
    """Replace ``processed_game_id`` relations in ``join_table`` with ``lookup_ids``."""

    try:
        conn.execute(
            f"DELETE FROM {join_table} WHERE processed_game_id = ?",
            (processed_game_id,),
        )
    except sqlite3.OperationalError:
        return

    to_insert: list[tuple[int, int]] = []
    for value in lookup_ids:
        coerced = _coerce_int(value)
        if coerced is None:
            continue
        to_insert.append((processed_game_id, coerced))

    if not to_insert:
        return

    try:
        conn.executemany(
            f"INSERT OR IGNORE INTO {join_table} "
            f"(processed_game_id, {join_column}) VALUES (?, ?)",
            to_insert,
        )
    except sqlite3.OperationalError:
        return


def persist_relations(
    conn: sqlite3.Connection,
    processed_game_id: int,
    selections: Mapping[str, Mapping[str, Any]] | Mapping[str, Any],
    relations: Sequence[Mapping[str, Any]],
) -> None:
    """Persist lookup selections for ``processed_game_id``."""

    for relation in relations:
        response_key = relation["response_key"]
        join_table = relation["join_table"]
        join_column = relation["join_column"]

        ids: list[int] = []
        selection: Any = None
        if isinstance(selections, Mapping):
            selection = selections.get(response_key)
        if isinstance(selection, Mapping):
            raw_ids = selection.get("ids")
            if isinstance(raw_ids, (list, tuple)):
                ids = [value for value in raw_ids if value is not None]

        _replace_relations(conn, join_table, join_column, processed_game_id, ids)


def lookup_entries_to_selection(
    entries: Mapping[str, Sequence[Mapping[str, Any]]],
    relations: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, list[int]]]:
    """Convert lookup entries into the selections payload expected by editors."""

    payload: dict[str, dict[str, list[int]]] = {}

    for relation in relations:
        response_key = relation["response_key"]
        relation_entries = entries.get(response_key, []) or []
        ids: list[int] = []
        for entry in relation_entries:
            if not isinstance(entry, Mapping):
                continue
            entry_id = _coerce_int(entry.get("id"))
            if entry_id is None:
                continue
            ids.append(entry_id)
        payload[response_key] = {"ids": ids}

    return payload


def apply_relations_to_game(
    conn: sqlite3.Connection,
    processed_game_id: int,
    entries: Mapping[str, Sequence[Mapping[str, Any]]],
    relations: Sequence[Mapping[str, Any]],
    *,
    normalize_lookup_name: Callable[[Any], str],
    encode_lookup_id_list: Callable[[Iterable[int]], str],
    lookup_display_text: Callable[[list[str]], str],
    columns: set[str] | None = None,
) -> None:
    """Write lookup entry metadata back to ``processed_games`` columns."""

    if columns is None:
        columns = db_utils.get_processed_games_columns(conn)

    set_fragments: list[str] = []
    params: list[Any] = []

    for relation in relations:
        response_key = relation["response_key"]
        processed_column = relation["processed_column"]
        id_column = relation.get("id_column")
        relation_entries = entries.get(response_key, []) or []

        if processed_column in columns:
            names: list[str] = []
            for entry in relation_entries:
                if not isinstance(entry, Mapping):
                    continue
                normalized = normalize_lookup_name(entry.get("name"))
                if normalized:
                    names.append(normalized)
            set_fragments.append(
                f"{db_utils._quote_identifier(processed_column)} = ?"
            )
            params.append(lookup_display_text(names))

        if id_column and id_column in columns:
            ids: list[int] = []
            for entry in relation_entries:
                if not isinstance(entry, Mapping):
                    continue
                coerced = _coerce_int(entry.get("id"))
                if coerced is None:
                    continue
                ids.append(coerced)
            set_fragments.append(f"{db_utils._quote_identifier(id_column)} = ?")
            params.append(encode_lookup_id_list(ids))

    if not set_fragments:
        return

    params.append(processed_game_id)
    conn.execute(
        f"UPDATE processed_games SET {', '.join(set_fragments)} WHERE \"ID\" = ?",
        params,
    )


def remove_lookup_id_from_entries(
    entries: MutableMapping[str, list[Mapping[str, Any]]],
    relation: Mapping[str, Any],
    lookup_id: int,
) -> None:
    """Remove ``lookup_id`` from cached relation entries."""

    response_key = relation["response_key"]
    relation_entries = list(entries.get(response_key, []) or [])
    filtered: list[Mapping[str, Any]] = []

    for entry in relation_entries:
        if not isinstance(entry, Mapping):
            continue
        coerced = _coerce_int(entry.get("id"))
        if coerced == lookup_id:
            continue
        filtered.append(entry)

    entries[response_key] = filtered


def fetch_lookup_entries_for_game(
    conn: sqlite3.Connection,
    processed_game_id: int,
    relations: Sequence[Mapping[str, Any]],
    *,
    normalize_lookup_name: Callable[[Any], str],
    decode_lookup_id_list: Callable[[Any], Iterable[int]],
    parse_iterable: Callable[[Any], Iterable[str]],
    row_value: Callable[[sqlite3.Row | Sequence[Any], str, int], Any],
) -> dict[str, list[dict[str, Any]]]:
    """Return lookup entries associated with ``processed_game_id``."""

    selections: dict[str, list[dict[str, Any]]] = {}

    try:
        cur = conn.execute('SELECT * FROM processed_games WHERE "ID"=?', (processed_game_id,))
        processed_row = cur.fetchone()
    except sqlite3.OperationalError:
        processed_row = None

    processed_mapping = dict(processed_row) if processed_row is not None else {}

    for relation in relations:
        response_key = relation["response_key"]
        lookup_table = relation["lookup_table"]
        join_table = relation["join_table"]
        join_column = relation["join_column"]
        id_column = relation.get("id_column")

        stored_ids: list[int] = []
        if id_column and processed_mapping:
            stored_ids = list(decode_lookup_id_list(processed_mapping.get(id_column)))

        entries: list[dict[str, Any]] = []

        try:
            cur_lookup = conn.execute(
                f"""
                    SELECT j.{join_column} AS lookup_id, l.name
                    FROM {join_table} j
                    LEFT JOIN {lookup_table} l ON l.id = j.{join_column}
                    WHERE j.processed_game_id = ?
                    ORDER BY j.rowid
                """,
                (processed_game_id,),
            )
            join_rows = cur_lookup.fetchall()
        except sqlite3.OperationalError:
            join_rows = []

        join_id_to_name: dict[int, str] = {}
        join_order: list[int] = []

        for join_row in join_rows:
            raw_id = row_value(join_row, "lookup_id", 0)
            lookup_id = _coerce_int(raw_id)
            if lookup_id is None or lookup_id in join_id_to_name:
                continue

            lookup_name = normalize_lookup_name(row_value(join_row, "name", 1))
            if not lookup_name:
                lookup_name = lookup_name_for_id(
                    conn,
                    lookup_table,
                    lookup_id,
                    normalize_lookup_name=normalize_lookup_name,
                )

            join_id_to_name[lookup_id] = normalize_lookup_name(lookup_name)
            join_order.append(lookup_id)

        id_sequence = stored_ids if stored_ids else join_order
        seen_ids: set[int] = set()

        for lookup_id in id_sequence:
            coerced = _coerce_int(lookup_id)
            if coerced is None or coerced in seen_ids:
                continue

            seen_ids.add(coerced)
            lookup_name = join_id_to_name.get(coerced)
            if not lookup_name:
                lookup_name = lookup_name_for_id(
                    conn,
                    lookup_table,
                    coerced,
                    normalize_lookup_name=normalize_lookup_name,
                )

            entries.append({"id": coerced, "name": normalize_lookup_name(lookup_name)})

        if not entries and processed_mapping:
            raw_value = processed_mapping.get(relation["processed_column"])
            for name in parse_iterable(raw_value):
                normalized = normalize_lookup_name(name)
                if normalized:
                    entries.append({"id": None, "name": normalized})

        selections[response_key] = entries

    return selections


def load_lookup_tables(
    conn: sqlite3.Connection,
    tables: Sequence[Mapping[str, Any]],
    *,
    data_dir: Path,
    normalize_lookup_name: Callable[[Any], str],
    log: logging.Logger | None = None,
) -> None:
    """Populate lookup tables using static workbooks when available."""

    logger_ref = log or logger

    for table_config in tables:
        path = data_dir / table_config["filename"]
        if not path.exists():
            continue

        try:
            df = pd.read_excel(path)
        except Exception:  # pragma: no cover - defensive logging path
            logger_ref.exception("Failed to load lookup workbook %s", path)
            continue

        column_name = table_config["column"]
        if column_name not in df.columns:
            logger_ref.warning(
                "Workbook %s missing expected column %s", path, column_name
            )
            continue

        series = df[column_name].dropna()
        for raw_value in series:
            name = normalize_lookup_name(raw_value)
            if not name:
                continue
            conn.execute(
                f"""
                INSERT INTO {table_config['table']} (name)
                VALUES (?)
                ON CONFLICT(name) DO UPDATE SET name=excluded.name
                """,
                (name,),
            )


def _backfill_lookup_id_columns(
    conn: sqlite3.Connection,
    relations: Sequence[Mapping[str, Any]],
    *,
    encode_lookup_id_list: Callable[[Iterable[int]], str],
    decode_lookup_id_list: Callable[[Any], Iterable[int]],
    row_value: Callable[[sqlite3.Row | Sequence[Any], str, int], Any],
) -> None:
    """Synchronize lookup ID columns with join tables."""

    try:
        cur = conn.execute("SELECT * FROM processed_games")
    except sqlite3.OperationalError:
        return

    rows = cur.fetchall()
    column_names = [desc[0] for desc in cur.description] if cur.description else []

    for row in rows:
        if isinstance(row, sqlite3.Row):
            row_dict = dict(row)
        else:
            row_dict = {
                column_names[idx]: value
                for idx, value in enumerate(row)
                if idx < len(column_names)
            }

        try:
            game_id = int(row_dict.get("ID"))
        except (TypeError, ValueError):
            continue

        updates: dict[str, str] = {}

        for relation in relations:
            id_column = relation.get("id_column")
            if not id_column:
                continue

            join_table = relation["join_table"]
            join_column = relation["join_column"]
            stored_value = row_dict.get(id_column)
            existing_serialized = stored_value if isinstance(stored_value, str) else ""

            join_ids: list[int] = []
            try:
                cur_join = conn.execute(
                    f"SELECT {join_column} FROM {join_table} "
                    "WHERE processed_game_id = ? ORDER BY rowid",
                    (game_id,),
                )
                join_rows = cur_join.fetchall()
            except sqlite3.OperationalError:
                join_rows = []

            seen: set[int] = set()
            for join_row in join_rows:
                raw_val = row_value(join_row, join_column, 0)
                coerced = _coerce_int(raw_val)
                if coerced is None or coerced in seen:
                    continue
                seen.add(coerced)
                join_ids.append(coerced)

            serialized = encode_lookup_id_list(join_ids)
            if serialized != (existing_serialized or ""):
                updates[id_column] = serialized

        if updates:
            assignments = ", ".join(
                f"{db_utils._quote_identifier(column)}=?" for column in updates
            )
            params = list(updates.values()) + [game_id]
            conn.execute(
                f'UPDATE processed_games SET {assignments} WHERE "ID"=?',
                params,
            )


def ensure_lookup_id_columns(
    conn: sqlite3.Connection,
    relations: Sequence[Mapping[str, Any]],
    *,
    encode_lookup_id_list: Callable[[Iterable[int]], str],
    decode_lookup_id_list: Callable[[Any], Iterable[int]],
    row_value: Callable[[sqlite3.Row | Sequence[Any], str, int], Any],
) -> None:
    """Ensure processed-game tables include serialized lookup ID columns."""

    try:
        cur = conn.execute("PRAGMA table_info(processed_games)")
    except sqlite3.OperationalError:
        return

    rows = cur.fetchall()
    existing_columns = {row[1] for row in rows}
    added = False

    for relation in relations:
        id_column = relation.get("id_column")
        if not id_column or id_column in existing_columns:
            continue
        try:
            conn.execute(
                f"ALTER TABLE processed_games ADD COLUMN {db_utils._quote_identifier(id_column)} TEXT"
            )
            added = True
        except sqlite3.OperationalError:
            continue

    if added:
        try:
            cur = conn.execute("PRAGMA table_info(processed_games)")
        except sqlite3.OperationalError:
            return
        rows = cur.fetchall()
        existing_columns = {row[1] for row in rows}

    expected_columns = {
        relation["id_column"]
        for relation in relations
        if relation.get("id_column")
    }

    if expected_columns & existing_columns:
        _backfill_lookup_id_columns(
            conn,
            relations,
            encode_lookup_id_list=encode_lookup_id_list,
            decode_lookup_id_list=decode_lookup_id_list,
            row_value=row_value,
        )


def backfill_relations(
    conn: sqlite3.Connection,
    relations: Sequence[Mapping[str, Any]],
    *,
    normalize_lookup_name: Callable[[Any], str],
    parse_iterable: Callable[[Any], Iterable[str]],
    get_or_create_lookup_id: Callable[[sqlite3.Connection, str, str], int | None],
    decode_lookup_id_list: Callable[[Any], Iterable[int]],
    row_value: Callable[[sqlite3.Row | Sequence[Any], str, int], Any],
) -> None:
    """Populate join tables using existing processed-game values."""

    try:
        cur = conn.execute("SELECT * FROM processed_games")
    except sqlite3.OperationalError:
        return

    rows = cur.fetchall()
    column_names = [desc[0] for desc in cur.description] if cur.description else []

    for row in rows:
        if isinstance(row, sqlite3.Row):
            row_dict = dict(row)
        else:
            row_dict = {
                column_names[idx]: value
                for idx, value in enumerate(row)
                if idx < len(column_names)
            }

        try:
            game_id = int(row_dict.get("ID"))
        except (TypeError, ValueError):
            continue

        for relation in relations:
            lookup_table = relation["lookup_table"]
            processed_column = relation["processed_column"]
            join_table = relation["join_table"]
            join_column = relation["join_column"]
            id_column = relation.get("id_column")

            existing_ids: list[int] = []
            try:
                cur_existing = conn.execute(
                    f"SELECT {join_column} FROM {join_table} "
                    "WHERE processed_game_id = ? ORDER BY rowid",
                    (game_id,),
                )
            except sqlite3.OperationalError:
                existing_ids = []
            else:
                for existing_row in cur_existing.fetchall():
                    coerced = _coerce_int(row_value(existing_row, join_column, 0))
                    if coerced is None:
                        continue
                    existing_ids.append(coerced)

            candidate_ids: list[int] = list(existing_ids)

            if not candidate_ids and id_column in row_dict:
                candidate_ids.extend(decode_lookup_id_list(row_dict.get(id_column)))

            if not candidate_ids:
                raw_value = row_dict.get(processed_column)
                names = parse_iterable(raw_value)
                seen_names: set[str] = set()
                for name in names:
                    normalized = normalize_lookup_name(name)
                    if not normalized:
                        continue
                    fingerprint = normalized.casefold()
                    if fingerprint in seen_names:
                        continue
                    seen_names.add(fingerprint)
                    lookup_id = get_or_create_lookup_id(conn, lookup_table, normalized)
                    if lookup_id is None:
                        continue
                    candidate_ids.append(int(lookup_id))

            deduped_ids: list[int] = []
            seen_ids: set[int] = set()
            for value in candidate_ids:
                coerced = _coerce_int(value)
                if coerced is None or coerced in seen_ids:
                    continue
                seen_ids.add(coerced)
                deduped_ids.append(coerced)

            if existing_ids and deduped_ids == existing_ids:
                continue

            _replace_relations(conn, join_table, join_column, game_id, deduped_ids)

