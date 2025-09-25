"""Lookup-table persistence and service-layer helpers."""

from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence

import sqlite3

from db import utils as db_utils


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

