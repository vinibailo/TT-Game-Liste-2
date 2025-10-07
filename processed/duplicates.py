"""Duplicate detection and cleanup utilities for processed games.

This module contains the core logic that powers the duplicate detection
workflows exposed through the updates API.  The code originally lived inside
``app.py`` but has been extracted so it can be reused from other entry points
and keeps the heuristics in a single, testable location.

The helpers are intentionally lightweight – callers supply the database access
functions and other integration specific behaviour.  This keeps the module free
from Flask specific concepts while still mirroring the behaviour that existing
tests rely upon.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field
from datetime import datetime
import numbers
from typing import Any, Callable, Iterable, Mapping, Sequence

from igdb.client import coerce_igdb_id

from db.utils import _quote_identifier


__all__ = [
    "DuplicateGroupResolution",
    "coerce_int",
    "compute_metadata_updates",
    "scan_duplicate_candidates",
    "merge_duplicate_resolutions",
    "remove_processed_games",
]


@dataclass
class DuplicateGroupResolution:
    """Container describing how to reconcile a detected duplicate group."""

    canonical: Mapping[str, Any]
    duplicates: list[Mapping[str, Any]] = field(default_factory=list)
    metadata_updates: dict[str, Any] = field(default_factory=dict)


def _row_get(row: Mapping[str, Any] | Sequence[Any], column: str) -> Any:
    """Best-effort retrieval that works with ``sqlite3.Row`` instances."""

    if isinstance(row, Mapping):
        return row.get(column)
    try:
        return row[column]  # type: ignore[index]
    except (KeyError, IndexError, TypeError):
        return None


def _row_keys(row: Mapping[str, Any] | Sequence[Any]) -> Sequence[str]:
    if isinstance(row, Mapping):
        return tuple(str(key) for key in row.keys())
    try:
        keys_method = row.keys  # type: ignore[attr-defined]
    except AttributeError:
        return ()
    try:
        return tuple(str(key) for key in keys_method())
    except TypeError:
        return ()


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, numbers.Number):
        return str(value)
    return str(value).strip()


def coerce_int(value: Any) -> int | None:
    """Attempt to coerce ``value`` to an integer, returning ``None`` on failure."""

    try:
        if isinstance(value, numbers.Integral):
            return int(value)
        if isinstance(value, numbers.Real):
            float_value = float(value)
            if float_value != float_value:  # NaN check without importing math
                return None
            if float_value.is_integer():
                return int(float_value)
            return None
        text = str(value).strip()
        if not text:
            return None
        return int(text)
    except (TypeError, ValueError):
        return None


def _metadata_value_score(column: str, value: Any) -> float:
    if column in {"Summary", "First Launch Date", "Category"}:
        normalized = _normalize_text(value)
        return float(len(normalized)) if normalized else 0.0
    if column == "Cover Path":
        text = str(value).strip() if value is not None else ""
        return 1.0 if text else 0.0
    if column in {"Width", "Height"}:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return 0.0
        return numeric if numeric > 0 else 0.0
    if column == "last_edited_at":
        if not value:
            return 0.0
        try:
            parsed = datetime.fromisoformat(str(value))
        except ValueError:
            return 0.0
        return parsed.timestamp()
    return 0.0


def compute_metadata_updates(
    canonical: Mapping[str, Any] | Sequence[Any],
    duplicates: Iterable[Mapping[str, Any] | Sequence[Any]],
    *,
    metadata_columns: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Determine the best metadata values across ``duplicates``.

    The heuristic mirrors the existing behaviour: for each interesting metadata
    column we choose the value with the highest score (longest text, existing
    cover, latest edit timestamp, …) and update the canonical record with that
    value.
    """

    duplicate_list = list(duplicates)
    if not duplicate_list:
        return {}

    if metadata_columns is None:
        metadata_columns = (
            "Summary",
            "Cover Path",
            "First Launch Date",
            "Category",
            "Width",
            "Height",
            "last_edited_at",
        )

    updates: dict[str, Any] = {}
    for column in metadata_columns:
        canonical_value = _row_get(canonical, column)
        canonical_score = _metadata_value_score(column, canonical_value)
        best_value = canonical_value
        best_score = canonical_score
        for entry in duplicate_list:
            candidate_value = _row_get(entry, column)
            candidate_score = _metadata_value_score(column, candidate_value)
            if candidate_score > best_score:
                best_score = candidate_score
                best_value = candidate_value
        if best_score > canonical_score:
            updates[column] = best_value
    return updates


def _relation_count_from_row(
    row: Mapping[str, Any] | Sequence[Any],
    relation_count_columns: Sequence[str] | None,
) -> int:
    total = 0
    columns: Sequence[str]
    if relation_count_columns is None:
        keys = _row_keys(row)
        columns = tuple(key for key in keys if key.endswith("_count"))
    else:
        columns = relation_count_columns
    for column in columns:
        value = _row_get(row, column)
        count = coerce_int(value)
        if count:
            total += count
    return total


def _choose_canonical_duplicate(
    group: Sequence[Mapping[str, Any] | Sequence[Any]],
    relation_count_columns: Sequence[str] | None,
) -> Mapping[str, Any] | Sequence[Any] | None:
    best_row: Mapping[str, Any] | Sequence[Any] | None = None
    best_score: tuple[Any, ...] | None = None
    for entry in group:
        entry_id = coerce_int(_row_get(entry, "ID"))
        if entry_id is None:
            continue
        relation_score = _relation_count_from_row(entry, relation_count_columns)
        metadata_presence = sum(
            1
            for column in ("Summary", "Cover Path", "First Launch Date", "Category")
            if _metadata_value_score(column, _row_get(entry, column)) > 0
        )
        cover_score = _metadata_value_score("Cover Path", _row_get(entry, "Cover Path"))
        summary_score = _metadata_value_score("Summary", _row_get(entry, "Summary"))
        edited_score = _metadata_value_score("last_edited_at", _row_get(entry, "last_edited_at"))
        score = (
            relation_score,
            metadata_presence,
            cover_score,
            summary_score,
            edited_score,
            -entry_id,
        )
        if best_score is None or score > best_score:
            best_score = score
            best_row = entry
    return best_row


def scan_duplicate_candidates(
    rows: Iterable[Mapping[str, Any] | Sequence[Any]],
    *,
    progress_callback: Callable[[int, int, int, int], None] | None = None,
    relation_count_columns: Sequence[str] | None = None,
    metadata_columns: Sequence[str] | None = None,
) -> tuple[list[DuplicateGroupResolution], int, int, int]:
    """Group processed games that look like potential duplicates.

    The heuristic groups entries that share the same normalised name and IGDB
    identifier.  Each group is evaluated and, when appropriate, a
    :class:`DuplicateGroupResolution` is produced describing the canonical
    record, the rows that should be merged into it and any metadata overrides.
    """

    groups: dict[tuple[str, str], list[Mapping[str, Any] | Sequence[Any]]] = {}
    for row in rows:
        name_value = _normalize_text(_row_get(row, "Name"))
        igdb_value = coerce_igdb_id(_row_get(row, "igdb_id"))
        if not name_value or not igdb_value:
            continue
        key = (name_value.casefold(), igdb_value)
        groups.setdefault(key, []).append(row)

    group_values = list(groups.values())
    total_groups = len(group_values)

    duplicate_groups = 0
    skipped_groups = 0
    resolutions: list[DuplicateGroupResolution] = []

    for index, group in enumerate(group_values, start=1):
        if len(group) <= 1:
            continue
        duplicate_groups += 1
        canonical_row = _choose_canonical_duplicate(group, relation_count_columns)
        if canonical_row is None:
            skipped_groups += 1
            continue
        canonical_id = coerce_int(_row_get(canonical_row, "ID"))
        if canonical_id is None:
            skipped_groups += 1
            continue
        duplicate_rows: list[Mapping[str, Any] | Sequence[Any]] = []
        for entry in group:
            if entry is canonical_row:
                continue
            entry_id = coerce_int(_row_get(entry, "ID"))
            if entry_id is None:
                continue
            duplicate_rows.append(entry)
        if not duplicate_rows:
            skipped_groups += 1
            continue
        metadata_updates = compute_metadata_updates(
            canonical_row,
            duplicate_rows,
            metadata_columns=metadata_columns,
        )
        resolutions.append(
            DuplicateGroupResolution(
                canonical=canonical_row,
                duplicates=duplicate_rows,
                metadata_updates=metadata_updates,
            )
        )
        if progress_callback is not None:
            progress_callback(index, total_groups or len(group_values) or 1, duplicate_groups, skipped_groups)

    return resolutions, duplicate_groups, skipped_groups, total_groups


def _determine_param_placeholder(conn: Any) -> str:
    """Return the appropriate placeholder token for parameterized queries."""

    paramstyle: str | None = None

    engine = getattr(conn, "engine", None)
    if engine is not None:
        dialect = getattr(engine, "dialect", None)
        if dialect is not None:
            paramstyle = getattr(dialect, "paramstyle", None)

    if paramstyle is None:
        dialect = getattr(conn, "dialect", None)
        if dialect is not None:
            paramstyle = getattr(dialect, "paramstyle", None)

    if paramstyle is None:
        module = getattr(getattr(conn, "__class__", None), "__module__", "")
        if module.startswith("pymysql"):
            paramstyle = "format"

    if paramstyle in {"format", "pyformat"}:
        return "%s"
    return "?"


def merge_duplicate_resolutions(
    resolutions: Iterable[DuplicateGroupResolution],
    *,
    db_lock: Any,
    get_db: Callable[[], Any],
    lookup_relations: Iterable[Mapping[str, Any]],
    apply_metadata_updates: Callable[[Any, int, Mapping[str, Any]], None] | None = None,
    fetch_lookup_entries_for_game: Callable[[Any, int], Any] | None = None,
    apply_lookup_entries_to_processed_game: Callable[[Any, int, Any], None] | None = None,
) -> set[int]:
    """Merge duplicate records into their canonical counterpart.

    The function moves lookup relations, applies metadata updates and returns
    the set of duplicate IDs that should be removed from ``processed_games``.
    """

    resolution_list = [resolution for resolution in resolutions if resolution.duplicates]
    if not resolution_list:
        return set()

    ids_to_delete: set[int] = set()
    canonical_ids: set[int] = set()

    with db_lock:
        conn = get_db()
        with conn:
            placeholder = _determine_param_placeholder(conn)
            processed_column = _quote_identifier("processed_game_id")
            relation_statements: dict[tuple[str, str], tuple[str, str]] = {}
            for resolution in resolution_list:
                canonical_id = coerce_int(_row_get(resolution.canonical, "ID"))
                if canonical_id is None:
                    continue
                duplicate_ids: list[int] = []
                for duplicate_row in resolution.duplicates:
                    duplicate_id = coerce_int(_row_get(duplicate_row, "ID"))
                    if duplicate_id is None:
                        continue
                    duplicate_ids.append(duplicate_id)
                    for relation in lookup_relations:
                        join_table = relation["join_table"]
                        join_column = relation["join_column"]
                        key = (join_table, join_column)
                        if key not in relation_statements:
                            join_table_sql = _quote_identifier(join_table)
                            join_column_sql = _quote_identifier(join_column)
                            insert_sql = (
                                f"INSERT IGNORE INTO {join_table_sql} "
                                f"({processed_column}, {join_column_sql}) "
                                f"SELECT {placeholder}, {join_column_sql} "
                                f"FROM {join_table_sql} "
                                f"WHERE {processed_column} = {placeholder}"
                            )
                            delete_sql = (
                                f"DELETE FROM {join_table_sql} "
                                f"WHERE {processed_column} = {placeholder}"
                            )
                            relation_statements[key] = (insert_sql, delete_sql)
                        insert_sql, delete_sql = relation_statements[key]
                        conn.execute(insert_sql, (canonical_id, duplicate_id))
                        conn.execute(delete_sql, (duplicate_id,))
                if not duplicate_ids:
                    continue
                ids_to_delete.update(duplicate_ids)
                canonical_ids.add(canonical_id)
                if resolution.metadata_updates and apply_metadata_updates is not None:
                    apply_metadata_updates(conn, canonical_id, resolution.metadata_updates)
            if (
                canonical_ids
                and fetch_lookup_entries_for_game is not None
                and apply_lookup_entries_to_processed_game is not None
            ):
                unique_ids = sorted(canonical_ids)
                for game_id in unique_ids:
                    entries = fetch_lookup_entries_for_game(conn, game_id)
                    if not entries:
                        continue
                    apply_lookup_entries_to_processed_game(conn, game_id, entries)

    return ids_to_delete


def remove_processed_games(
    ids_to_delete: Iterable[int],
    *,
    catalog_state: Any,
    db_lock: Any,
    get_db: Callable[[], Any],
    navigator_canonical: Callable[[Any], Any],
    get_position_for_source_index: Callable[[Any], int | None],
    normalize_processed_games: Callable[[], None],
) -> tuple[int, int]:
    """Delete processed games and keep catalogue state in sync."""

    games_df = catalog_state.games_df

    unique_ids = sorted({int(game_id) for game_id in ids_to_delete if str(game_id).strip()})
    if not unique_ids:
        return 0, len(games_df)

    with db_lock:
        conn = get_db()
        placeholder = _determine_param_placeholder(conn)
        placeholders = ", ".join(placeholder for _ in unique_ids)
        id_column = _quote_identifier("ID")
        source_index_column = _quote_identifier("Source Index")
        processed_games_table = _quote_identifier("processed_games")
        cur = conn.execute(
            (
                f"SELECT {id_column}, {source_index_column} "
                f"FROM {processed_games_table} "
                f"WHERE {id_column} IN ({placeholders})"
            ),
            tuple(unique_ids),
        )
        rows = cur.fetchall()

    if not rows:
        return 0, len(games_df)

    delete_params = [(row["ID"],) for row in rows]
    raw_source_indices = [row["Source Index"] for row in rows if row["Source Index"] is not None]

    with db_lock:
        conn = get_db()
        with conn:
            processed_games_table = _quote_identifier("processed_games")
            updates_table = _quote_identifier("igdb_updates")
            id_column = _quote_identifier("ID")
            source_index_column = _quote_identifier("Source Index")
            processed_game_id_column = _quote_identifier("processed_game_id")
            placeholder = _determine_param_placeholder(conn)
            delete_updates_sql = (
                f"DELETE FROM {updates_table} "
                f"WHERE {processed_game_id_column} = {placeholder}"
            )
            delete_processed_sql = (
                f"DELETE FROM {processed_games_table} "
                f"WHERE {id_column} = {placeholder}"
            )
            conn.executemany(
                delete_updates_sql,
                delete_params,
            )
            conn.executemany(
                delete_processed_sql,
                delete_params,
            )

    canonical_indices: set[str] = set()
    for value in raw_source_indices:
        canonical = navigator_canonical(value)
        if canonical is not None:
            canonical_indices.add(canonical)

    removed_numeric = sorted({int(candidate) for candidate in canonical_indices if str(candidate).isdigit()})

    positions_to_remove: set[int] = set()
    for value in raw_source_indices:
        position = get_position_for_source_index(value)
        if position is None:
            canonical = navigator_canonical(value)
            if canonical is not None and canonical != value:
                position = get_position_for_source_index(canonical)
        if position is not None:
            positions_to_remove.add(position)

    updated_df = games_df
    if not updated_df.empty and positions_to_remove:
        drop_indices = sorted(positions_to_remove)
        updated_df = updated_df.drop(updated_df.index[drop_indices]).reset_index(drop=True)

    if not updated_df.empty:
        if "Source Index" in updated_df.columns:
            current_values = updated_df["Source Index"].tolist()
        else:
            current_values = [str(idx) for idx in range(len(updated_df))]
        new_values: list[str] = []
        for idx, value in enumerate(current_values):
            canonical = navigator_canonical(value)
            if canonical is None:
                new_values.append(str(idx))
                continue
            canonical_text = str(canonical)
            if canonical_text.isdigit():
                numeric_value = int(canonical_text)
                shift = bisect_left(removed_numeric, numeric_value)
                new_numeric = numeric_value - shift
                stripped = str(value).strip()
                if stripped.isdigit():
                    formatted = str(new_numeric).zfill(len(stripped))
                else:
                    formatted = str(new_numeric)
                new_values.append(formatted)
            else:
                new_values.append(canonical_text)
        updated_df = updated_df.copy()
        updated_df["Source Index"] = new_values

    if removed_numeric:
        with db_lock:
            conn = get_db()
            with conn:
                processed_games_table = _quote_identifier("processed_games")
                id_column = _quote_identifier("ID")
                source_index_column = _quote_identifier("Source Index")
                placeholder = _determine_param_placeholder(conn)
                select_sql = (
                    f"SELECT {id_column}, {source_index_column} "
                    f"FROM {processed_games_table}"
                )
                update_sql = (
                    f"UPDATE {processed_games_table} "
                    f"SET {source_index_column} = {placeholder} "
                    f"WHERE {id_column} = {placeholder}"
                )
                cur = conn.execute(select_sql)
                stored_rows = cur.fetchall()
                for entry in stored_rows:
                    canonical = navigator_canonical(entry["Source Index"])
                    if canonical is None:
                        continue
                    canonical_text = str(canonical)
                    if not canonical_text.isdigit():
                        continue
                    numeric_value = int(canonical_text)
                    shift = bisect_left(removed_numeric, numeric_value)
                    if shift <= 0:
                        continue
                    new_numeric = numeric_value - shift
                    stored_text = str(entry["Source Index"])
                    stripped = stored_text.strip()
                    if stripped.isdigit():
                        new_value = str(new_numeric).zfill(len(stripped))
                    else:
                        new_value = str(new_numeric)
                    conn.execute(update_sql, (new_value, entry["ID"]))

    normalize_processed_games()

    try:
        catalog_state.set_games_dataframe(
            updated_df,
            rebuild_metadata=False,
            rebuild_navigator=True,
        )
    except Exception:
        pass

    remaining_total = catalog_state.total_games
    removed_count = len(delete_params)
    return removed_count, remaining_total
