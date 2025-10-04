"""Services for orchestrating update-related background jobs."""

from __future__ import annotations

import json
import numbers
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from functools import partial
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence

from config import IGDB_USER_AGENT
from helpers import _format_first_release_date, _normalize_lookup_name
from igdb.cache import (
    IGDB_CACHE_TABLE,
    get_cached_total as cache_get_cached_total,
    set_cached_total as cache_set_cached_total,
    upsert_igdb_games,
)
from igdb.client import (
    IGDBClient,
    download_igdb_game_count as client_download_igdb_game_count,
    download_igdb_games as client_download_igdb_games,
    map_igdb_genres,
    map_igdb_modes,
)
from lookups.service import get_or_create_lookup_id


ProgressCallback = Callable[..., None]


def _quote_identifier(identifier: str) -> str:
    text = str(identifier or "")
    return f'"{text.replace("\"", "\"\"")}"'


def _fetch_row_dict(
    conn: sqlite3.Connection, sql: str, params: Iterable[Any] | None = None
) -> dict[str, Any] | None:
    cursor = conn.execute(sql, tuple(params or ()))
    row = cursor.fetchone()
    if row is None:
        return None
    columns = [description[0] for description in cursor.description]
    return {
        columns[index] if index < len(columns) else str(index): value
        for index, value in enumerate(row)
    }


def _parse_cache_list(raw_value: Any) -> list[str]:
    if raw_value in (None, ""):
        return []
    if isinstance(raw_value, str):
        text = raw_value.strip()
        if not text:
            return []
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            return [
                item
                for item in (segment.strip() for segment in text.split(","))
                if item
            ]
    elif isinstance(raw_value, (list, tuple)):
        value = list(raw_value)
    else:
        value = [raw_value]
    result: list[str] = []
    for item in value or []:
        text = str(item).strip()
        if text:
            result.append(text)
    return result


def _normalize_name_list(names: Iterable[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for name in names:
        text = _normalize_lookup_name(name)
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(text)
    return normalized


def _encode_lookup_id_list(values: Iterable[int]) -> str:
    normalized: list[int] = []
    seen: set[int] = set()
    for value in values:
        try:
            coerced = int(value)
        except (TypeError, ValueError):
            continue
        if coerced in seen:
            continue
        seen.add(coerced)
        normalized.append(coerced)
    if not normalized:
        return ""
    return json.dumps(normalized)


def _resolve_lookup_ids(
    conn: sqlite3.Connection, table_name: str, names: Iterable[str]
) -> list[int]:
    identifiers: list[int] = []
    seen: set[int] = set()
    for name in _normalize_name_list(names):
        lookup_id = get_or_create_lookup_id(
            conn, table_name, name, normalize_lookup_name=_normalize_lookup_name
        )
        if lookup_id is None or lookup_id in seen:
            continue
        seen.add(lookup_id)
        identifiers.append(lookup_id)
    return identifiers


def _coerce_igdb_id(value: Any) -> int | None:
    if value in (None, ""):
        return None
    if isinstance(value, numbers.Integral):
        return int(value)
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def apply_processed_game_patch(
    processed_game_id: int,
    field_actions: Mapping[str, Any],
    *,
    db_lock: Any,
    get_db: Callable[[], sqlite3.Connection],
    now_utc_iso: Callable[[], str],
) -> dict[str, Any]:
    """Apply cache-backed patches to a processed game entry."""

    if processed_game_id is None:
        raise ValueError("Processed game identifier is required.")
    if not isinstance(field_actions, Mapping) or not field_actions:
        raise ValueError("Field selections are required.")

    normalized_actions: dict[str, str] = {}
    for key, raw_action in field_actions.items():
        if not key:
            continue
        action = str(raw_action or "").strip().lower()
        if not action:
            continue
        normalized_actions[str(key)] = action

    if not normalized_actions:
        raise ValueError("No valid field selections provided.")

    lookup_specs = {
        "Developers": {"cache": "developers", "ids": "developers_ids", "table": "developers"},
        "Publishers": {"cache": "publishers", "ids": "publishers_ids", "table": "publishers"},
        "Platforms": {"cache": "platforms", "ids": "platforms_ids", "table": "platforms"},
        "Genres": {"cache": "genres", "ids": "genres_ids", "table": "genres"},
        "Game Modes": {"cache": "game_modes", "ids": "game_modes_ids", "table": "game_modes"},
    }

    with db_lock:
        conn = get_db()
        processed_row = _fetch_row_dict(
            conn, 'SELECT * FROM processed_games WHERE "ID"=?', (processed_game_id,)
        )
        if not processed_row:
            raise LookupError("Processed game not found.")

        cache_row: dict[str, Any] | None = None
        cache_required = any(
            action.startswith("from_cache") for action in normalized_actions.values()
        )
        if cache_required:
            igdb_id = _coerce_igdb_id(processed_row.get("igdb_id"))
            if igdb_id is None:
                raise ValueError("Processed game is missing an IGDB identifier.")
            cache_row = _fetch_row_dict(
                conn,
                f"SELECT * FROM {_quote_identifier(IGDB_CACHE_TABLE)} WHERE igdb_id=?",
                (igdb_id,),
            )
            if cache_row is None:
                raise ValueError("IGDB cache entry not found for processed game.")

        pragma_cursor = conn.execute('PRAGMA table_info("processed_games")')
        processed_columns = {row[1] for row in pragma_cursor.fetchall() if len(row) > 1}

        updates: dict[str, Any] = {}
        lookup_names: dict[str, list[str]] = {}

        for field_name, action in normalized_actions.items():
            if action == "keep_current":
                continue
            if not action.startswith("from_cache"):
                raise ValueError(f"Unsupported action '{action}' for field '{field_name}'.")
            if cache_row is None:
                raise ValueError("IGDB cache entry required to fulfil selection.")

            cache_key = field_name
            value: Any = None

            if field_name == "Name":
                cache_key = "name"
                value = cache_row.get(cache_key)
                value = value.strip() if isinstance(value, str) else value
            elif field_name == "Summary":
                cache_key = "summary"
                value = cache_row.get(cache_key)
                value = value.strip() if isinstance(value, str) else value
            elif field_name == "First Launch Date":
                cache_key = "first_release_date"
                value = _format_first_release_date(cache_row.get(cache_key))
            elif field_name == "Category":
                cache_key = "category"
                value = IGDBClient.translate_category(cache_row.get(cache_key))
            elif field_name in lookup_specs:
                spec = lookup_specs[field_name]
                cache_key = spec["cache"]
                raw_names = _parse_cache_list(cache_row.get(cache_key))
                if field_name == "Genres" and action == "from_cache_mapped":
                    mapped = map_igdb_genres(raw_names)
                elif field_name == "Game Modes" and action == "from_cache_mapped":
                    mapped = map_igdb_modes(raw_names)
                else:
                    mapped = raw_names
                normalized_names = _normalize_name_list(mapped)
                lookup_names[field_name] = normalized_names
                value = ", ".join(normalized_names)
            else:
                raise ValueError(f"Unsupported field '{field_name}'.")

            column_exists = field_name in processed_columns
            if not column_exists and field_name in lookup_specs:
                column_exists = lookup_specs[field_name]["ids"] in processed_columns
            if not column_exists:
                continue

            if isinstance(value, str):
                updates[field_name] = value
            elif value is None:
                updates[field_name] = ""
            else:
                updates[field_name] = str(value)

        if not updates:
            raise ValueError("No cache fields could be applied to the processed game.")

        for field_name, names in lookup_names.items():
            spec = lookup_specs[field_name]
            ids_column = spec.get("ids")
            table_name = spec.get("table")
            if not ids_column or ids_column not in processed_columns:
                continue
            if not table_name:
                continue
            ids = _resolve_lookup_ids(conn, table_name, names)
            updates[ids_column] = _encode_lookup_id_list(ids)

        timestamp: str | None = None
        if "last_edited_at" in processed_columns:
            timestamp = now_utc_iso()
            updates["last_edited_at"] = timestamp

        set_fragments: list[str] = []
        params: list[Any] = []
        for column, value in updates.items():
            if column not in processed_columns:
                continue
            set_fragments.append(f"{_quote_identifier(column)} = ?")
            params.append(value)

        if not set_fragments:
            raise ValueError("No valid processed columns were updated.")

        params.append(processed_game_id)
        conn.execute(
            f"UPDATE processed_games SET {', '.join(set_fragments)} WHERE \"ID\" = ?",
            params,
        )
        conn.commit()

    updated_fields = [column for column in updates if column in processed_columns]
    return {
        "processed_game_id": processed_game_id,
        "updated_fields": sorted(updated_fields),
        "last_edited_at": updates.get("last_edited_at") if "last_edited_at" in updates else None,
    }


def _resolve_route_helper(name: str, fallback: Any) -> Any:
    """Return a helper override from the update routes context if available."""

    try:  # Lazy import to avoid circular imports at module load time.
        from routes import updates as routes_updates  # type: ignore
    except Exception:
        return fallback

    context = getattr(routes_updates, "_context", None)
    if not isinstance(context, Mapping):
        return fallback

    helper = context.get(name)
    candidate = helper
    if callable(helper):
        try:
            candidate = helper()
        except TypeError:
            candidate = helper
        except Exception:
            return fallback

    if callable(candidate):
        return candidate

    return fallback


def fetch_batches_concurrently(
    access_token: str,
    client_id: str,
    offsets: Iterable[int],
    limit: int,
    *,
    max_workers: int = 4,
) -> list[tuple[int, Sequence[Mapping[str, Any]]]]:
    """Fetch multiple IGDB batches concurrently while preserving order."""

    try:
        worker_count = int(max_workers)
    except (TypeError, ValueError):
        worker_count = 1
    if worker_count <= 0:
        worker_count = 1

    offset_list = list(offsets)
    if not offset_list:
        return []

    futures: dict[Any, tuple[int, int]] = {}
    results: list[tuple[int, int, Sequence[Mapping[str, Any]]]] = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        for index, offset in enumerate(offset_list):
            futures[
                executor.submit(
                    client_download_igdb_games, access_token, client_id, offset, limit
                )
            ] = (index, offset)
        for future in as_completed(futures):
            index, offset = futures[future]
            games = future.result()
            results.append((index, offset, games))

    results.sort(key=lambda item: item[0])
    return [(offset, games) for _, offset, games in results]

def _looks_like_auth_error(error: BaseException) -> bool:
    message = str(error).strip().casefold()
    if not message:
        return False
    for token in ("401", "403", "unauthorized", "invalid", "forbidden"):
        if token in message:
            return True
    return False


def refresh_igdb_cache(
    access_token: str,
    client_id: str,
    offset: Any,
    limit: Any,
    *,
    db_lock: Any | None = None,
    get_db: Callable[[], Any] | None = None,
    conn: Any | None = None,
    get_cached_total: Callable[[Any], int | None] = cache_get_cached_total,
    set_cached_total: Callable[[Any, int | None], None] = cache_set_cached_total,
    download_total: Callable[[str, str], int] | None = None,
    download_games: Callable[[str, str, int, int], Sequence[Mapping[str, Any]]] | None = None,
    upsert_games: Callable[[Any, Iterable[Mapping[str, Any]]], tuple[int, int, int]] = upsert_igdb_games,
    exchange_credentials: Callable[[], tuple[str, str]] | None = None,
    igdb_prefill_cache: MutableMapping[str, Any] | None = None,
    igdb_prefill_lock: Any | None = None,
) -> dict[str, Any]:
    """Synchronise the local IGDB cache with the remote catalogue."""

    try:
        offset_value = int(offset)
    except (TypeError, ValueError):
        offset_value = 0
    if offset_value < 0:
        offset_value = 0

    try:
        limit_value = int(limit)
    except (TypeError, ValueError):
        limit_value = 0
    if limit_value <= 0:
        limit_value = 500
    limit_value = max(1, min(limit_value, 500))

    download_total_fn = download_total or partial(
        client_download_igdb_game_count, user_agent=IGDB_USER_AGENT
    )
    download_total_fn = _resolve_route_helper('download_igdb_game_count', download_total_fn)
    download_games_fn = download_games or partial(
        client_download_igdb_games, user_agent=IGDB_USER_AGENT
    )
    download_games_fn = _resolve_route_helper('download_igdb_games', download_games_fn)

    token_value = (access_token or "").strip()
    client_value = (client_id or "").strip()

    def _ensure_credentials(force_refresh: bool = False) -> tuple[str, str]:
        nonlocal token_value, client_value
        if force_refresh or not token_value or not client_value:
            if exchange_credentials is None:
                raise RuntimeError("Invalid IGDB credentials")
            try:
                new_token, new_client = exchange_credentials(force_refresh=force_refresh)
            except TypeError:
                new_token, new_client = exchange_credentials()
            token_value = (new_token or "").strip()
            client_value = (new_client or "").strip()
        if not token_value:
            raise RuntimeError(
                "Missing IGDB access token; call exchange_twitch_credentials first"
            )
        if not client_value:
            raise RuntimeError("Missing IGDB client identifier; set IGDB_CLIENT_ID")
        return token_value, client_value

    def _download_total_with_retry() -> int:
        nonlocal token_value, client_value
        token_value, client_value = _ensure_credentials()
        try:
            return download_total_fn(token_value, client_value)
        except RuntimeError as exc:
            if exchange_credentials and _looks_like_auth_error(exc):
                token_value, client_value = _ensure_credentials(force_refresh=True)
                return download_total_fn(token_value, client_value)
            raise

    def _download_games_with_retry(offset_param: int, limit_param: int):
        nonlocal token_value, client_value
        token_value, client_value = _ensure_credentials()
        try:
            return download_games_fn(token_value, client_value, offset_param, limit_param)
        except RuntimeError as exc:
            if exchange_credentials and _looks_like_auth_error(exc):
                token_value, client_value = _ensure_credentials(force_refresh=True)
                return download_games_fn(token_value, client_value, offset_param, limit_param)
            raise

    if conn is not None:
        get_conn = lambda: conn  # noqa: E731

        if db_lock is not None:
            def acquire_lock() -> Any:
                return db_lock
        else:
            def acquire_lock() -> Any:
                return nullcontext()
    else:
        if get_db is None or db_lock is None:
            raise RuntimeError('refresh_igdb_cache requires database helpers')

        get_conn = get_db

        def acquire_lock() -> Any:
            return db_lock

    with acquire_lock():
        conn_obj = get_conn()
        cached_total = get_cached_total(conn_obj) if get_cached_total else None

    should_refresh_total = cached_total is None or offset_value == 0
    total = cached_total
    if should_refresh_total:
        try:
            total = _download_total_with_retry()
        except RuntimeError as exc:
            return {
                'status': 'error',
                'error': str(exc),
                'total': cached_total or 0,
                'processed': max(offset_value, 0),
                'inserted': 0,
                'updated': 0,
                'unchanged': 0,
                'done': True,
                'next_offset': max(offset_value, 0),
                'batch_count': 0,
            }
        if set_cached_total:
            with acquire_lock():
                conn_obj = get_conn()
                with conn_obj:
                    set_cached_total(conn_obj, total)

    if total is None or total <= 0:
        if set_cached_total:
            with acquire_lock():
                conn_obj = get_conn()
                with conn_obj:
                    set_cached_total(conn_obj, total)
        return {
            'status': 'ok',
            'total': total or 0,
            'processed': 0,
            'inserted': 0,
            'updated': 0,
            'unchanged': 0,
            'done': True,
            'next_offset': 0,
            'batch_count': 0,
        }

    if offset_value >= total:
        return {
            'status': 'ok',
            'total': total,
            'processed': total,
            'inserted': 0,
            'updated': 0,
            'unchanged': 0,
            'done': True,
            'next_offset': total,
            'batch_count': 0,
        }

    window_offsets: list[int] = [offset_value]
    if offset_value > 0:
        current_offset = offset_value + limit_value
        max_window = 4
        while len(window_offsets) < max_window and current_offset < total:
            window_offsets.append(current_offset)
            current_offset += limit_value

    if not window_offsets:
        return {
            'status': 'ok',
            'total': total,
            'processed': offset_value,
            'inserted': 0,
            'updated': 0,
            'unchanged': 0,
            'done': True,
            'next_offset': offset_value,
            'batch_count': 0,
        }

    def _download_window() -> list[tuple[int, Sequence[Mapping[str, Any]]]]:
        nonlocal token_value, client_value

        def _perform() -> list[tuple[int, Sequence[Mapping[str, Any]]]]:
            global client_download_igdb_games

            original_downloader = client_download_igdb_games
            override_downloader = download_games_fn or original_downloader
            try:
                if override_downloader is not original_downloader:
                    client_download_igdb_games = override_downloader
                return fetch_batches_concurrently(
                    token_value,
                    client_value,
                    window_offsets,
                    limit_value,
                    max_workers=4,
                )
            finally:
                client_download_igdb_games = original_downloader

        token_value, client_value = _ensure_credentials()
        try:
            return _perform()
        except RuntimeError as exc:
            if exchange_credentials and _looks_like_auth_error(exc):
                token_value, client_value = _ensure_credentials(force_refresh=True)
                return _perform()
            raise

    try:
        batch_payloads = _download_window()
    except RuntimeError as exc:
        return {
            'status': 'error',
            'error': str(exc),
            'total': total or 0,
            'processed': max(offset_value, 0),
            'inserted': 0,
            'updated': 0,
            'unchanged': 0,
            'done': True,
            'next_offset': max(offset_value, 0),
            'batch_count': 0,
        }

    batch_payloads.sort(key=lambda item: item[0])
    all_games: list[Mapping[str, Any]] = []
    total_records = 0
    highest_end = offset_value
    for batch_offset, games in batch_payloads:
        batch_list = list(games or [])
        if batch_list:
            all_games.extend(batch_list)
        count = len(batch_list)
        total_records += count
        candidate_end = batch_offset + count
        if candidate_end > highest_end:
            highest_end = candidate_end

    inserted = updated = unchanged = 0
    if all_games:
        with acquire_lock():
            conn_obj = get_conn()
            with conn_obj:
                inserted, updated, unchanged = upsert_games(conn_obj, all_games)
        if igdb_prefill_lock is not None and igdb_prefill_cache is not None:
            with igdb_prefill_lock:
                igdb_prefill_cache.clear()

    processed = highest_end if total_records else offset_value
    if processed < 0:
        processed = 0
    if total is not None:
        processed = min(processed, total)

    done = processed >= total or total_records == 0
    next_offset = highest_end if total_records else processed

    return {
        'status': 'ok',
        'total': total,
        'processed': processed,
        'inserted': inserted,
        'updated': updated,
        'unchanged': unchanged,
        'done': done,
        'next_offset': next_offset,
        'batch_count': total_records,
    }




