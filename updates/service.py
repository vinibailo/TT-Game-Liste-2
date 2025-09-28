"""Services for orchestrating update-related background jobs."""

from __future__ import annotations

from functools import partial
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence

from config import IGDB_USER_AGENT
from igdb.cache import (
    get_cached_total as cache_get_cached_total,
    set_cached_total as cache_set_cached_total,
    upsert_igdb_games,
)
from igdb.client import (
    download_igdb_game_count as client_download_igdb_game_count,
    download_igdb_games as client_download_igdb_games,
)


ProgressCallback = Callable[..., None]

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
    db_lock: Any,
    get_db: Callable[[], Any],
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
    download_games_fn = download_games or partial(
        client_download_igdb_games, user_agent=IGDB_USER_AGENT
    )

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

    with db_lock:
        conn = get_db()
        cached_total = get_cached_total(conn) if get_cached_total else None

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
            with db_lock:
                conn = get_db()
                with conn:
                    set_cached_total(conn, total)

    if total is None or total <= 0:
        if set_cached_total:
            with db_lock:
                conn = get_db()
                with conn:
                    set_cached_total(conn, total)
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

    try:
        payloads = _download_games_with_retry(offset_value, limit_value)
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
    batch_count = len(payloads)

    inserted = updated = unchanged = 0
    if batch_count:
        with db_lock:
            conn = get_db()
            with conn:
                inserted, updated, unchanged = upsert_games(conn, payloads)
        if igdb_prefill_lock is not None and igdb_prefill_cache is not None:
            with igdb_prefill_lock:
                igdb_prefill_cache.clear()

    processed = offset_value + batch_count
    if processed < 0:
        processed = 0
    if total is not None:
        processed = min(processed, total)

    done = processed >= total or batch_count == 0
    next_offset = offset_value + batch_count if batch_count else processed

    return {
        'status': 'ok',
        'total': total,
        'processed': processed,
        'inserted': inserted,
        'updated': updated,
        'unchanged': unchanged,
        'done': done,
        'next_offset': next_offset,
        'batch_count': batch_count,
    }


def fix_names_job(
    update_progress: ProgressCallback,
    *,
    db_lock: Any,
    get_db: Callable[[], Any],
    fetch_igdb_metadata: Callable[[Sequence[str]], Mapping[str, Mapping[str, Any]] | None],
    coerce_igdb_id: Callable[[Any], str | None],
    normalize_text: Callable[[Any], str],
    now_utc_iso: Callable[[], str],
    default_limit: int,
    offset: int = 0,
    limit: int | None = None,
    process_all: bool = True,
) -> dict[str, Any]:
    """Synchronise processed game names with IGDB metadata."""

    try:
        start_offset = int(offset)
    except (TypeError, ValueError):
        start_offset = 0
    if start_offset < 0:
        start_offset = 0

    limit_value: int | None = None
    if limit is not None:
        try:
            limit_value = int(limit)
        except (TypeError, ValueError):
            limit_value = None
    if limit_value is None or limit_value <= 0:
        limit_value = default_limit
    limit_value = max(1, min(int(limit_value), 200))

    with db_lock:
        conn = get_db()
        cur = conn.execute(
            'SELECT COUNT(*) AS total FROM processed_games '
            'WHERE TRIM(COALESCE("igdb_id", "")) != ""'
        )
        total_row = cur.fetchone()

    total = 0
    if total_row is not None:
        try:
            total = int(total_row['total'])
        except (KeyError, TypeError, ValueError):
            try:
                total = int(total_row[0])
            except (IndexError, TypeError, ValueError):
                total = 0
    if total < 0:
        total = 0

    update_progress(total=total, current=0, message='Scanning processed games…')

    if total == 0:
        return {
            'status': 'ok',
            'total': 0,
            'processed': 0,
            'updated': 0,
            'unchanged': 0,
            'missing': [],
            'missing_remote': [],
            'missing_name': [],
            'invalid': 0,
            'toast_type': 'warning',
            'message': 'No games with an IGDB ID were found.',
            'done': True,
            'next_offset': 0,
        }

    current_offset = start_offset
    processed = start_offset
    updated_total = 0
    unchanged_total = 0
    invalid_total = 0
    missing_remote: set[str] = set()
    missing_name: set[str] = set()

    timestamp = now_utc_iso()

    while current_offset < total:
        with db_lock:
            conn = get_db()
            cur = conn.execute(
                '''SELECT "ID", "igdb_id", "Name" FROM processed_games
                   WHERE TRIM(COALESCE("igdb_id", "")) != ""
                   ORDER BY "ID"
                   LIMIT ? OFFSET ?''',
                (limit_value, current_offset),
            )
            rows: Iterable[Mapping[str, Any]] = cur.fetchall()

        batch_count = len(rows)
        if batch_count == 0:
            break

        entries: list[dict[str, Any]] = []
        unique_ids: list[str] = []
        seen_ids: set[str] = set()

        for row in rows:
            db_id = row['ID']
            raw_igdb_id = row['igdb_id']
            igdb_id = coerce_igdb_id(raw_igdb_id)
            current_name = normalize_text(row['Name'])
            entries.append(
                {
                    'id': db_id,
                    'igdb_id': igdb_id,
                    'current_name': current_name,
                }
            )
            if igdb_id:
                if igdb_id not in seen_ids:
                    seen_ids.add(igdb_id)
                    unique_ids.append(igdb_id)
            else:
                invalid_total += 1

        metadata: Mapping[str, Mapping[str, Any]] = {}
        if unique_ids:
            metadata = fetch_igdb_metadata(unique_ids) or {}

        updates: list[tuple[str, str, Any]] = []

        for entry in entries:
            igdb_id = entry['igdb_id']
            if not igdb_id:
                continue
            payload = metadata.get(igdb_id)
            if not isinstance(payload, Mapping):
                missing_remote.add(igdb_id)
                continue
            remote_name = normalize_text(payload.get('name'))
            if not remote_name:
                missing_name.add(igdb_id)
                continue
            current_name = entry['current_name']
            if remote_name == current_name:
                unchanged_total += 1
                continue
            db_id = entry['id']
            try:
                numeric_id = int(db_id)
            except (TypeError, ValueError):
                missing_remote.add(igdb_id)
                continue
            updates.append((remote_name, timestamp, numeric_id))
            updated_total += 1

        if updates:
            with db_lock:
                conn = get_db()
            conn.executemany(
                'UPDATE processed_games SET "Name"=?, last_edited_at=? WHERE "ID"=?',
                updates,
            )
            conn.commit()

        current_offset += batch_count
        processed = min(current_offset, total)
        update_progress(
            current=processed,
            total=total,
            message='Fixing IGDB names…',
            data={
                'updated': updated_total,
                'unchanged': unchanged_total,
                'invalid': invalid_total,
                'missing_remote': len(missing_remote),
                'missing_name': len(missing_name),
            },
        )

        if not process_all:
            break

    missing_remote_list = sorted(missing_remote)
    missing_name_list = sorted(missing_name)
    missing_combined = sorted({*missing_remote, *missing_name})

    toast_type = 'success'
    if updated_total > 0:
        message = f"Updated {updated_total} game name{'s' if updated_total != 1 else ''} from IGDB."
    else:
        message = 'No game names required updating.'
    if processed == 0:
        message = 'No games with an IGDB ID were found.'
        toast_type = 'warning'
    if missing_combined:
        plural = 's' if len(missing_combined) != 1 else ''
        message += f" {len(missing_combined)} IGDB record{plural} missing."
        toast_type = 'warning'

    update_progress(
        current=processed,
        total=total,
        message='Finished fixing IGDB names.',
        data={
            'updated': updated_total,
            'unchanged': unchanged_total,
            'invalid': invalid_total,
            'missing_remote': len(missing_remote),
            'missing_name': len(missing_name),
        },
    )

    processed_value = min(processed, total)

    return {
        'status': 'ok',
        'total': total,
        'processed': processed_value,
        'updated': updated_total,
        'unchanged': unchanged_total,
        'missing': missing_combined,
        'missing_remote': missing_remote_list,
        'missing_name': missing_name_list,
        'invalid': invalid_total,
        'toast_type': toast_type,
        'message': message.strip(),
        'done': processed_value >= total,
        'next_offset': processed_value,
    }


def remove_duplicates_job(
    update_progress: ProgressCallback,
    *,
    catalog_state: Any,
    db_lock: Any,
    get_db: Callable[[], Any],
    lookup_relations: Iterable[Mapping[str, Any]],
    scan_duplicate_candidates: Callable[..., Any],
    merge_duplicate_resolutions: Callable[[Iterable[Any]], Iterable[int]],
    remove_processed_games: Callable[[Iterable[int]], tuple[int, int]],
) -> dict[str, Any]:
    """Identify and remove duplicate processed games."""

    games_df = catalog_state.games_df

    update_progress(message='Scanning for duplicates…', data={'phase': 'dedupe'}, current=0, total=0)

    with db_lock:
        conn = get_db()
        relation_count_sql = ', '.join(
            f'(SELECT COUNT(*) FROM {relation["join_table"]} WHERE processed_game_id = pg."ID") AS {relation["join_table"]}_count'
            for relation in lookup_relations
        )
        cur = conn.execute(
            f'''SELECT
                    pg."ID",
                    pg."Source Index",
                    pg."Name",
                    pg."igdb_id",
                    pg."Summary",
                    pg."Cover Path",
                    pg."First Launch Date",
                    pg."Category",
                    pg."Width",
                    pg."Height",
                    pg.last_edited_at,
                    {relation_count_sql}
               FROM processed_games AS pg'''
        )
        rows = cur.fetchall()

    def _progress_callback(index: int, total_groups: int, duplicate_groups: int, skipped: int) -> None:
        update_progress(
            current=index,
            total=total_groups or len(rows) or 1,
            message='Evaluating duplicate groups…',
            data={
                'phase': 'dedupe',
                'duplicate_groups': duplicate_groups,
                'skipped': skipped,
            },
        )

    resolutions, duplicate_groups, skipped_groups, total_groups = scan_duplicate_candidates(
        rows, progress_callback=_progress_callback
    )

    ids_to_delete = list(merge_duplicate_resolutions(resolutions))

    if not ids_to_delete:
        remaining_total = catalog_state.total_games
        message = (
            'No removable duplicates found.'
            if duplicate_groups
            else 'No duplicates detected.'
        )
        toast_type = 'info'
        if skipped_groups and not duplicate_groups:
            toast_type = 'warning'
        update_progress(
            message=message,
            current=total_groups,
            total=total_groups or len(rows) or 1,
            data={
                'phase': 'dedupe',
                'removed': 0,
                'duplicate_groups': duplicate_groups,
                'skipped': skipped_groups,
            },
        )
        return {
            'status': 'ok',
            'removed': 0,
            'duplicate_groups': duplicate_groups,
            'skipped': skipped_groups,
            'remaining': remaining_total,
            'message': message,
            'toast_type': toast_type,
        }

    removed_count, remaining_total = remove_processed_games(ids_to_delete)

    message = (
        f"Removed {removed_count} duplicate{'s' if removed_count != 1 else ''}."
        if removed_count > 0
        else 'No removable duplicates found.'
    )
    toast_type = 'success' if removed_count > 0 else 'info'
    if skipped_groups and removed_count == 0:
        toast_type = 'warning'
        message += f" Skipped {skipped_groups} duplicate group{'s' if skipped_groups != 1 else ''}."

    update_progress(
        message='Removed duplicate entries.',
        current=total_groups or len(resolutions) or 1,
        total=total_groups or len(resolutions) or 1,
        data={
            'phase': 'dedupe',
            'removed': removed_count,
            'duplicate_groups': duplicate_groups,
            'skipped': skipped_groups,
        },
    )

    return {
        'status': 'ok',
        'removed': removed_count,
        'duplicate_groups': duplicate_groups,
        'skipped': skipped_groups,
        'remaining': remaining_total,
        'message': message.strip(),
        'toast_type': toast_type,
    }
