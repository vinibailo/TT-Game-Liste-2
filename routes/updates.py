"""Update-related API routes."""

from __future__ import annotations

import json
from typing import Any, Mapping

from flask import Blueprint, current_app, jsonify, render_template, request, url_for

from routes.api_utils import (
    BadRequestError,
    NotFoundError,
    UpstreamServiceError,
    handle_api_errors,
)
from updates import service as updates_service

updates_blueprint = Blueprint("updates", __name__)

_context: dict[str, Any] = {}


def configure(context: Mapping[str, Any]) -> None:
    """Provide shared update helpers and background-job hooks."""
    _context.update(context)


def _ctx(key: str) -> Any:
    if key not in _context:
        raise RuntimeError(f"update routes missing context value: {key}")
    return _context[key]


@updates_blueprint.route('/updates')
def updates_page():
    return render_template(
        'updates.html',
        igdb_batch_size=_ctx('IGDB_BATCH_SIZE'),
        FIX_NAMES_BATCH_LIMIT=_ctx('FIX_NAMES_BATCH_LIMIT'),
    )


@updates_blueprint.route('/api/igdb/cache', methods=['POST'])
@handle_api_errors
def api_igdb_cache_refresh():
    payload = request.get_json(silent=True) or {}
    offset = payload.get('offset', 0)
    limit = payload.get('limit', _ctx('IGDB_BATCH_SIZE'))

    exchange_twitch_credentials = _ctx('exchange_twitch_credentials')()
    try:
        access_token, client_id = exchange_twitch_credentials()
    except RuntimeError as exc:
        raise BadRequestError(str(exc)) from exc

    db_lock = _ctx('db_lock')
    get_db = _ctx('get_db')
    get_cached_total = _ctx('_get_cached_igdb_total')
    set_cached_total = _ctx('_set_cached_igdb_total')
    download_igdb_game_count = _ctx('download_igdb_game_count')()
    download_igdb_games = _ctx('download_igdb_games')()
    upsert_cache_entries = _ctx('_upsert_igdb_cache_entries')
    igdb_prefill_lock = _ctx('_igdb_prefill_lock')
    igdb_prefill_cache = _ctx('_igdb_prefill_cache')

    try:
        result = updates_service.refresh_igdb_cache(
            access_token,
            client_id,
            offset,
            limit,
            db_lock=db_lock,
            get_db=get_db,
            get_cached_total=get_cached_total,
            set_cached_total=set_cached_total,
            download_total=download_igdb_game_count,
            download_games=download_igdb_games,
            upsert_games=upsert_cache_entries,
            igdb_prefill_cache=igdb_prefill_cache,
            igdb_prefill_lock=igdb_prefill_lock,
        )
    except RuntimeError as exc:
        raise UpstreamServiceError(str(exc)) from exc

    total = result.get('total') or 0
    processed = result.get('processed') or 0
    try:
        total_value = max(int(total), 0)
    except (TypeError, ValueError):
        total_value = 0
    try:
        processed_value = max(int(processed), 0)
    except (TypeError, ValueError):
        processed_value = 0

    progress_total = total_value if total_value > 0 else processed_value
    progress_current = (
        min(processed_value, progress_total)
        if progress_total > 0
        else processed_value
    )
    progress_percent = 0.0
    if progress_total > 0:
        progress_percent = min(
            100.0,
            max(0.0, (progress_current / progress_total) * 100),
        )

    response_payload = {
        **result,
        'progress_total': progress_total,
        'progress_current': progress_current,
        'progress_percent': progress_percent,
    }

    return jsonify(response_payload)


@updates_blueprint.route('/api/updates/refresh', methods=['POST'])
@handle_api_errors
def api_updates_refresh():
    run_sync = request.args.get('sync') not in (None, '', '0', 'false', 'False') or current_app.config.get('TESTING')
    execute_refresh_job = _ctx('_execute_refresh_job')
    job_manager = _ctx('job_manager')

    if run_sync:
        try:
            result = execute_refresh_job(lambda **_kwargs: None)
        except RuntimeError as exc:
            raise UpstreamServiceError(str(exc)) from exc
        return jsonify(result)

    job, created = job_manager.enqueue_job(
        'refresh_updates',
        'app._execute_refresh_job',
        description='Refreshing IGDB updates…',
    )
    status_code = 202 if created else 200
    response = jsonify({'status': 'accepted', 'job': job, 'created': created})
    if job:
        response.headers['Location'] = url_for(
            'updates.api_updates_job_detail', job_id=job['id']
        )
    return response, status_code


@updates_blueprint.route('/api/updates/fix-names', methods=['POST'])
@handle_api_errors
def api_updates_fix_names():
    payload = request.get_json(silent=True) or {}
    limit_default = _ctx('FIX_NAMES_BATCH_LIMIT')
    try:
        offset = int(payload.get('offset', 0))
    except (TypeError, ValueError):
        offset = 0
    try:
        limit = int(payload.get('limit', limit_default))
    except (TypeError, ValueError):
        limit = limit_default
    if offset < 0:
        offset = 0
    if limit <= 0:
        limit = limit_default
    job_manager = _ctx('job_manager')
    execute_fix_names_job = _ctx('fix_names_job')

    run_sync = request.args.get('sync') not in (None, '', '0', 'false', 'False') or current_app.config.get('TESTING')
    if run_sync:
        result = execute_fix_names_job(
            lambda **_kwargs: None,
            offset=offset,
            limit=limit,
            process_all=False,
        )
        return jsonify(result)

    job, created = job_manager.enqueue_job(
        'fix_names',
        'app._execute_fix_names_job',
        description='Fixing IGDB names…',
    )
    status_code = 202 if created else 200
    response = jsonify({'status': 'accepted', 'job': job, 'created': created})
    if job:
        response.headers['Location'] = url_for(
            'updates.api_updates_job_detail', job_id=job['id']
        )
    return response, status_code


@updates_blueprint.route('/api/updates/remove-duplicates', methods=['POST'])
@handle_api_errors
def api_updates_remove_duplicates():
    job_manager = _ctx('job_manager')
    execute_remove_duplicates_job = _ctx('remove_duplicates_job')

    run_sync = request.args.get('sync') not in (None, '', '0', 'false', 'False') or current_app.config.get('TESTING')
    if run_sync:
        return jsonify(execute_remove_duplicates_job(lambda **_kwargs: None))

    job, created = job_manager.enqueue_job(
        'remove_duplicates',
        'app._execute_remove_duplicates_job',
        description='Removing duplicates…',
    )
    status_code = 202 if created else 200
    response = jsonify({'status': 'accepted', 'job': job, 'created': created})
    if job:
        response.headers['Location'] = url_for(
            'updates.api_updates_job_detail', job_id=job['id']
        )
    return response, status_code


@updates_blueprint.route('/api/updates/remove-duplicate/<int:processed_game_id>', methods=['POST'])
@handle_api_errors
def api_updates_remove_duplicate(processed_game_id: int):
    db_lock = _ctx('db_lock')
    get_db = _ctx('get_db')
    lookup_relations = _ctx('LOOKUP_RELATIONS')
    scan_duplicate_candidates = _ctx('_scan_duplicate_candidates')
    coerce_int = _ctx('_coerce_int')
    compute_metadata_updates = _ctx('_compute_metadata_updates')
    merge_duplicate_resolutions = _ctx('_merge_duplicate_resolutions')
    remove_processed_games = _ctx('_remove_processed_games')
    normalize_text = _ctx('_normalize_text')
    duplicate_group_resolution = _ctx('DuplicateGroupResolution')

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

    resolutions, duplicate_groups, skipped_groups, _ = scan_duplicate_candidates(rows)
    target_resolution = None
    for resolution in resolutions:
        for entry in resolution.duplicates:
            entry_id = coerce_int(entry['ID'])
            if entry_id != processed_game_id:
                continue
            metadata_updates = compute_metadata_updates(resolution.canonical, [entry])
            target_resolution = duplicate_group_resolution(
                canonical=resolution.canonical,
                duplicates=[entry],
                metadata_updates=metadata_updates,
            )
            break
        if target_resolution is not None:
            break

    if target_resolution is None:
        raise NotFoundError('Duplicate not found or already processed.')

    ids_to_delete = merge_duplicate_resolutions([target_resolution])
    if processed_game_id not in ids_to_delete:
        raise BadRequestError('Unable to remove duplicate entry.')

    removed_count, remaining_total = remove_processed_games(ids_to_delete)
    if removed_count <= 0:
        raise BadRequestError('Unable to remove duplicate entry.')

    name_value = normalize_text(target_resolution.duplicates[0]['Name']) or f'ID {processed_game_id}'
    message = f'Removed duplicate entry for {name_value}.'
    return jsonify(
        {
            'status': 'ok',
            'removed': removed_count,
            'remaining': remaining_total,
            'duplicate_groups': duplicate_groups,
            'skipped': skipped_groups,
            'removed_id': processed_game_id,
            'message': message,
            'toast_type': 'success',
        }
    )


@updates_blueprint.route('/api/updates/jobs', methods=['GET'])
@handle_api_errors
def api_updates_job_list():
    job_manager = _ctx('job_manager')
    return jsonify({'jobs': job_manager.list_jobs()})


@updates_blueprint.route('/api/updates/jobs/<job_id>', methods=['GET'])
@handle_api_errors
def api_updates_job_detail(job_id: str):
    job_manager = _ctx('job_manager')
    job = job_manager.get_job(job_id)
    if job is None:
        raise NotFoundError('job not found')
    return jsonify(job)


@updates_blueprint.route('/api/updates', methods=['GET'])
@handle_api_errors
def api_updates_list():
    fetch_cached_updates = _ctx('fetch_cached_updates')
    return jsonify({'updates': fetch_cached_updates()})


@updates_blueprint.route('/api/updates/<int:processed_game_id>', methods=['GET'])
@handle_api_errors
def api_updates_detail(processed_game_id: int):
    db_lock = _ctx('db_lock')
    get_db = _ctx('get_db')
    get_processed_games_columns = _ctx('get_processed_games_columns')
    load_cover_data = _ctx('load_cover_data')

    with db_lock:
        conn = get_db()
        processed_columns = get_processed_games_columns(conn)
        cover_url_select = (
            'p."Large Cover Image (URL)" AS cover_url'
            if 'Large Cover Image (URL)' in processed_columns
            else 'NULL AS cover_url'
        )
        cur = conn.execute(
            f'''SELECT
                   u.processed_game_id,
                   u.igdb_id,
                   u.igdb_updated_at,
                   u.igdb_payload,
                   u.diff,
                   u.local_last_edited_at,
                   u.refreshed_at,
                   p."Name" AS game_name,
                   p."Cover Path" AS cover_path,
                   {cover_url_select}
               FROM igdb_updates u
               LEFT JOIN processed_games p ON p."ID" = u.processed_game_id
               WHERE u.processed_game_id=?''',
            (processed_game_id,),
        )
        row = cur.fetchone()

    if row is None:
        raise NotFoundError('not found')

    payload = json.loads(row['igdb_payload']) if row['igdb_payload'] else None
    diff = json.loads(row['diff']) if row['diff'] else {}

    return jsonify(
        {
            'processed_game_id': row['processed_game_id'],
            'igdb_id': row['igdb_id'],
            'igdb_updated_at': row['igdb_updated_at'],
            'igdb_payload': payload,
            'diff': diff,
            'local_last_edited_at': row['local_last_edited_at'],
            'refreshed_at': row['refreshed_at'],
            'name': row['game_name'],
            'cover': load_cover_data(row['cover_path'], row['cover_url']),
        }
    )
