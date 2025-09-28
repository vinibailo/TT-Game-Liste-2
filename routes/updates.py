"""Update-related API routes."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any, Mapping

from flask import Blueprint, current_app, jsonify, render_template, request, url_for

from jobs.manager import JOB_STATUS_ERROR, JOB_STATUS_PENDING, JOB_STATUS_RUNNING
from routes.api_utils import (
    NotFoundError,
    UpstreamServiceError,
    handle_api_errors,
)
updates_blueprint = Blueprint("updates", __name__)

_context: dict[str, Any] = {}


def configure(context: Mapping[str, Any]) -> None:
    """Provide shared update helpers and background-job hooks."""
    _context.update(context)


def _ctx(key: str) -> Any:
    if key not in _context:
        raise RuntimeError(f"update routes missing context value: {key}")
    return _context[key]


def _iter_error_strings(value: Any) -> Iterable[str]:
    if not value:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        return [f"{k}: {v}" if v else str(k) for k, v in value.items() if v or k]
    if isinstance(value, Iterable):
        return [str(item) for item in value if item]
    return [str(value)]


def _collect_job_errors(job: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []

    for candidate in (
        job.get('error'),
        job.get('result', {}).get('errors')
        if isinstance(job.get('result'), Mapping)
        else None,
        job.get('data', {}).get('errors')
        if isinstance(job.get('data'), Mapping)
        else None,
    ):
        for entry in _iter_error_strings(candidate):
            text = str(entry).strip()
            if not text:
                continue
            if text not in errors:
                errors.append(text)

    return errors


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
    if not _ctx('validate_igdb_credentials')():
        return jsonify({'error': 'IGDB credentials missing'}), 503

    payload = request.get_json(silent=True) or {}
    try:
        offset = int(payload.get('offset', 0))
    except (TypeError, ValueError):
        offset = 0
    if offset < 0:
        offset = 0

    default_limit = _ctx('IGDB_BATCH_SIZE')
    try:
        limit = int(payload.get('limit', default_limit))
    except (TypeError, ValueError):
        limit = default_limit
    if not isinstance(limit, int):
        limit = default_limit if isinstance(default_limit, int) else 0
    if limit <= 0:
        limit = default_limit if isinstance(default_limit, int) and default_limit > 0 else 500
    limit = min(max(int(limit), 1), 500)

    job_manager = _ctx('job_manager')
    refresh_cache_job = _ctx('refresh_cache_job')

    run_sync = request.args.get('sync') not in (None, '', '0', 'false', 'False') or current_app.config.get('TESTING')
    if run_sync:
        try:
            result = refresh_cache_job(
                lambda **_kwargs: None,
                offset=offset,
                limit=limit,
            )
        except RuntimeError as exc:
            raise UpstreamServiceError(str(exc)) from exc
        return jsonify(result)

    job, created = job_manager.enqueue_job(
        'refresh_igdb_cache',
        'app._execute_refresh_cache_job',
        description='Refreshing IGDB cache…',
        kwargs={'offset': offset, 'limit': limit},
    )

    payload = {
        'status': 'accepted',
        'job_id': job.get('id') if job else None,
        'created': created,
    }
    response = jsonify(payload)
    if job:
        response.headers['Location'] = url_for(
            'updates.api_updates_job_detail', job_id=job['id']
        )
    status_code = 202 if created else 200
    return response, status_code


@updates_blueprint.route('/api/updates/status', methods=['GET'])
@handle_api_errors
def api_updates_status():
    job_manager = _ctx('job_manager')
    db_lock = _ctx('db_lock')
    get_db = _ctx('get_db')
    coerce_int = _ctx('_coerce_int')

    with db_lock:
        conn = get_db()
        cur = conn.execute(
            'SELECT MAX(refreshed_at) AS last_refreshed_at FROM igdb_updates'
        )
        row = cur.fetchone()

    last_refreshed_at = row['last_refreshed_at'] if row else None

    payload = {
        'phase': 'idle',
        'queued': 0,
        'processed': 0,
        'last_refreshed_at': last_refreshed_at,
        'errors': [],
    }

    active_job = job_manager.get_active_job('refresh_updates')
    if isinstance(active_job, Mapping):
        data = active_job.get('data') if isinstance(active_job.get('data'), Mapping) else {}
        phase = data.get('phase') if isinstance(data, Mapping) else None
        if not phase:
            status = active_job.get('status')
            if status == JOB_STATUS_RUNNING:
                phase = 'running'
            elif status == JOB_STATUS_PENDING:
                phase = 'pending'
            else:
                phase = 'running'

        processed = coerce_int(active_job.get('progress_current')) or 0
        total = coerce_int(active_job.get('progress_total')) or 0
        if total and processed > total:
            processed = total
        queued = max(total - processed, 0) if total else 0

        payload.update(
            phase=str(phase),
            queued=queued,
            processed=processed,
            errors=_collect_job_errors(active_job),
        )
        return jsonify(payload)

    latest_job: Mapping[str, Any] | None = None
    job_history = job_manager.list_jobs('refresh_updates')
    if job_history:
        latest_job = job_history[-1]

    if isinstance(latest_job, Mapping):
        payload['errors'] = _collect_job_errors(latest_job)
        if latest_job.get('status') == JOB_STATUS_ERROR and payload['phase'] == 'idle':
            payload['phase'] = 'error'

    return jsonify(payload)


@updates_blueprint.route('/api/updates/refresh', methods=['POST'])
@handle_api_errors
def api_updates_refresh():
    if not _ctx('validate_igdb_credentials')():
        return jsonify({'error': 'IGDB credentials missing'}), 503

    payload = request.get_json(silent=True) or {}
    try:
        offset = int(payload.get('offset', 0))
    except (TypeError, ValueError):
        offset = 0
    if offset < 0:
        offset = 0

    try:
        limit_value = int(payload.get('limit', 200))
    except (TypeError, ValueError):
        limit_value = 200
    if limit_value <= 0:
        limit_value = 200
    if limit_value < 50:
        limit_value = 50
    if limit_value > 500:
        limit_value = 500

    job_manager = _ctx('job_manager')

    job, created = job_manager.enqueue_job(
        'refresh_updates',
        'app._execute_refresh_job',
        description='Refreshing IGDB updates…',
        kwargs={'offset': offset, 'limit': limit_value},
    )

    accepted = limit_value if created else 0
    next_offset = offset + accepted
    current_app.logger.info('updates.refresh accepted=%s offset=%s', accepted, offset)

    response = jsonify({'accepted': accepted, 'next_offset': next_offset})
    status_code = 202 if created else 200
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
    try:
        offset = int(request.args.get('offset', 0))
    except (TypeError, ValueError):
        offset = 0
    if offset < 0:
        offset = 0

    try:
        limit = int(request.args.get('limit', 100))
    except (TypeError, ValueError):
        limit = 100
    if limit <= 0:
        limit = 100
    limit = min(limit, 500)

    items, total, normalized_offset = fetch_cached_updates(
        offset=offset, limit=limit
    )
    return jsonify(
        {
            'items': items,
            'total': total,
            'offset': normalized_offset,
            'limit': limit,
        }
    )


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
