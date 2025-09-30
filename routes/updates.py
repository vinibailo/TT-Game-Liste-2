"""Update-related API routes."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterable
from typing import Any, Mapping

from datetime import datetime

from flask import (
    Blueprint,
    Response,
    current_app,
    jsonify,
    render_template,
    request,
    stream_with_context,
    url_for,
)
from werkzeug.routing import BaseConverter

from updates.service import refresh_igdb_cache
from igdb.cache import get_cache_status as cache_get_status

from jobs.manager import JOB_STATUS_ERROR, JOB_STATUS_PENDING, JOB_STATUS_RUNNING
from routes.api_utils import (
    NotFoundError,
    UpstreamServiceError,
    handle_api_errors,
)
updates_blueprint = Blueprint("updates", __name__)


class HexJobIdConverter(BaseConverter):
    """Route converter for 32-character hexadecimal job identifiers."""

    regex = r"[0-9a-fA-F]{32}"

    def to_python(self, value: str) -> str:
        return str(value or "").strip().lower()

    def to_url(self, value: str) -> str:
        return super().to_url(str(value or "").strip().lower())


@updates_blueprint.record
def _register_hex_converter(setup_state: Any) -> None:
    setup_state.app.url_map.converters["hexjob"] = HexJobIdConverter

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


def _build_progress_snapshot() -> dict[str, Any]:
    """Return a normalized snapshot of background job progress."""

    job_manager = _ctx("job_manager")
    jobs = job_manager.list_jobs()
    payload: list[dict[str, Any]] = []
    for job in jobs:
        if isinstance(job, Mapping):
            payload.append(dict(job))
    return {"jobs": payload}


@updates_blueprint.route("/api/progress", methods=["GET"])
@handle_api_errors
def api_progress_snapshot():
    """Return the current background job progress snapshot."""

    return jsonify(_build_progress_snapshot())


@updates_blueprint.route("/api/progress/stream", methods=["GET"])
def api_progress_stream():
    """Stream background job progress updates using server-sent events."""

    def _event_stream():
        last_payload = ""
        last_heartbeat = time.monotonic()
        keepalive_interval = 15.0
        poll_interval = 1.0

        try:
            while True:
                try:
                    snapshot = _build_progress_snapshot()
                except Exception:  # pragma: no cover - defensive logging
                    current_app.logger.exception("Failed to build progress snapshot")
                    snapshot = {"jobs": []}

                data = json.dumps(snapshot, sort_keys=True)
                if data != last_payload:
                    yield f"data: {data}\n\n"
                    last_payload = data
                    last_heartbeat = time.monotonic()
                elif time.monotonic() - last_heartbeat >= keepalive_interval:
                    yield ": keep-alive\n\n"
                    last_heartbeat = time.monotonic()

                time.sleep(poll_interval)
        except GeneratorExit:  # pragma: no cover - connection closed
            return

    response = Response(
        stream_with_context(_event_stream()),
        mimetype="text/event-stream",
    )
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"
    return response


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
        coerce_int = _ctx('_coerce_int')
        exchange_helper = _ctx('exchange_twitch_credentials')
        try:
            helper_result = exchange_helper() if callable(exchange_helper) else exchange_helper
            if callable(helper_result):
                exchange_callable = helper_result
                access_token, client_id = exchange_callable()
            elif isinstance(helper_result, (tuple, list)) and len(helper_result) >= 2:
                exchange_callable = None
                access_token, client_id = helper_result[0], helper_result[1]
            else:
                raise RuntimeError('Unable to resolve IGDB credentials')

            download_total_helper = _ctx('download_igdb_game_count')
            download_total = (
                download_total_helper()
                if callable(download_total_helper)
                else download_total_helper
            )
            download_games_helper = _ctx('download_igdb_games')
            download_games = (
                download_games_helper()
                if callable(download_games_helper)
                else download_games_helper
            )

            result = refresh_igdb_cache(
                access_token,
                client_id,
                offset,
                limit,
                conn=_ctx('get_db')(),
                db_lock=_ctx('db_lock'),
                get_cached_total=_ctx('_get_cached_igdb_total'),
                set_cached_total=_ctx('_set_cached_igdb_total'),
                download_total=download_total,
                download_games=download_games,
                upsert_games=_ctx('_upsert_igdb_cache_entries'),
                exchange_credentials=exchange_callable,
                igdb_prefill_cache=_ctx('_igdb_prefill_cache'),
                igdb_prefill_lock=_ctx('_igdb_prefill_lock'),
            )
        except RuntimeError as exc:
            raise UpstreamServiceError(str(exc)) from exc
        payload: dict[str, Any] = dict(result) if isinstance(result, Mapping) else {'status': 'error'}

        status_value = str(payload.get('status') or '').lower()
        inserted = coerce_int(payload.get('inserted')) or 0
        updated = coerce_int(payload.get('updated')) or 0
        batch_count = coerce_int(payload.get('batch_count')) or 0
        done = bool(payload.get('done'))
        if status_value == 'ok' and not done and batch_count < limit:
            done = True
            payload['done'] = True

        formatter = _context.get('format_refresh_response')
        if callable(formatter):
            formatted = formatter(dict(payload), offset=offset, limit=limit)
            if isinstance(formatted, Mapping):
                return jsonify(dict(formatted))

        message: str
        toast_type = 'info'
        if status_value != 'ok':
            error_message = str(payload.get('error') or 'Unknown error').strip()
            if not error_message:
                error_message = 'Unknown error'
            message = f'IGDB cache refresh failed: {error_message}'
            toast_type = 'error'
        else:
            changed = inserted + updated
            if changed > 0:
                suffix = 'record' if changed == 1 else 'records'
                message = f'Cached {changed} IGDB {suffix}.'
                toast_type = 'success'
            elif done:
                message = 'IGDB cache is already up to date.'
            else:
                message = 'Refreshing IGDB cache…'

        payload.pop('batch_count', None)

        payload.update(
            message=message,
            toast_type=toast_type,
            offset=offset,
            limit=limit,
            done=done,
        )
        return jsonify(payload)

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


@updates_blueprint.route('/api/updates/cache-status', methods=['GET'])
@handle_api_errors
def api_updates_cache_status():
    db_lock = _ctx('db_lock')
    get_db = _ctx('get_db')
    job_manager = _ctx('job_manager')
    coerce_int = _ctx('_coerce_int')

    with db_lock:
        conn = get_db()
        cache_status = cache_get_status(conn)

    payload: dict[str, Any] = {
        'cached_entries': coerce_int(cache_status.get('cached_entries')) or 0,
        'remote_total': coerce_int(cache_status.get('remote_total')),
        'last_synced_at': cache_status.get('last_synced_at'),
        'last_refresh': None,
    }

    job_history = job_manager.list_jobs('refresh_updates')
    last_job: Mapping[str, Any] | None = None
    last_summary: Mapping[str, Any] | None = None
    for job in reversed(job_history):
        if not isinstance(job, Mapping):
            continue
        result = job.get('result')
        if isinstance(result, Mapping):
            summary = result.get('cache_summary')
            if isinstance(summary, Mapping):
                last_job = job
                last_summary = summary
                break

    if last_summary is not None:
        inserted = coerce_int(last_summary.get('inserted')) or 0
        updated = coerce_int(last_summary.get('updated')) or 0
        unchanged = coerce_int(last_summary.get('unchanged')) or 0
        total = last_summary.get('total')
        processed = last_summary.get('processed')
        message = last_summary.get('message') if isinstance(last_summary.get('message'), str) else None
        finished_at = None
        if isinstance(last_summary.get('finished_at'), str):
            finished_at = last_summary.get('finished_at')
        elif isinstance(last_job, Mapping):
            finished_at = last_job.get('finished_at') or last_job.get('updated_at')

        payload['last_refresh'] = {
            'inserted': inserted,
            'updated': updated,
            'unchanged': unchanged,
            'total': coerce_int(total),
            'processed': coerce_int(processed),
            'message': message,
            'finished_at': finished_at,
            'started_at': last_job.get('started_at') if isinstance(last_job, Mapping) else None,
        }

    return jsonify(payload)


@updates_blueprint.route('/api/updates/status', methods=['GET'])
@handle_api_errors
def api_updates_status():
    job_manager = _ctx('job_manager')
    db_lock = _ctx('db_lock')
    get_db = _ctx('get_db')
    coerce_int = _ctx('_coerce_int')
    timeout_fn = _context.get('get_igdb_timeout_count')
    timeout_errors: list[str] = []
    if callable(timeout_fn):
        try:
            timeout_count = int(timeout_fn())
        except Exception:  # pragma: no cover - defensive guard
            timeout_count = 0
        if timeout_count > 0:
            timeout_errors.append(f'IGDB timeouts: {timeout_count}')

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
            errors=_collect_job_errors(active_job) + timeout_errors,
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

    if timeout_errors:
        payload['errors'] = list(payload.get('errors', [])) + timeout_errors

    return jsonify(payload)


@updates_blueprint.route('/api/updates/refresh', methods=['POST'])
@handle_api_errors
def api_updates_refresh():
    if not _ctx('validate_igdb_credentials')():
        return jsonify({'error': 'IGDB credentials missing'}), 503

    try:
        offset = int(request.args.get('offset', 0))
    except (TypeError, ValueError):
        offset = 0
    if offset < 0:
        offset = 0

    try:
        limit_value = int(request.args.get('limit', 200))
    except (TypeError, ValueError):
        limit_value = 200
    if limit_value <= 0:
        limit_value = 200
    if limit_value > 500:
        limit_value = 500
    if limit_value < 1:
        limit_value = 1

    exchange_helper = _ctx('exchange_twitch_credentials')
    try:
        helper_result = exchange_helper() if callable(exchange_helper) else exchange_helper
        if callable(helper_result):
            exchange_callable = helper_result
            access_token, client_id = exchange_callable()
        elif isinstance(helper_result, (tuple, list)) and len(helper_result) >= 2:
            exchange_callable = None
            access_token, client_id = helper_result[0], helper_result[1]
        else:
            raise RuntimeError('Unable to resolve IGDB credentials')

        download_total_helper = _ctx('download_igdb_game_count')
        download_total = (
            download_total_helper()
            if callable(download_total_helper)
            else download_total_helper
        )
        download_games_helper = _ctx('download_igdb_games')
        download_games = (
            download_games_helper()
            if callable(download_games_helper)
            else download_games_helper
        )

        result = refresh_igdb_cache(
            access_token,
            client_id,
            offset,
            limit_value,
            conn=_ctx('get_db')(),
            db_lock=_ctx('db_lock'),
            get_cached_total=_ctx('_get_cached_igdb_total'),
            set_cached_total=_ctx('_set_cached_igdb_total'),
            download_total=download_total,
            download_games=download_games,
            upsert_games=_ctx('_upsert_igdb_cache_entries'),
            exchange_credentials=exchange_callable,
            igdb_prefill_cache=_ctx('_igdb_prefill_cache'),
            igdb_prefill_lock=_ctx('_igdb_prefill_lock'),
        )
    except RuntimeError as exc:
        return jsonify({'error': str(exc)}), 502

    progress = {
        'total': result.get('total', 0) if isinstance(result, Mapping) else 0,
        'processed': result.get('processed', 0) if isinstance(result, Mapping) else 0,
        'inserted': result.get('inserted', 0) if isinstance(result, Mapping) else 0,
        'updated': result.get('updated', 0) if isinstance(result, Mapping) else 0,
        'unchanged': result.get('unchanged', 0) if isinstance(result, Mapping) else 0,
        'done': bool(result.get('done')) if isinstance(result, Mapping) else False,
        'next_offset': result.get('next_offset', offset) if isinstance(result, Mapping) else offset,
    }

    current_app.logger.info(
        'updates.refresh batch offset=%s limit=%s processed=%s',
        offset,
        limit_value,
        progress['processed'],
    )
    return jsonify(progress)


@updates_blueprint.route('/api/updates/compare', methods=['POST'])
@handle_api_errors
def api_updates_compare():
    job_manager = _ctx('job_manager')
    compare_updates_job = _ctx('compare_updates_job')

    run_sync = request.args.get('sync') not in (None, '', '0', 'false', 'False') or current_app.config.get('TESTING')
    if run_sync:
        return jsonify(compare_updates_job(lambda **_kwargs: None))

    job, created = job_manager.enqueue_job(
        'compare_updates',
        'app._execute_compare_updates_job',
        description='Comparing IGDB cache with processed games…',
    )

    status_code = 202 if created else 200
    response = jsonify({'status': 'accepted', 'job': job, 'created': created})
    if job:
        response.headers['Location'] = url_for('updates.api_updates_job_detail', job_id=job['id'])
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


@updates_blueprint.route('/api/updates/<hexjob:job_id>', methods=['GET'])
@handle_api_errors
def api_updates_job_detail(job_id: str):
    job_manager = _ctx('job_manager')
    job = job_manager.get_job(job_id)
    if job is None:
        raise NotFoundError('job not found')
    return jsonify(job)


@updates_blueprint.route('/api/updates/jobs/<job_id>', methods=['GET'])
@handle_api_errors
def api_updates_job_detail_legacy(job_id: str):
    """Backward compatible job detail endpoint."""

    normalized = str(job_id or '').strip().lower()
    if len(normalized) == 33 and normalized.endswith('s'):
        normalized = normalized[:-1]
    return api_updates_job_detail(normalized)


@updates_blueprint.route('/api/updates', methods=['GET'])
@handle_api_errors
def api_updates_list():
    fetch_cached_updates = _ctx('fetch_cached_updates')

    def _normalize_since(value: str | None) -> str | None:
        if not value or not isinstance(value, str):
            return None
        text = value.strip()
        if not text:
            return None
        if 'T' in text and ' ' in text and '+' not in text and '-' not in text[text.rfind(' ') + 1 :]:
            text = text.replace(' ', '+', 1)
        elif 'T' in text and ' ' in text and '+' not in text:
            text = text.replace(' ', '+', 1)
        if text.endswith('Z'):
            text = f"{text[:-1]}+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
        return parsed.isoformat()

    def _entry_updated(entry: Mapping[str, Any]) -> str:
        for key in ('refreshed_at', 'local_last_edited_at', 'igdb_updated_at'):
            value = entry.get(key)
            if value:
                return str(value)
        return ''

    try:
        raw_limit = int(request.args.get('limit', 100))
    except (TypeError, ValueError):
        raw_limit = 100
    if raw_limit <= 0:
        raw_limit = 100
    limit = min(raw_limit, 500)

    cursor_param = request.args.get('cursor')
    cursor_value = cursor_param.strip() if isinstance(cursor_param, str) else None
    cursor_value = cursor_value or None

    since_param = request.args.get('since')
    normalized_since = _normalize_since(since_param)

    entries, total_count_value, next_cursor, has_more = fetch_cached_updates(
        cursor=cursor_value,
        limit=limit,
    )

    if normalized_since:
        filtered_entries = [
            entry
            for entry in entries
            if _entry_updated(entry) and _entry_updated(entry) > normalized_since
        ]
        entries = filtered_entries
        has_more = False
        next_cursor = None

    items: list[dict[str, Any]] = []
    for entry in entries[:limit]:
        updated_at_value = _entry_updated(entry)
        items.append(
            {
                'id': entry.get('processed_game_id'),
                'processed_game_id': entry.get('processed_game_id'),
                'igdb_id': entry.get('igdb_id'),
                'igdb_updated_at': entry.get('igdb_updated_at'),
                'local_last_edited_at': entry.get('local_last_edited_at'),
                'refreshed_at': entry.get('refreshed_at'),
                'updated_at': updated_at_value,
                'name': entry.get('name'),
                'has_diff': bool(entry.get('has_diff')),
                'cover': entry.get('cover'),
                'cover_available': bool(entry.get('cover_available')),
                'update_type': entry.get('update_type') or 'mismatch',
                'detail_available': bool(entry.get('detail_available')),
            }
        )

    max_updated_at_value = ''
    if items:
        max_updated_at_value = max(_entry_updated(entry) for entry in entries[:limit]) or ''
    elif normalized_since:
        max_updated_at_value = normalized_since

    etag_source = f"{max_updated_at_value}:{int(total_count_value or 0)}"
    etag_hash = hashlib.sha256(etag_source.encode('utf-8')).hexdigest()
    etag_header_value = f'W/"{etag_hash}"'

    if_none_match_header = request.headers.get('If-None-Match', '')
    if if_none_match_header:
        candidates = [token.strip() for token in if_none_match_header.split(',') if token.strip()]
        normalized_candidates = set(candidates)
        for candidate in candidates:
            if candidate.startswith('W/'):
                normalized_candidates.add(candidate[2:])
        target_values = {etag_header_value}
        if etag_header_value.startswith('W/'):
            target_values.add(etag_header_value[2:])
        if '*' in normalized_candidates or target_values & normalized_candidates:
            response = Response(status=304)
            response.headers['ETag'] = etag_header_value
            response.headers['Cache-Control'] = 'max-age=30, stale-while-revalidate=120'
            return response

    payload: dict[str, Any] = {
        'items': items,
        'total': int(total_count_value or 0),
        'limit': limit,
        'has_more': bool(has_more and next_cursor),
        'nextAfter': None,
    }
    if payload['has_more'] and next_cursor:
        payload['next_cursor'] = next_cursor
    else:
        payload['has_more'] = False

    response = jsonify(payload)
    response.headers['ETag'] = etag_header_value
    response.headers['Cache-Control'] = 'max-age=30, stale-while-revalidate=120'
    return response

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
    cover_available = bool(row['cover_path'] or row['cover_url'])
    cover_data = load_cover_data(row['cover_path'], row['cover_url'])

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
            'cover': cover_data,
            'cover_available': cover_available,
        }
    )


@updates_blueprint.route('/api/updates/<int:processed_game_id>/cover', methods=['GET'])
@handle_api_errors
def api_updates_cover(processed_game_id: int):
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
                   p."Cover Path" AS cover_path,
                   {cover_url_select}
               FROM processed_games p
               WHERE p."ID"=?''',
            (processed_game_id,),
        )
        row = cur.fetchone()

    if row is None:
        raise NotFoundError('not found')

    cover_data = load_cover_data(row['cover_path'], row['cover_url'])
    if not cover_data:
        raise NotFoundError('not found')

    return jsonify({'cover': cover_data})
