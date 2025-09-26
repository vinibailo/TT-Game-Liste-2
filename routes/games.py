"""Game workflow API routes."""

from __future__ import annotations

import base64
import io
import os
import sqlite3
import uuid
from typing import Any, Callable, Mapping

from flask import Blueprint, jsonify, request

from routes.api_utils import (
    APIError,
    BadRequestError,
    ConflictError,
    NotFoundError,
    handle_api_errors,
)

# TODO: Register endpoints for editor workflows, uploads, and navigation.
games_blueprint = Blueprint("games", __name__)

_context: dict[str, Any] = {}


def configure(context: Mapping[str, Any]) -> None:
    """Provide shared state required by the game editor endpoints."""
    _context.update(context)


def _ctx(key: str) -> Any:
    if key not in _context:
        raise RuntimeError(f"games routes missing context value: {key}")
    return _context[key]


def _get_total_games() -> int:
    getter: Callable[[], int] = _ctx("get_total_games")
    return getter()


def _get_games_df():
    getter: Callable[[], Any] = _ctx('get_games_df')
    return getter()


def _get_navigator():
    getter: Callable[[], Any] = _ctx('get_navigator')
    return getter()


@games_blueprint.route('/api/game')
@handle_api_errors
def api_game():
    navigator = _get_navigator()
    index = navigator.current()
    if index >= _get_total_games():
        return jsonify(
            {
                'done': True,
                'message': 'Todos os jogos foram processados.',
                'completion': navigator.completion_percentage(),
            }
        )
    data = _ctx('build_game_payload')(
        index,
        navigator.seq_index,
        navigator.processed_total + 1,
    )
    data['completion'] = navigator.completion_percentage()
    return jsonify(data)


@games_blueprint.route('/api/game/<int:index>/raw')
@handle_api_errors
def api_game_raw(index: int):
    total_games = _get_total_games()
    if index < 0 or index >= total_games:
        raise NotFoundError('invalid index')
    games_df = _get_games_df()
    try:
        row = games_df.iloc[index]
    except Exception as exc:  # pragma: no cover - mirrors existing defensive behaviour
        raise NotFoundError('invalid index') from exc

    extract_igdb_id = _ctx('extract_igdb_id')
    get_igdb_prefill_for_id = _ctx('get_igdb_prefill_for_id')
    load_cover_data = _ctx('load_cover_data')
    extract_list = _ctx('extract_list')
    get_cell = _ctx('get_cell')
    navigator = _get_navigator()

    source_row = row.copy()
    fallback_cover_url = str(source_row.get('Large Cover Image (URL)', '') or '')
    igdb_id = extract_igdb_id(source_row, allow_generic_id=True)
    prefill = get_igdb_prefill_for_id(igdb_id)
    if prefill:
        for key, value in prefill.items():
            if key in source_row.index:
                source_row.at[key] = value
            else:
                source_row[key] = value
        cover_override = prefill.get('Large Cover Image (URL)')
        if cover_override:
            fallback_cover_url = cover_override
        igdb_id = prefill.get('IGDB ID') or prefill.get('igdb_id') or igdb_id
    if not igdb_id:
        igdb_id = extract_igdb_id(row, allow_generic_id=True)

    cover_data = load_cover_data(None, fallback_cover_url)
    genres = extract_list(source_row, ['Genres', 'Genre'])
    modes = extract_list(source_row, ['Game Modes', 'Mode'])
    platforms = extract_list(source_row, ['Platforms', 'Platform'])
    dummy: list[str] = []
    game_fields = {
        'Name': get_cell(source_row, 'Name', dummy),
        'Summary': get_cell(source_row, 'Summary', dummy),
        'FirstLaunchDate': get_cell(source_row, 'First Launch Date', dummy),
        'Developers': get_cell(source_row, 'Developers', dummy),
        'Publishers': get_cell(source_row, 'Publishers', dummy),
        'Genres': genres,
        'GameModes': modes,
        'Category': get_cell(source_row, 'Category', dummy),
        'Platforms': platforms,
        'IGDBID': igdb_id or None,
    }
    return jsonify({
        'index': int(index),
        'total': total_games,
        'game': game_fields,
        'cover': cover_data,
        'seq': navigator.processed_total + 1,
    })


@games_blueprint.route('/api/summary', methods=['POST'])
@handle_api_errors
def api_summary():
    data = request.get_json(force=True)
    game_name = data.get('game_name', '')
    try:
        summary_pt = _ctx('generate_pt_summary')(game_name)
    except Exception as exc:  # pragma: no cover - defensive logging
        raise APIError('Failed to generate summary') from exc
    return jsonify({'summary': summary_pt})


@games_blueprint.route('/api/upload', methods=['POST'])
@handle_api_errors
def api_upload():
    file = request.files.get('file')
    if not file:
        raise BadRequestError('no file')
    try:
        img = _ctx('open_image_auto_rotate')(file.stream)
    except Exception as exc:
        raise BadRequestError('invalid image upload') from exc
    filename = f"{uuid.uuid4().hex}.jpg"
    upload_dir = _ctx('upload_dir')
    path = os.path.join(upload_dir, filename)
    img.save(path, format='JPEG')
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    data = 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()
    return jsonify({'filename': filename, 'data': data})


@games_blueprint.route('/api/save', methods=['POST'])
@handle_api_errors
def api_save():
    data = request.get_json(force=True)
    try:
        index = int(data.get('index', 0))
    except (TypeError, ValueError) as exc:
        raise BadRequestError('invalid index') from exc
    expected_id = data.get('id')
    fields = data.get('fields', {})
    image_b64 = data.get('image')
    upload_name = data.get('upload_name')

    if expected_id is None:
        raise BadRequestError('missing id')
    try:
        expected_id = int(expected_id)
    except (TypeError, ValueError) as exc:
        raise BadRequestError('invalid id') from exc

    games_df = _get_games_df()
    try:
        total_rows = len(games_df)
    except Exception:
        total_rows = 0

    get_source_index_for_position = _ctx('get_source_index_for_position')
    navigator = _get_navigator()
    db_lock = _ctx('db_lock')
    get_db = _ctx('get_db')
    is_processed_game_done = _ctx('is_processed_game_done')
    lookups = _ctx('LOOKUP_RELATIONS')
    resolve_lookup_selection = _ctx('_resolve_lookup_selection')
    lookup_display_text = _ctx('_lookup_display_text')
    encode_lookup_id_list = _ctx('_encode_lookup_id_list')
    persist_lookup_relations = _ctx('_persist_lookup_relations')
    now_utc_iso = _ctx('now_utc_iso')
    extract_igdb_id = _ctx('extract_igdb_id')
    coerce_igdb_id = _ctx('coerce_igdb_id')
    upload_dir = _ctx('upload_dir')
    processed_dir = _ctx('processed_dir')

    if 0 <= index < total_rows:
        try:
            source_index = get_source_index_for_position(index)
        except IndexError:
            source_index = str(index)
    else:
        source_index = str(index)

    with navigator.lock:
        if index != navigator.current_index:
            expected_index = navigator.current_index
            expected_seq_id = navigator.seq_index
            if 0 <= expected_index < total_rows:
                try:
                    expected_source_index = get_source_index_for_position(expected_index)
                except IndexError:
                    expected_source_index = str(expected_index)
            else:
                expected_source_index = str(expected_index)
            with db_lock:
                conn = get_db()
                cur = conn.execute(
                    'SELECT "ID" FROM processed_games WHERE "Source Index"=?',
                    (expected_source_index,),
                )
                row = cur.fetchone()
                if row is not None:
                    expected_seq_id = row['ID']
            raise ConflictError(
                'index mismatch',
                payload={
                    'expected': expected_index,
                    'actual': index,
                    'expected_id': expected_seq_id,
                },
            )

        was_processed_before = False
        existing_summary = None
        existing_cover_path = None
        existing_width = None
        existing_height = None
        new_record = False
        existing: Mapping[str, Any] | None = None

        with db_lock:
            conn = get_db()
            cur = conn.execute(
                'SELECT "ID", "igdb_id", "Summary", "Cover Path", "Width", "Height" '
                'FROM processed_games WHERE "Source Index"=?',
                (source_index,),
            )
            existing = cur.fetchone()
            if existing:
                existing_id = existing['ID']
                if existing_id != expected_id:
                    raise ConflictError(
                        'id mismatch',
                        payload={'expected': existing_id, 'actual': expected_id},
                    )
                seq_id = existing_id
                try:
                    existing_summary = existing['Summary']
                except (KeyError, IndexError, TypeError):
                    existing_summary = None
                try:
                    existing_cover_path = existing['Cover Path']
                except (KeyError, IndexError, TypeError):
                    existing_cover_path = None
                try:
                    existing_width = existing['Width']
                except (KeyError, IndexError, TypeError):
                    existing_width = None
                try:
                    existing_height = existing['Height']
                except (KeyError, IndexError, TypeError):
                    existing_height = None
                was_processed_before = is_processed_game_done(
                    existing_summary, existing_cover_path
                )
            else:
                seq_id = navigator.seq_index
                if expected_id != seq_id:
                    raise ConflictError(
                        'id mismatch',
                        payload={'expected': seq_id, 'actual': expected_id},
                    )
                new_record = True

        cover_path = ''
        width = height = 0
        if image_b64:
            header, b64data = image_b64.split(',', 1)
            _ = header
            try:
                decoded = base64.b64decode(b64data)
            except Exception as exc:
                raise BadRequestError('invalid image upload') from exc
            try:
                img = _ctx('open_image_auto_rotate')(io.BytesIO(decoded))
            except Exception as exc:
                raise BadRequestError('invalid image upload') from exc
            cover_path = os.path.join(processed_dir, f"{seq_id}.jpg")
            save_cover_image = _ctx('save_cover_image')
            _processed_img, width, height = save_cover_image(img, cover_path)
        elif existing_cover_path:
            cover_path = str(existing_cover_path)
            try:
                width = int(existing_width)
            except (TypeError, ValueError):
                width = 0
            try:
                height = int(existing_height)
            except (TypeError, ValueError):
                height = 0

        igdb_id_value = None
        if existing is not None:
            try:
                existing_raw = existing['igdb_id']
            except (KeyError, IndexError):
                existing_raw = None
            existing_coerced = coerce_igdb_id(existing_raw)
            if existing_coerced:
                igdb_id_value = existing_coerced
        if igdb_id_value is None and 0 <= index < len(games_df):
            igdb_candidate = extract_igdb_id(
                games_df.iloc[index], allow_generic_id=True
            )
            if igdb_candidate:
                igdb_id_value = igdb_candidate

        last_edit_ts = now_utc_iso()
        row = {
            'ID': seq_id,
            'Source Index': source_index,
            'Name': fields.get('Name', ''),
            'Summary': fields.get('Summary', ''),
            'First Launch Date': fields.get('FirstLaunchDate', ''),
            'Category': fields.get('Category', ''),
            'igdb_id': igdb_id_value,
            'Cover Path': cover_path,
            'Width': width,
            'Height': height,
            'last_edited_at': last_edit_ts,
        }

        for relation in lookups:
            id_column = relation.get('id_column')
            if id_column:
                row[id_column] = ''

        lookups_input = fields.get('Lookups') if isinstance(fields, Mapping) else {}

        with db_lock:
            conn = get_db()
            try:
                normalized_lookups: dict[str, dict[str, Any]] = {}
                for relation in lookups:
                    response_key = relation['response_key']
                    processed_column = relation['processed_column']
                    raw_value: Any = None
                    if isinstance(lookups_input, Mapping):
                        raw_value = lookups_input.get(response_key)
                        if raw_value is None:
                            raw_value = lookups_input.get(processed_column)
                    if raw_value is None:
                        raw_value = fields.get(response_key)
                        if raw_value is None:
                            raw_value = fields.get(processed_column)
                    selection = resolve_lookup_selection(conn, relation, raw_value)
                    normalized_lookups[response_key] = selection
                    row[processed_column] = lookup_display_text(selection['names'])
                    id_column = relation.get('id_column')
                    if id_column:
                        row[id_column] = encode_lookup_id_list(selection['ids'])

                if existing:
                    conn.execute(
                        '''UPDATE processed_games SET
                            "Name"=?, "Summary"=?, "First Launch Date"=?,
                            "Developers"=?, "developers_ids"=?,
                            "Publishers"=?, "publishers_ids"=?,
                            "Genres"=?, "genres_ids"=?,
                            "Game Modes"=?, "game_modes_ids"=?,
                            "Category"=?, "Platforms"=?, "platforms_ids"=?,
                            "igdb_id"=?, "Cover Path"=?, "Width"=?, "Height"=?,
                            last_edited_at=?
                           WHERE "ID"=?''',
                        (
                            row['Name'],
                            row['Summary'],
                            row['First Launch Date'],
                            row.get('Developers', ''),
                            row.get('developers_ids', ''),
                            row.get('Publishers', ''),
                            row.get('publishers_ids', ''),
                            row.get('Genres', ''),
                            row.get('genres_ids', ''),
                            row.get('Game Modes', ''),
                            row.get('game_modes_ids', ''),
                            row['Category'],
                            row.get('Platforms', ''),
                            row.get('platforms_ids', ''),
                            row['igdb_id'],
                            row['Cover Path'],
                            row['Width'],
                            row['Height'],
                            row['last_edited_at'],
                            seq_id,
                        ),
                    )
                else:
                    conn.execute(
                        '''INSERT INTO processed_games (
                            "ID", "Source Index", "Name", "Summary",
                            "First Launch Date", "Developers", "developers_ids",
                            "Publishers", "publishers_ids",
                            "Genres", "genres_ids", "Game Modes", "game_modes_ids",
                            "Category", "Platforms", "platforms_ids",
                            "igdb_id", "Cover Path", "Width", "Height", last_edited_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                        (
                            seq_id,
                            row['Source Index'],
                            row['Name'],
                            row['Summary'],
                            row['First Launch Date'],
                            row.get('Developers', ''),
                            row.get('developers_ids', ''),
                            row.get('Publishers', ''),
                            row.get('publishers_ids', ''),
                            row.get('Genres', ''),
                            row.get('genres_ids', ''),
                            row.get('Game Modes', ''),
                            row.get('game_modes_ids', ''),
                            row['Category'],
                            row.get('Platforms', ''),
                            row.get('platforms_ids', ''),
                            row['igdb_id'],
                            row['Cover Path'],
                            row['Width'],
                            row['Height'],
                            row['last_edited_at'],
                        ),
                    )

                persist_lookup_relations(conn, seq_id, normalized_lookups)
                conn.commit()
            except sqlite3.IntegrityError as exc:
                conn.rollback()
                raise ConflictError('conflict') from exc

        if new_record:
            navigator.seq_index += 1
        new_is_done = is_processed_game_done(row['Summary'], row['Cover Path'])
        if new_is_done and not was_processed_before:
            navigator.processed_total = min(
                navigator.total,
                navigator.processed_total + 1,
            )
        elif was_processed_before and not new_is_done:
            navigator.processed_total = max(
                0,
                navigator.processed_total - 1,
            )

        if upload_name:
            up_path = os.path.join(upload_dir, upload_name)
            if os.path.exists(up_path):
                os.remove(up_path)
        navigator.skip_queue = [s for s in navigator.skip_queue if s['index'] != index]
        navigator._save()
    return jsonify({'status': 'ok'})


@games_blueprint.route('/api/skip', methods=['POST'])
@handle_api_errors
def api_skip():
    data = request.get_json(force=True)
    try:
        index = int(data.get('index', 0))
    except (TypeError, ValueError) as exc:
        raise BadRequestError('invalid index') from exc
    upload_name = data.get('upload_name')
    upload_dir = _ctx('upload_dir')

    if upload_name:
        up_path = os.path.join(upload_dir, upload_name)
        if os.path.exists(up_path):
            os.remove(up_path)
    navigator = _get_navigator()
    try:
        navigator.skip(index)
    except IndexError as exc:
        raise BadRequestError('invalid index') from exc
    return jsonify({'status': 'ok'})


@games_blueprint.route('/api/set_index', methods=['POST'])
@handle_api_errors
def api_set_index():
    data = request.get_json(silent=True) or {}
    try:
        index = int(data.get('index', 0))
    except (TypeError, ValueError) as exc:
        raise BadRequestError('invalid index') from exc
    upload_name = data.get('upload_name')
    upload_dir = _ctx('upload_dir')
    if upload_name:
        up_path = os.path.join(upload_dir, upload_name)
        if os.path.exists(up_path):
            os.remove(up_path)
    navigator = _get_navigator()
    total_games = _get_total_games()
    if index < 0 or index >= total_games:
        raise BadRequestError('invalid index')
    build_game_payload = _ctx('build_game_payload')
    try:
        navigator.set_index(index)
    except IndexError as exc:
        raise BadRequestError('invalid index') from exc
    current_index = navigator.current()
    payload = build_game_payload(
        current_index,
        navigator.seq_index,
        navigator.processed_total + 1,
    )
    payload['completion'] = navigator.completion_percentage()
    return jsonify(payload)


@games_blueprint.route('/api/search', methods=['POST'])
@handle_api_errors
def api_search():
    payload = request.get_json(force=True) or {}
    raw_query = payload.get('query')
    query = str(raw_query).strip() if raw_query is not None else ''
    raw_category = payload.get('category')
    category_filters: list[str] = []
    if isinstance(raw_category, (list, tuple)):
        category_filters = [str(value).strip().lower() for value in raw_category if str(value).strip()]
    elif isinstance(raw_category, str):
        normalized = raw_category.strip().lower()
        if normalized:
            category_filters = [normalized]

    raw_genres = payload.get('genres')
    genres: list[str]
    if isinstance(raw_genres, (list, tuple)):
        genres = [str(value).strip().lower() for value in raw_genres if str(value).strip()]
    elif isinstance(raw_genres, str):
        cleaned = raw_genres.strip().lower()
        genres = [cleaned] if cleaned else []
    else:
        genres = []

    limit = payload.get('limit', 25)
    try:
        limit_value = int(limit)
    except (TypeError, ValueError):
        limit_value = 25
    limit_value = max(1, min(limit_value, 100))

    games_df = _get_games_df()
    if games_df is None:
        return jsonify({'results': [], 'matches': 0, 'limit': limit_value})

    try:
        total_rows = len(games_df)
    except Exception:
        total_rows = 0

    if total_rows <= 0:
        return jsonify({'results': [], 'matches': 0, 'limit': limit_value})

    extract_list = _ctx('extract_list')
    get_source_index_for_position = _ctx('get_source_index_for_position')

    results: list[dict[str, Any]] = []
    matches = 0
    query_lower = query.lower()

    for position in range(total_rows):
        try:
            row = games_df.iloc[position]
        except Exception:
            continue

        name_value = str(row.get('Name', '') or '')
        summary_value = str(row.get('Summary', '') or '')
        category_value = str(row.get('Category', '') or '')

        if category_filters:
            if category_value.strip().lower() not in category_filters:
                continue

        row_genres = extract_list(row, ['Genres', 'Genre'])
        if genres:
            genre_set = {str(g).strip().lower() for g in row_genres if str(g).strip()}
            if not set(genres).issubset(genre_set):
                continue

        if query_lower:
            name_lower = name_value.lower()
            summary_lower = summary_value.lower()
            matches_query = query_lower in name_lower or query_lower in summary_lower
            if not matches_query:
                candidates = []
                candidates.append(str(row.get('Source Index', '') or '').strip())
                candidates.append(str(row.get('IGDB ID', '') or '').strip())
                candidates.append(str(row.get('igdb_id', '') or '').strip())
                candidates.append(str(row.get('ID', '') or '').strip())
                candidates.append(str(row.get('id', '') or '').strip())
                for candidate in candidates:
                    if candidate and query_lower in candidate.lower():
                        matches_query = True
                        break
            if not matches_query:
                continue

        matches += 1
        if len(results) >= limit_value:
            continue

        try:
            source_index_value = get_source_index_for_position(position)
        except Exception:
            source_index_value = str(row.get('Source Index', position))

        igdb_id_value = row.get('IGDB ID') or row.get('igdb_id') or ''
        entry = {
            'index': int(position),
            'name': name_value,
            'category': category_value,
            'genres': row_genres,
            'source_index': source_index_value,
        }
        igdb_id_text = str(igdb_id_value).strip()
        if igdb_id_text:
            entry['igdb_id'] = igdb_id_text
        results.append(entry)

    source_indices = [entry['source_index'] for entry in results if entry.get('source_index')]
    if source_indices:
        unique_indices = list(dict.fromkeys(str(value) for value in source_indices))
        placeholders = ','.join('?' for _ in unique_indices)
        db_lock = _ctx('db_lock')
        get_db = _ctx('get_db')
        with db_lock:
            conn = get_db()
            cur = conn.execute(
                f'SELECT "Source Index", "ID" FROM processed_games WHERE "Source Index" IN ({placeholders})',
                tuple(unique_indices),
            )
            rows = cur.fetchall()
        processed_map = {str(row['Source Index']): row['ID'] for row in rows}
        for entry in results:
            source_index_text = str(entry.get('source_index'))
            if source_index_text in processed_map:
                entry['processed_id'] = processed_map[source_index_text]

    return jsonify({'results': results, 'matches': matches, 'limit': limit_value})


@games_blueprint.route('/api/game_by_id', methods=['POST'])
@handle_api_errors
def api_game_by_id():
    data = request.get_json(silent=True) or {}
    raw_id = data.get('id')
    if raw_id is None:
        raise BadRequestError('missing id')
    try:
        game_id = int(str(raw_id).strip())
    except (TypeError, ValueError) as exc:
        raise BadRequestError('invalid id') from exc

    upload_name = data.get('upload_name')
    upload_dir = _ctx('upload_dir')
    if upload_name:
        up_path = os.path.join(upload_dir, upload_name)
        if os.path.exists(up_path):
            os.remove(up_path)

    db_lock = _ctx('db_lock')
    get_db = _ctx('get_db')
    with db_lock:
        conn = get_db()
        cur = conn.execute(
            'SELECT "Source Index" FROM processed_games WHERE "ID"=?',
            (game_id,),
        )
        row = cur.fetchone()

    if row is None:
        raise NotFoundError('id not found')

    get_position_for_source_index = _ctx('get_position_for_source_index')
    index = get_position_for_source_index(row['Source Index'])
    if index is None:
        raise APIError(
            'invalid source index', payload={'id': game_id, 'source_index': row['Source Index']}
        )

    navigator = _get_navigator()
    build_game_payload = _ctx('build_game_payload')
    try:
        navigator.set_index(index)
    except IndexError as exc:
        raise NotFoundError('invalid index') from exc
    payload = build_game_payload(
        index,
        navigator.seq_index,
        navigator.processed_total + 1,
    )
    payload['completion'] = navigator.completion_percentage()
    return jsonify(payload)


@games_blueprint.route('/api/next', methods=['POST'])
@handle_api_errors
def api_next():
    data = request.get_json(silent=True) or {}
    upload_name = data.get('upload_name')
    upload_dir = _ctx('upload_dir')
    if upload_name:
        up_path = os.path.join(upload_dir, upload_name)
        if os.path.exists(up_path):
            os.remove(up_path)
    navigator = _get_navigator()
    build_game_payload = _ctx('build_game_payload')
    index = navigator.next()
    if index >= _get_total_games():
        return jsonify({
            'done': True,
            'message': 'Todos os jogos foram processados.',
            'completion': navigator.completion_percentage(),
        })
    payload = build_game_payload(
        index,
        navigator.seq_index,
        navigator.processed_total + 1,
    )
    payload['completion'] = navigator.completion_percentage()
    return jsonify(payload)


@games_blueprint.route('/api/back', methods=['POST'])
@handle_api_errors
def api_back():
    data = request.get_json(silent=True) or {}
    upload_name = data.get('upload_name')
    upload_dir = _ctx('upload_dir')
    if upload_name:
        up_path = os.path.join(upload_dir, upload_name)
        if os.path.exists(up_path):
            os.remove(up_path)
    navigator = _get_navigator()
    build_game_payload = _ctx('build_game_payload')
    index = navigator.back()
    payload = build_game_payload(
        index,
        navigator.seq_index,
        navigator.processed_total + 1,
    )
    payload['completion'] = navigator.completion_percentage()
    return jsonify(payload)
