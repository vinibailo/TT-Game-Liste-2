"""Lookup-table API routes."""

from __future__ import annotations

from typing import Any, Mapping

from flask import Blueprint, jsonify, render_template, request

from routes.api_utils import (
    BadRequestError,
    ConflictError,
    MethodNotAllowedError,
    NotFoundError,
    handle_api_errors,
)

lookups_blueprint = Blueprint("lookups", __name__)

_context: dict[str, Any] = {}


def configure(context: Mapping[str, Any]) -> None:
    """Inject shared lookup helpers and metadata."""
    _context.update(context)


def _ctx(key: str) -> Any:
    if key not in _context:
        raise RuntimeError(f"lookup routes missing context value: {key}")
    return _context[key]


@lookups_blueprint.route('/lookups')
def lookups_page():
    lookup_tables_config = _ctx('LOOKUP_TABLES')
    format_lookup_label = _ctx('_format_lookup_label')
    lookup_tables = [
        {
            'type': table_config['table'],
            'label': table_config['table'].replace('_', ' ').title(),
            'singular_label': format_lookup_label(
                table_config.get('column', table_config['table'])
            ),
        }
        for table_config in lookup_tables_config
    ]
    default_lookup = lookup_tables[0]['type'] if lookup_tables else ''
    return render_template(
        'lookups.html',
        lookup_tables=lookup_tables,
        default_lookup=default_lookup,
        default_label=lookup_tables[0]['label'] if lookup_tables else '',
    )


@lookups_blueprint.route('/api/lookups/<lookup_type>', methods=['GET', 'POST', 'PUT', 'DELETE'])
@handle_api_errors
def api_lookup_options(lookup_type: str):
    lookup_endpoint_map = _ctx('LOOKUP_ENDPOINT_MAP')
    lookup_tables_by_name = _ctx('LOOKUP_TABLES_BY_NAME')
    table_relations = _ctx('LOOKUP_RELATIONS_BY_TABLE')
    normalized = lookup_type.strip().lower().replace('-', '_').replace(' ', '_')
    table_name = lookup_endpoint_map.get(normalized)
    if not table_name or table_name not in lookup_tables_by_name:
        raise NotFoundError('unknown lookup type')

    db_lock = _ctx('db_lock')
    get_db = _ctx('get_db')
    fetch_lookup_entries_for_game = _ctx('_fetch_lookup_entries_for_game')
    lookup_entries_to_selection = _ctx('_lookup_entries_to_selection')
    persist_lookup_relations = _ctx('_persist_lookup_relations')
    apply_lookup_entries_to_processed_game = _ctx('_apply_lookup_entries_to_processed_game')
    remove_lookup_id_from_entries = _ctx('_remove_lookup_id_from_entries')
    list_lookup_entries = _ctx('_list_lookup_entries')
    create_lookup_entry = _ctx('_create_lookup_entry')
    update_lookup_entry = _ctx('_update_lookup_entry')
    delete_lookup_entry = _ctx('_delete_lookup_entry')
    related_processed_game_ids = _ctx('_related_processed_game_ids')

    if request.method == 'GET':
        try:
            requested_limit = int(request.args.get('limit', 200))
        except (TypeError, ValueError):
            requested_limit = 200
        if requested_limit <= 0:
            requested_limit = 200
        limit = min(requested_limit, 200)

        offset = 0
        page_value: int | None = None
        page_param = request.args.get('page')
        if isinstance(page_param, str) and page_param.strip():
            try:
                page_value = int(page_param)
            except (TypeError, ValueError):
                page_value = 1
            if page_value < 1:
                page_value = 1
            offset = (page_value - 1) * limit
        else:
            offset_param = request.args.get('offset')
            if isinstance(offset_param, str) and offset_param.strip():
                try:
                    offset = int(offset_param)
                except (TypeError, ValueError):
                    offset = 0
                if offset < 0:
                    offset = 0

        with db_lock:
            conn = get_db()
            items, total = list_lookup_entries(
                conn, table_name, limit=limit, offset=offset
            )

        has_more = offset + len(items) < total
        payload: dict[str, Any] = {
            'items': items,
            'type': table_name,
            'total': total,
            'limit': limit,
            'offset': offset,
            'has_more': has_more,
        }
        if page_value is not None:
            payload['page'] = page_value
        return jsonify(payload)

    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, Mapping):
        raise BadRequestError('invalid payload')

    if request.method == 'POST':
        raw_name = payload.get('name')
        status: str
        item: dict[str, Any] | None
        created_item: dict[str, Any] | None
        status: str
        with db_lock:
            conn = get_db()
            status, created_item = create_lookup_entry(conn, table_name, raw_name)
            if status == 'created':
                conn.commit()
        if created_item is None or status == 'invalid':
            raise BadRequestError('invalid name')
        status_code = 201 if status == 'created' else 200
        return jsonify({'item': created_item, 'type': table_name}), status_code

    try:
        lookup_id = int(payload.get('id'))
    except (TypeError, ValueError) as exc:
        raise BadRequestError('invalid lookup id') from exc

    relation = table_relations.get(table_name)

    if request.method == 'PUT':
        new_name = payload.get('name')
        error_status: str | None = None
        updated_item: dict[str, Any] | None = None
        with db_lock:
            conn = get_db()
            status, updated_item = update_lookup_entry(conn, table_name, lookup_id, new_name)
            if status == 'invalid':
                error_status = 'invalid'
            elif status == 'not_found':
                error_status = 'not_found'
            elif status == 'conflict':
                error_status = 'conflict'
            elif updated_item is None:
                error_status = 'invalid'
            else:
                affected_game_ids = (
                    related_processed_game_ids(conn, relation, lookup_id)
                    if relation
                    else []
                )
                for game_id in affected_game_ids:
                    entries_map = {
                        key: list(value)
                        for key, value in fetch_lookup_entries_for_game(conn, game_id).items()
                    }
                    selections = lookup_entries_to_selection(entries_map)
                    persist_lookup_relations(conn, game_id, selections)
                    apply_lookup_entries_to_processed_game(conn, game_id, entries_map)
                conn.commit()
        if error_status == 'invalid':
            raise BadRequestError('invalid name')
        if error_status == 'not_found':
            raise NotFoundError('lookup not found')
        if error_status == 'conflict':
            raise ConflictError('name conflict')
        return jsonify({'item': updated_item, 'type': table_name})

    if request.method == 'DELETE':
        deleted_successfully = False
        with db_lock:
            conn = get_db()
            if relation:
                entries_by_game: dict[int, dict[str, list[dict[str, Any]]]] = {}
                for game_id in related_processed_game_ids(conn, relation, lookup_id):
                    entries_map = {
                        key: list(value)
                        for key, value in fetch_lookup_entries_for_game(conn, game_id).items()
                    }
                    remove_lookup_id_from_entries(entries_map, relation, lookup_id)
                    entries_by_game[game_id] = entries_map
                for game_id, entries_map in entries_by_game.items():
                    selections = lookup_entries_to_selection(entries_map)
                    persist_lookup_relations(conn, game_id, selections)
                    apply_lookup_entries_to_processed_game(conn, game_id, entries_map)
            deleted_successfully = delete_lookup_entry(conn, table_name, lookup_id)
            if deleted_successfully:
                conn.commit()
        if not deleted_successfully:
            raise NotFoundError('lookup not found')
        return jsonify({'status': 'deleted', 'type': table_name, 'id': lookup_id})

    raise MethodNotAllowedError('unsupported method')
