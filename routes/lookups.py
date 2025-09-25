"""Lookup-table API routes."""

from __future__ import annotations

from typing import Any, Mapping

from flask import Blueprint, jsonify, render_template, request

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
def api_lookup_options(lookup_type: str):
    lookup_endpoint_map = _ctx('LOOKUP_ENDPOINT_MAP')
    lookup_tables_by_name = _ctx('LOOKUP_TABLES_BY_NAME')
    table_relations = _ctx('LOOKUP_RELATIONS_BY_TABLE')
    normalized = lookup_type.strip().lower().replace('-', '_').replace(' ', '_')
    table_name = lookup_endpoint_map.get(normalized)
    if not table_name or table_name not in lookup_tables_by_name:
        return jsonify({'error': 'unknown lookup type'}), 404

    db_lock = _ctx('db_lock')
    get_db = _ctx('get_db')
    row_value = _ctx('_row_value')
    normalize_lookup_name = _ctx('_normalize_lookup_name')
    get_or_create_lookup_id = _ctx('_get_or_create_lookup_id')
    lookup_name_for_id = _ctx('_lookup_name_for_id')
    fetch_lookup_entries_for_game = _ctx('_fetch_lookup_entries_for_game')
    lookup_entries_to_selection = _ctx('_lookup_entries_to_selection')
    persist_lookup_relations = _ctx('_persist_lookup_relations')
    apply_lookup_entries_to_processed_game = _ctx('_apply_lookup_entries_to_processed_game')
    remove_lookup_id_from_entries = _ctx('_remove_lookup_id_from_entries')

    if request.method == 'GET':
        with db_lock:
            conn = get_db()
            cur = conn.execute(
                f'SELECT id, name FROM {table_name} ORDER BY name COLLATE NOCASE'
            )
            rows = cur.fetchall()
        items: list[dict[str, Any]] = []
        seen_ids: set[int] = set()
        for row in rows:
            raw_id = row_value(row, 'id', 0)
            try:
                coerced_id = int(raw_id)
            except (TypeError, ValueError):
                coerced_id = None
            if coerced_id is None or coerced_id in seen_ids:
                continue
            name = normalize_lookup_name(row_value(row, 'name', 1))
            if not name:
                continue
            seen_ids.add(coerced_id)
            items.append({'id': coerced_id, 'name': name})
        return jsonify({'items': items, 'type': table_name})

    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, Mapping):
        return jsonify({'error': 'invalid payload'}), 400

    if request.method == 'POST':
        name = normalize_lookup_name(payload.get('name'))
        if not name:
            return jsonify({'error': 'invalid name'}), 400
        with db_lock:
            conn = get_db()
            existing = conn.execute(
                f'SELECT id FROM {table_name} WHERE name = ? COLLATE NOCASE',
                (name,),
            ).fetchone()
            lookup_id = get_or_create_lookup_id(conn, table_name, name)
            final_name = lookup_name_for_id(conn, table_name, lookup_id) or name
            status_code = 201 if existing is None else 200
            if existing is None:
                conn.commit()
        return (
            jsonify({'item': {'id': lookup_id, 'name': final_name}, 'type': table_name}),
            status_code,
        )

    try:
        lookup_id = int(payload.get('id'))
    except (TypeError, ValueError):
        return jsonify({'error': 'invalid lookup id'}), 400

    relation = table_relations.get(table_name)

    if request.method == 'PUT':
        new_name = normalize_lookup_name(payload.get('name'))
        if not new_name:
            return jsonify({'error': 'invalid name'}), 400
        with db_lock:
            conn = get_db()
            row = conn.execute(
                f'SELECT id, name FROM {table_name} WHERE id = ?', (lookup_id,)
            ).fetchone()
            if row is None:
                return jsonify({'error': 'lookup not found'}), 404
            existing_name = normalize_lookup_name(row_value(row, 'name', 1))
            if existing_name != new_name:
                conflict = conn.execute(
                    f'SELECT id FROM {table_name} WHERE name = ? COLLATE NOCASE AND id != ?',
                    (new_name, lookup_id),
                ).fetchone()
                if conflict is not None:
                    return jsonify({'error': 'name conflict'}), 409
                conn.execute(
                    f'UPDATE {table_name} SET name = ? WHERE id = ?',
                    (new_name, lookup_id),
                )
            affected_game_ids: list[int] = []
            if relation:
                join_table = relation['join_table']
                join_column = relation['join_column']
                cur_games = conn.execute(
                    f'SELECT DISTINCT processed_game_id FROM {join_table} '
                    f'WHERE {join_column} = ?',
                    (lookup_id,),
                )
                for game_row in cur_games.fetchall():
                    try:
                        game_id = int(row_value(game_row, 'processed_game_id', 0))
                    except (TypeError, ValueError):
                        continue
                    affected_game_ids.append(game_id)
            for game_id in affected_game_ids:
                entries_map = {
                    key: list(value)
                    for key, value in fetch_lookup_entries_for_game(conn, game_id).items()
                }
                selections = lookup_entries_to_selection(entries_map)
                persist_lookup_relations(conn, game_id, selections)
                apply_lookup_entries_to_processed_game(conn, game_id, entries_map)
            updated_name = lookup_name_for_id(conn, table_name, lookup_id) or new_name
            conn.commit()
        return jsonify({'item': {'id': lookup_id, 'name': updated_name}, 'type': table_name})

    if request.method == 'DELETE':
        with db_lock:
            conn = get_db()
            row = conn.execute(
                f'SELECT id FROM {table_name} WHERE id = ?', (lookup_id,)
            ).fetchone()
            if row is None:
                return jsonify({'error': 'lookup not found'}), 404
            if relation:
                join_table = relation['join_table']
                join_column = relation['join_column']
                cur_games = conn.execute(
                    f'SELECT DISTINCT processed_game_id FROM {join_table} '
                    f'WHERE {join_column} = ?',
                    (lookup_id,),
                )
                entries_by_game: dict[int, dict[str, list[dict[str, Any]]]] = {}
                for game_row in cur_games.fetchall():
                    try:
                        game_id = int(row_value(game_row, 'processed_game_id', 0))
                    except (TypeError, ValueError):
                        continue
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
            conn.execute(f'DELETE FROM {table_name} WHERE id = ?', (lookup_id,))
            conn.commit()
        return jsonify({'status': 'deleted', 'type': table_name, 'id': lookup_id})

    return jsonify({'error': 'unsupported method'}), 405
