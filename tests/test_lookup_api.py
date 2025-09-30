import json

import pytest

from tests.app_helpers import load_app


LOOKUP_TYPES = ['developers', 'publishers', 'genres', 'game_modes', 'platforms']


@pytest.fixture
def app_client(tmp_path):
    app_module = load_app(tmp_path)
    client = app_module.app.test_client()
    with client.session_transaction() as sess:
        sess['authenticated'] = True
    return app_module, client


def clear_processed_tables(app_module):
    tables = (
        'processed_game_developers',
        'processed_game_publishers',
        'processed_game_genres',
        'processed_game_game_modes',
        'processed_game_platforms',
        'processed_games',
        'updates_list',
    )
    with app_module.db_lock:
        with app_module.db:
            for table in tables:
                app_module.db.execute(f'DELETE FROM {table}')


def create_processed_game_with_entries(app_module, relation, entries, game_id):
    processed_column = relation['processed_column']
    id_column = relation.get('id_column')
    processed_sql = app_module._quote_identifier(processed_column)
    id_sql = app_module._quote_identifier(id_column) if id_column else None
    names = [name for _, name in entries if name]
    ids = [lookup_id for lookup_id, _ in entries]
    display_value = app_module._lookup_display_text(names)
    encoded_ids = app_module._encode_lookup_id_list(ids)
    selections = {
        rel['response_key']: {'ids': []}
        for rel in app_module.LOOKUP_RELATIONS
    }
    selections[relation['response_key']] = {'ids': ids}
    columns = ['"ID"', '"Source Index"', '"Name"', processed_sql]
    values = [game_id, str(game_id), f"Game {game_id}", display_value]
    if id_sql:
        columns.append(id_sql)
        values.append(encoded_ids)
    with app_module.db_lock:
        with app_module.db:
            placeholders = ', '.join('?' for _ in values)
            columns_sql = ', '.join(columns)
            app_module.db.execute(
                f'INSERT OR REPLACE INTO processed_games ({columns_sql}) '
                f'VALUES ({placeholders})',
                values,
            )
            app_module._persist_lookup_relations(app_module.db, game_id, selections)


@pytest.mark.parametrize('lookup_type', LOOKUP_TYPES)
def test_lookup_post_creates_entry(app_client, lookup_type):
    app_module, client = app_client
    with app_module.db_lock:
        with app_module.db:
            app_module.db.execute(f'DELETE FROM {lookup_type}')
    name = f'Example {lookup_type.replace("_", " ").title()}'
    response = client.post(f'/api/lookups/{lookup_type}', json={'name': name})
    assert response.status_code == 201
    data = response.get_json()
    assert data['item']['name'] == name
    with app_module.db_lock:
        row = app_module.db.execute(
            f'SELECT id, name FROM {lookup_type} WHERE id=?',
            (data['item']['id'],),
        ).fetchone()
    assert row is not None
    assert row['name'] == name


@pytest.mark.parametrize('lookup_type', LOOKUP_TYPES)
def test_lookup_post_reuses_existing_entry(app_client, lookup_type):
    app_module, client = app_client
    with app_module.db_lock:
        with app_module.db:
            app_module.db.execute(f'DELETE FROM {lookup_type}')
    name = f'Reused {lookup_type.replace("_", " ").title()}'
    first = client.post(f'/api/lookups/{lookup_type}', json={'name': name})
    assert first.status_code == 201
    first_data = first.get_json()
    second = client.post(f'/api/lookups/{lookup_type}', json={'name': name})
    assert second.status_code == 200
    second_data = second.get_json()
    assert second_data['item']['id'] == first_data['item']['id']


@pytest.mark.parametrize('lookup_type', LOOKUP_TYPES)
def test_lookup_post_validates_name(app_client, lookup_type):
    _, client = app_client
    response = client.post(f'/api/lookups/{lookup_type}', json={'name': '   '})
    assert response.status_code == 400
    invalid = client.post(f'/api/lookups/{lookup_type}', json=['invalid'])
    assert invalid.status_code == 400


@pytest.mark.parametrize('lookup_type', LOOKUP_TYPES)
def test_lookup_put_updates_name_and_processed_games(app_client, lookup_type):
    app_module, client = app_client
    relation = app_module.LOOKUP_RELATIONS_BY_TABLE[lookup_type]
    with app_module.db_lock:
        with app_module.db:
            app_module.db.execute(f'DELETE FROM {lookup_type}')
    clear_processed_tables(app_module)
    original = f'Original {lookup_type.replace("_", " ").title()}'
    created = client.post(f'/api/lookups/{lookup_type}', json={'name': original})
    assert created.status_code == 201
    created_item = created.get_json()['item']
    lookup_id = created_item['id']
    create_processed_game_with_entries(
        app_module,
        relation,
        [(lookup_id, created_item['name'])],
        game_id=42,
    )
    updated_name = f'Updated {lookup_type.replace("_", " ").title()}'
    response = client.put(
        f'/api/lookups/{lookup_type}',
        json={'id': lookup_id, 'name': updated_name},
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data['item']['name'] == updated_name
    processed_sql = app_module._quote_identifier(relation['processed_column'])
    id_sql = app_module._quote_identifier(relation['id_column'])
    join_table = relation['join_table']
    join_column = relation['join_column']
    with app_module.db_lock:
        lookup_row = app_module.db.execute(
            f'SELECT name FROM {lookup_type} WHERE id=?',
            (lookup_id,),
        ).fetchone()
        assert lookup_row['name'] == updated_name
        processed_row = app_module.db.execute(
            f'SELECT {processed_sql} AS value, {id_sql} AS ids '
            'FROM processed_games WHERE "ID"=?',
            (42,),
        ).fetchone()
        assert processed_row['value'] == updated_name
        assert json.loads(processed_row['ids']) == [lookup_id]
        join_rows = app_module.db.execute(
            f'SELECT {join_column} FROM {join_table} '
            'WHERE processed_game_id=? ORDER BY rowid',
            (42,),
        ).fetchall()
        assert [row[join_column] for row in join_rows] == [lookup_id]


@pytest.mark.parametrize('lookup_type', LOOKUP_TYPES)
def test_lookup_put_conflict_is_rejected(app_client, lookup_type):
    app_module, client = app_client
    with app_module.db_lock:
        with app_module.db:
            app_module.db.execute(f'DELETE FROM {lookup_type}')
    first = client.post(
        f'/api/lookups/{lookup_type}', json={'name': f'First {lookup_type}'}
    ).get_json()
    second = client.post(
        f'/api/lookups/{lookup_type}', json={'name': f'Second {lookup_type}'}
    ).get_json()
    conflict = client.put(
        f'/api/lookups/{lookup_type}',
        json={'id': first['item']['id'], 'name': second['item']['name']},
    )
    assert conflict.status_code == 409


@pytest.mark.parametrize('lookup_type', LOOKUP_TYPES)
def test_lookup_put_validates_payload(app_client, lookup_type):
    _, client = app_client
    missing = client.put(f'/api/lookups/{lookup_type}', json={'name': 'Name'})
    assert missing.status_code == 400
    invalid_id = client.put(
        f'/api/lookups/{lookup_type}', json={'id': 'abc', 'name': 'Test'}
    )
    assert invalid_id.status_code == 400
    invalid_payload = client.put(
        f'/api/lookups/{lookup_type}', json=['bad']
    )
    assert invalid_payload.status_code == 400


@pytest.mark.parametrize('lookup_type', LOOKUP_TYPES)
def test_lookup_put_unknown_id_returns_not_found(app_client, lookup_type):
    _, client = app_client
    response = client.put(
        f'/api/lookups/{lookup_type}',
        json={'id': 987654, 'name': 'Missing'},
    )
    assert response.status_code == 404


@pytest.mark.parametrize('lookup_type', LOOKUP_TYPES)
def test_lookup_delete_removes_entry_and_updates_games(app_client, lookup_type):
    app_module, client = app_client
    relation = app_module.LOOKUP_RELATIONS_BY_TABLE[lookup_type]
    with app_module.db_lock:
        with app_module.db:
            app_module.db.execute(f'DELETE FROM {lookup_type}')
    clear_processed_tables(app_module)
    primary = client.post(
        f'/api/lookups/{lookup_type}',
        json={'name': f'Primary {lookup_type}'},
    ).get_json()['item']
    secondary = client.post(
        f'/api/lookups/{lookup_type}',
        json={'name': f'Secondary {lookup_type}'},
    ).get_json()['item']
    create_processed_game_with_entries(
        app_module,
        relation,
        [
            (primary['id'], primary['name']),
            (secondary['id'], secondary['name']),
        ],
        game_id=77,
    )
    response = client.delete(
        f'/api/lookups/{lookup_type}', json={'id': primary['id']}
    )
    assert response.status_code == 200
    processed_sql = app_module._quote_identifier(relation['processed_column'])
    id_sql = app_module._quote_identifier(relation['id_column'])
    join_table = relation['join_table']
    join_column = relation['join_column']
    with app_module.db_lock:
        removed = app_module.db.execute(
            f'SELECT id FROM {lookup_type} WHERE id=?',
            (primary['id'],),
        ).fetchone()
        assert removed is None
        remaining = app_module.db.execute(
            f'SELECT name FROM {lookup_type} WHERE id=?',
            (secondary['id'],),
        ).fetchone()
        assert remaining is not None
        processed_row = app_module.db.execute(
            f'SELECT {processed_sql} AS value, {id_sql} AS ids '
            'FROM processed_games WHERE "ID"=?',
            (77,),
        ).fetchone()
        assert processed_row['value'] == secondary['name']
        assert json.loads(processed_row['ids']) == [secondary['id']]
        join_rows = app_module.db.execute(
            f'SELECT {join_column} FROM {join_table} '
            'WHERE processed_game_id=? ORDER BY rowid',
            (77,),
        ).fetchall()
        assert [row[join_column] for row in join_rows] == [secondary['id']]


@pytest.mark.parametrize('lookup_type', LOOKUP_TYPES)
def test_lookup_delete_validates_payload(app_client, lookup_type):
    _, client = app_client
    missing = client.delete(f'/api/lookups/{lookup_type}', json={})
    assert missing.status_code == 400
    invalid = client.delete(
        f'/api/lookups/{lookup_type}', json={'id': 'xyz'}
    )
    assert invalid.status_code == 400


@pytest.mark.parametrize('lookup_type', LOOKUP_TYPES)
def test_lookup_delete_unknown_id_returns_not_found(app_client, lookup_type):
    _, client = app_client
    response = client.delete(
        f'/api/lookups/{lookup_type}', json={'id': 987654}
    )
    assert response.status_code == 404


def test_lookup_invalid_type_returns_not_found(app_client):
    _, client = app_client
    response = client.post('/api/lookups/unknown', json={'name': 'Test'})
    assert response.status_code == 404
