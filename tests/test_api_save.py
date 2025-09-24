import pandas as pd

import json

from tests.app_helpers import load_app


def test_api_save_index_conflict(tmp_path):
    app = load_app(tmp_path)
    client = app.app.test_client()
    with client.session_transaction() as sess:
        sess['authenticated'] = True
    app.navigator.current_index = 0
    resp = client.post('/api/save', json={'index': 1, 'id': '1', 'fields': {}})
    assert resp.status_code == 409
    data = resp.get_json()
    assert data['error'] == 'index mismatch'


def test_api_save_id_conflict(tmp_path):
    app = load_app(tmp_path)
    with app.db_lock:
        with app.db:
            app.db.execute(
                'INSERT INTO processed_games ("ID", "Source Index") VALUES (?, ?)',
                (1, '0'),
            )
    app.navigator.current_index = 0
    client = app.app.test_client()
    with client.session_transaction() as sess:
        sess['authenticated'] = True
    resp = client.post('/api/save', json={'index': 0, 'id': '2', 'fields': {}})
    assert resp.status_code == 409
    data = resp.get_json()
    assert data['error'] == 'id mismatch'


def test_api_save_seq_mismatch(tmp_path):
    app = load_app(tmp_path)
    client = app.app.test_client()
    with client.session_transaction() as sess:
        sess['authenticated'] = True
    app.navigator.current_index = 0
    app.navigator.seq_index = 1
    resp = client.post('/api/save', json={'index': 0, 'id': '2', 'fields': {}})
    assert resp.status_code == 409
    data = resp.get_json()
    assert data['error'] == 'id mismatch'


def test_api_save_success_increments_seq(tmp_path):
    app = load_app(tmp_path)
    app.games_df = pd.DataFrame([
        {
            'Name': 'Test Game',
            'id': 4321,
        }
    ])
    app.total_games = len(app.games_df)
    app.navigator.total = app.total_games
    client = app.app.test_client()
    with client.session_transaction() as sess:
        sess['authenticated'] = True
    app.navigator.current_index = 0
    app.navigator.seq_index = 1
    fields_payload = {
        'Name': 'Test Game',
        'Lookups': {
            'Developers': [{'name': 'Dev Studio'}],
            'Publishers': [{'name': 'Pub Works'}],
            'Genres': [{'name': 'Action'}],
            'GameModes': [{'name': 'Single-player'}],
            'Platforms': [{'name': 'PC'}],
        },
    }
    resp = client.post(
        '/api/save',
        json={'index': 0, 'id': '1', 'fields': fields_payload},
    )
    assert resp.status_code == 200
    assert app.navigator.seq_index == 2
    with app.db_lock:
        cur = app.db.execute(
            'SELECT * FROM processed_games WHERE "Source Index"=?',
            ('0',),
        )
        row = cur.fetchone()
    assert row['ID'] == 1
    assert row['igdb_id'] == '4321'
    assert row['Developers'] == 'Dev Studio'
    assert row['Genres'] == 'Action'
    for relation in app.LOOKUP_RELATIONS:
        id_column = relation['id_column']
        join_table = relation['join_table']
        join_column = relation['join_column']
        stored_ids = json.loads(row[id_column]) if row[id_column] else []
        join_ids = [
            joined_row[0]
            for joined_row in app.db.execute(
                f'SELECT {join_column} FROM {join_table} '
                'WHERE processed_game_id=? ORDER BY rowid',
                (row['ID'],),
            ).fetchall()
        ]
        assert stored_ids == join_ids
    with app.db_lock:
        developer_row = app.db.execute(
            'SELECT d.name FROM processed_game_developers pgd '
            'JOIN developers d ON d.id = pgd.developer_id '
            'WHERE pgd.processed_game_id=?',
            (row['ID'],),
        ).fetchone()
        assert developer_row['name'] == 'Dev Studio'
        genre_row = app.db.execute(
            'SELECT g.name FROM processed_game_genres pgg '
            'JOIN genres g ON g.id = pgg.genre_id '
            'WHERE pgg.processed_game_id=?',
            (row['ID'],),
        ).fetchone()
        assert genre_row['name'] == 'Action'


def test_api_save_conflict_does_not_increment_seq(tmp_path):
    app = load_app(tmp_path)
    with app.db_lock:
        with app.db:
            app.db.execute(
                'INSERT INTO processed_games ("ID", "Source Index") VALUES (?, ?)',
                (2, '5'),
            )
    client = app.app.test_client()
    with client.session_transaction() as sess:
        sess['authenticated'] = True
    app.navigator.current_index = 0
    app.navigator.seq_index = 2
    resp = client.post('/api/save', json={'index': 0, 'id': '2', 'fields': {}})
    assert resp.status_code == 409
    data = resp.get_json()
    assert data['error'] == 'conflict'
    assert app.navigator.seq_index == 2


def test_api_save_existing_id_new_index_preserves_record(tmp_path):
    app = load_app(tmp_path)
    with app.db_lock:
        with app.db:
            app.db.execute(
                'INSERT INTO processed_games ("ID", "Source Index", "Name") VALUES (?, ?, ?)',
                (1, '0', 'Original'),
            )
    client = app.app.test_client()
    with client.session_transaction() as sess:
        sess['authenticated'] = True
    app.navigator.current_index = 1
    app.navigator.seq_index = 1
    resp = client.post(
        '/api/save',
        json={'index': 1, 'id': '1', 'fields': {'Name': 'Updated'}},
    )
    assert resp.status_code == 409
    data = resp.get_json()
    assert data['error'] == 'conflict'
    assert app.navigator.seq_index == 1
    with app.db_lock:
        cur = app.db.execute(
            'SELECT "Source Index", "Name" FROM processed_games WHERE "ID"=?',
            (1,),
        )
        row = cur.fetchone()
    assert row['Source Index'] == '0'
    assert row['Name'] == 'Original'

