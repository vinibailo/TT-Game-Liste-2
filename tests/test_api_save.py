import os
import uuid
import importlib.util
from pathlib import Path

import pandas as pd

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def load_app(tmp_path):
    os.chdir(tmp_path)
    module_name = f"app_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, APP_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    resp = client.post('/api/save', json={'index': 0, 'id': '1', 'fields': {}})
    assert resp.status_code == 200
    assert app.navigator.seq_index == 2
    with app.db_lock:
        cur = app.db.execute(
            'SELECT "ID", "igdb_id" FROM processed_games WHERE "Source Index"=?',
            ('0',),
        )
        row = cur.fetchone()
    assert row['ID'] == 1
    assert row['igdb_id'] == '4321'


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

