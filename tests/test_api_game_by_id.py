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


def seed_games(app_module):
    app_module.games_df = pd.DataFrame(
        [
            {
                "Name": "Catalogued Game",
                "Summary": "",
                "First Launch Date": "",
                "Developers": "",
                "Publishers": "",
                "Genres": "",
                "Game Modes": "",
                "Category": "",
                "Platforms": "",
                "Large Cover Image (URL)": "",
                "id": 987654321,
            }
        ]
    )
    app_module.total_games = len(app_module.games_df)
    app_module.navigator.total = app_module.total_games


def authenticate(client):
    with client.session_transaction() as sess:
        sess['authenticated'] = True


def test_game_by_id_returns_payload(tmp_path):
    app_module = load_app(tmp_path)
    seed_games(app_module)
    upload_dir = Path(tmp_path) / app_module.UPLOAD_DIR
    upload_dir.mkdir(parents=True, exist_ok=True)
    temp_name = "temp_upload.jpg"
    (upload_dir / temp_name).write_text("temp")
    with app_module.db_lock:
        with app_module.db:
            app_module.db.execute(
                'INSERT INTO processed_games ("ID", "Source Index", "Name") VALUES (?, ?, ?)',
                (1, '0', 'Catalogued Game'),
            )
            dev_id = app_module.db.execute(
                'INSERT INTO developers (name) VALUES (?)',
                ('Catalogued Dev',),
            ).lastrowid
            genre_id = app_module.db.execute(
                'INSERT INTO genres (name) VALUES (?)',
                ('Adventure',),
            ).lastrowid
            app_module.db.execute(
                'INSERT INTO processed_game_developers (processed_game_id, developer_id) VALUES (?, ?)',
                (1, dev_id),
            )
            app_module.db.execute(
                'INSERT INTO processed_game_genres (processed_game_id, genre_id) VALUES (?, ?)',
                (1, genre_id),
            )
    app_module.navigator.current_index = 1
    client = app_module.app.test_client()
    authenticate(client)
    response = client.post(
        '/api/game_by_id',
        json={'id': 1, 'upload_name': temp_name},
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data['index'] == 0
    assert data['id'] == 1
    assert data['game']['Name'] == 'Catalogued Game'
    assert data['game']['IGDBID'] == '987654321'
    assert data['game']['Developers'] == 'Catalogued Dev'
    lookups = data['game']['Lookups']
    assert lookups['Developers']['selected'][0]['name'] == 'Catalogued Dev'
    assert lookups['Genres']['names'] == ['Adventure']
    assert app_module.navigator.current_index == 0
    assert not (upload_dir / temp_name).exists()


def test_game_by_id_not_found(tmp_path):
    app_module = load_app(tmp_path)
    seed_games(app_module)
    client = app_module.app.test_client()
    authenticate(client)
    response = client.post('/api/game_by_id', json={'id': 999})
    assert response.status_code == 404
    assert response.get_json()['error'] == 'id not found'


def test_game_by_id_invalid_input(tmp_path):
    app_module = load_app(tmp_path)
    client = app_module.app.test_client()
    authenticate(client)
    response = client.post('/api/game_by_id', json={'id': 'abc'})
    assert response.status_code == 400
    assert response.get_json()['error'] == 'invalid id'

