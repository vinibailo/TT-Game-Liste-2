import pandas as pd

import json
from pathlib import Path

from tests.app_helpers import load_app


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
            dev_id = app_module.db.execute(
                'INSERT INTO developers (name) VALUES (?)',
                ('Catalogued Dev',),
            ).lastrowid
            genre_id = app_module.db.execute(
                'INSERT INTO genres (name) VALUES (?)',
                ('Adventure',),
            ).lastrowid
            app_module.db.execute(
                'INSERT INTO processed_games ('
                '"ID", "Source Index", "Name", "Developers", "Genres"'
                ') VALUES (?, ?, ?, ?, ?)',
                (
                    1,
                    '0',
                    'Catalogued Game',
                    'Catalogued Dev',
                    'Adventure',
                ),
            )
            app_module.db.execute(
                'INSERT INTO processed_game_developers (processed_game_id, developer_id) '
                'VALUES (?, ?)',
                (1, dev_id),
            )
            app_module.db.execute(
                'INSERT INTO processed_game_genres (processed_game_id, genre_id) '
                'VALUES (?, ?)',
                (1, genre_id),
            )
            app_module.db.execute(
                'UPDATE processed_games SET developers_ids=?, genres_ids=? '
                'WHERE "ID"=?',
                (json.dumps([dev_id]), json.dumps([genre_id]), 1),
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
    assert lookups['Developers']['ids'] == [dev_id]
    assert lookups['Genres']['names'] == ['Adventure']
    assert lookups['Genres']['ids'] == [genre_id]
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


def _setup_igdb_prefill_app(app_module, igdb_id='123'):
    app_module.games_df = pd.DataFrame(
        [
            {
                "Source Index": "0",
                "Name": "",
                "Summary": "",
                "First Launch Date": "",
                "Developers": "",
                "Publishers": "",
                "Genres": "",
                "Game Modes": "",
                "Category": "",
                "Platforms": "",
                "Large Cover Image (URL)": "",
                "IGDB ID": igdb_id,
                "igdb_id": igdb_id,
            }
        ]
    )
    app_module.reset_source_index_cache()
    app_module.total_games = len(app_module.games_df)
    app_module.navigator.total = app_module.total_games
    app_module.navigator.current_index = 0
    app_module.navigator.seq_index = 1
    with app_module.db_lock:
        with app_module.db:
            app_module.db.execute('DELETE FROM processed_games')
            app_module.db.execute('DELETE FROM navigator_state')
            app_module.db.execute(
                'INSERT INTO processed_games ("ID", "Source Index", "Name", "Summary", "Cover Path") '
                'VALUES (?, ?, ?, ?, ?)',
                (1, '0', '', '', None),
            )


def _fake_prefill_metadata(image_id='abc123'):
    return {
        'id': 123,
        'name': 'IGDB Prefill',
        'summary': 'Prefilled summary',
        'first_release_date': 1_609_459_200,
        'category': 0,
        'developers': ['IGDB Studio'],
        'publishers': ['IGDB Publisher'],
        'genres': [{'name': 'Action'}],
        'platforms': [{'name': 'PC'}, {'name': 'Switch'}],
        'game_modes': [{'name': 'Single-player'}],
        'cover': {'image_id': image_id},
    }


def test_build_game_payload_prefills_from_igdb(tmp_path):
    app_module = load_app(tmp_path)
    _setup_igdb_prefill_app(app_module)

    app_module.exchange_twitch_credentials = lambda: ("token", "client")
    app_module.fetch_igdb_metadata = lambda *_args, **_kwargs: {'123': _fake_prefill_metadata()}
    captured_urls: list[str] = []

    def fake_cover_data_from_url(url: str) -> str:
        captured_urls.append(url)
        return f'cover::{url}'

    app_module.cover_data_from_url = fake_cover_data_from_url

    payload = app_module.build_game_payload(0, 1, 1)
    game = payload['game']

    assert game['Name'] == 'IGDB Prefill'
    assert game['Summary'] == 'Prefilled summary'
    assert game['FirstLaunchDate'] == '2021-01-01'
    assert game['Developers'] == 'IGDB Studio'
    assert game['Publishers'] == 'IGDB Publisher'
    assert game['Platforms'] == ['PC', 'Switch']
    assert game['Genres'] == ['Ação e Aventura']
    assert game['GameModes'] == ['Single-player']
    assert game['IGDBID'] == '123'
    assert payload['cover'] == 'cover::https://images.igdb.com/igdb/image/upload/t_original/abc123.jpg'
    assert captured_urls[-1] == 'https://images.igdb.com/igdb/image/upload/t_original/abc123.jpg'


def test_api_game_raw_prefills_from_igdb(tmp_path):
    app_module = load_app(tmp_path)
    _setup_igdb_prefill_app(app_module)

    app_module.exchange_twitch_credentials = lambda: ("token", "client")
    app_module.fetch_igdb_metadata = lambda *_args, **_kwargs: {'123': _fake_prefill_metadata('xyz789')}
    captured_urls: list[str] = []

    def fake_cover_data_from_url(url: str) -> str:
        captured_urls.append(url)
        return f'cover::{url}'

    app_module.cover_data_from_url = fake_cover_data_from_url

    client = app_module.app.test_client()
    authenticate(client)
    response = client.get('/api/game/0/raw')

    assert response.status_code == 200
    data = response.get_json()
    game = data['game']

    assert game['Name'] == 'IGDB Prefill'
    assert game['Summary'] == 'Prefilled summary'
    assert game['Platforms'] == ['PC', 'Switch']
    assert game['Genres'] == ['Ação e Aventura']
    assert game['GameModes'] == ['Single-player']
    assert game['IGDBID'] == '123'
    assert data['cover'] == 'cover::https://images.igdb.com/igdb/image/upload/t_original/xyz789.jpg'
    assert captured_urls[-1] == 'https://images.igdb.com/igdb/image/upload/t_original/xyz789.jpg'

