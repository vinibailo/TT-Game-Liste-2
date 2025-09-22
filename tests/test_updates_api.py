import json
import os
import uuid
import importlib.util
from pathlib import Path
from unittest.mock import patch

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def load_app(tmp_path):
    os.chdir(tmp_path)
    module_name = f"app_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, APP_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def authenticate(client):
    with client.session_transaction() as sess:
        sess['authenticated'] = True


def insert_processed_game(app_module, **overrides):
    defaults = {
        "ID": 1,
        "Source Index": "0",
        "Name": "Local Game",
        "Summary": "Local summary",
        "First Launch Date": "2020-01-01",
        "Developers": "Local Dev",
        "Publishers": "Local Pub",
        "Genres": "Action",
        "Game Modes": "Single-player",
        "Category": "Main",
        "Platforms": "PC",
        "igdb_id": "100",
        "Cover Path": "",
        "Width": 0,
        "Height": 0,
        "last_edited_at": "2024-01-01T00:00:00+00:00",
    }
    defaults.update(overrides)
    with app_module.db_lock:
        with app_module.db:
            app_module.db.execute(
                '''INSERT INTO processed_games (
                    "ID", "Source Index", "Name", "Summary",
                    "First Launch Date", "Developers", "Publishers",
                    "Genres", "Game Modes", "Category", "Platforms",
                    "igdb_id", "Cover Path", "Width", "Height", last_edited_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (
                    defaults["ID"],
                    defaults["Source Index"],
                    defaults["Name"],
                    defaults["Summary"],
                    defaults["First Launch Date"],
                    defaults["Developers"],
                    defaults["Publishers"],
                    defaults["Genres"],
                    defaults["Game Modes"],
                    defaults["Category"],
                    defaults["Platforms"],
                    defaults["igdb_id"],
                    defaults["Cover Path"],
                    defaults["Width"],
                    defaults["Height"],
                    defaults["last_edited_at"],
                ),
            )


def test_refresh_creates_update_records(tmp_path):
    app_module = load_app(tmp_path)
    insert_processed_game(app_module)
    client = app_module.app.test_client()
    authenticate(client)

    captured_ids = []

    def fake_exchange():
        return "token", "client"

    def fake_fetch(token, client_id, igdb_ids):
        captured_ids.extend(igdb_ids)
        return {
            "100": {
                "id": 100,
                "updated_at": 1_700_000_000,
                "summary": "Remote summary",
                "genres": ["Action", "Adventure"],
                "platforms": ["PC", "Switch"],
            }
        }

    app_module.exchange_twitch_credentials = fake_exchange
    app_module.fetch_igdb_metadata = fake_fetch

    refresh = client.post('/api/updates/refresh')
    assert refresh.status_code == 200
    data = refresh.get_json()
    assert data['status'] == 'ok'
    assert data['updated'] == 1
    assert data['missing'] == []
    assert captured_ids == ['100']

    listing = client.get('/api/updates')
    assert listing.status_code == 200
    listing_data = listing.get_json()
    assert listing_data['updates']
    entry = listing_data['updates'][0]
    assert entry['processed_game_id'] == 1
    assert entry['igdb_id'] == '100'
    assert entry['name'] == 'Local Game'
    assert entry['has_diff'] is True


def test_refresh_parses_involved_companies(tmp_path):
    app_module = load_app(tmp_path)
    insert_processed_game(
        app_module,
        Developers='Local Dev',
        Publishers='Local Pub',
    )
    client = app_module.app.test_client()
    authenticate(client)

    app_module.exchange_twitch_credentials = lambda: ("token", "client")

    response_payload = json.dumps(
        [
            {
                "id": 100,
                "updated_at": 1_700_000_000,
                "involved_companies": [
                    {"company": {"name": "Remote Dev Co"}, "developer": True},
                    {"company": {"name": "Remote Pub Co"}, "publisher": True},
                    {
                        "company": {"name": "Dual Role Co"},
                        "developer": True,
                        "publisher": True,
                    },
                    {"company": "String Company", "developer": True},
                ],
            }
        ]
    ).encode('utf-8')

    class DummyResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return response_payload

    def fake_urlopen(_request):
        return DummyResponse()

    with patch.object(app_module, 'urlopen', new=fake_urlopen):
        refresh = client.post('/api/updates/refresh')
    assert refresh.status_code == 200
    payload = refresh.get_json()
    assert payload['status'] == 'ok'
    assert payload['updated'] == 1

    detail = client.get('/api/updates/1')
    assert detail.status_code == 200
    detail_payload = detail.get_json()
    assert detail_payload['igdb_id'] == '100'
    assert detail_payload['igdb_payload']['developers'] == [
        'Remote Dev Co',
        'Dual Role Co',
        'String Company',
    ]
    assert detail_payload['igdb_payload']['publishers'] == [
        'Remote Pub Co',
        'Dual Role Co',
    ]
    diff = detail_payload['diff']
    assert sorted(diff['Developers']['added']) == [
        'Dual Role Co',
        'Remote Dev Co',
        'String Company',
    ]
    assert diff['Developers']['removed'] == ['Local Dev']
    assert sorted(diff['Publishers']['added']) == ['Dual Role Co', 'Remote Pub Co']
    assert diff['Publishers']['removed'] == ['Local Pub']


def test_refresh_surfaces_igdb_failure(tmp_path):
    app_module = load_app(tmp_path)
    insert_processed_game(app_module)
    client = app_module.app.test_client()
    authenticate(client)

    app_module.exchange_twitch_credentials = lambda: ("token", "client")

    def failing_fetch(*_args, **_kwargs):
        raise RuntimeError("IGDB request failed: 401 invalid credentials")

    app_module.fetch_igdb_metadata = failing_fetch

    response = client.post('/api/updates/refresh')
    assert response.status_code == 502
    payload = response.get_json()
    assert payload == {'error': 'IGDB request failed: 401 invalid credentials'}


def test_fetch_igdb_metadata_sets_user_agent(tmp_path):
    app_module = load_app(tmp_path)

    original_request = app_module.Request
    captured = {}

    def capturing_request(*args, **kwargs):
        request = original_request(*args, **kwargs)
        captured['request'] = request
        return request

    class DummyResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'[]'

    def fake_urlopen(request):
        captured['opened_request'] = request
        return DummyResponse()

    with patch.object(app_module, 'Request', new=capturing_request), patch.object(
        app_module, 'urlopen', new=fake_urlopen
    ):
        result = app_module.fetch_igdb_metadata('token', 'client', ['100'])

    assert result == {}
    assert captured['request'] is captured['opened_request']
    assert (
        captured['request'].get_header('User-agent')
        == app_module.IGDB_USER_AGENT
    )


def test_updates_detail_returns_diff(tmp_path):
    app_module = load_app(tmp_path)
    insert_processed_game(app_module)
    client = app_module.app.test_client()
    authenticate(client)

    app_module.exchange_twitch_credentials = lambda: ("token", "client")

    def fake_fetch(token, client_id, igdb_ids):
        return {
            "100": {
                "id": 100,
                "updated_at": 1_700_000_000,
                "summary": "Remote summary",
                "genres": ["Action", "Adventure"],
                "platforms": ["PC", "Switch"],
            }
        }

    app_module.fetch_igdb_metadata = fake_fetch

    refresh = client.post('/api/updates/refresh')
    assert refresh.status_code == 200

    detail = client.get('/api/updates/1')
    assert detail.status_code == 200
    payload = detail.get_json()
    assert payload['igdb_payload']['summary'] == 'Remote summary'
    assert payload['diff']['Summary']['added'] == 'Remote summary'
    assert 'Adventure' in payload['diff']['Genres']['added']
    assert payload['processed_game_id'] == 1
    assert payload['igdb_id'] == '100'


def test_updates_detail_missing_returns_404(tmp_path):
    app_module = load_app(tmp_path)
    client = app_module.app.test_client()
    authenticate(client)
    response = client.get('/api/updates/999')
    assert response.status_code == 404
    assert response.get_json()['error'] == 'not found'
