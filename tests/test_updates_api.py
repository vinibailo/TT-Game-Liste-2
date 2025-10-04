from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import pandas as pd
from PIL import Image

from tests.app_helpers import load_app, set_games_dataframe


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
            lookup_selections: dict[str, dict[str, Any]] = {}
            for relation in app_module.LOOKUP_RELATIONS:
                processed_column = relation['processed_column']
                lookup_table = relation['lookup_table']
                values = app_module._parse_iterable(defaults.get(processed_column, ''))
                collected: list[int] = []
                for value in values:
                    normalized = app_module._normalize_lookup_name(value)
                    if not normalized:
                        continue
                    lookup_id = app_module._get_or_create_lookup_id(
                        app_module.db, lookup_table, normalized
                    )
                    if lookup_id is None or lookup_id in collected:
                        continue
                    collected.append(lookup_id)
                lookup_selections[relation['response_key']] = {
                    'ids': list(collected),
                    'names': values,
                }
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
            app_module._persist_lookup_relations(
                app_module.db, defaults['ID'], lookup_selections
            )


def insert_igdb_cache_entry(app_module, igdb_id: int | str, **overrides):
    defaults = {
        'name': 'Remote Game',
        'summary': 'Remote summary',
        'updated_at': 1_700_000_000,
        'first_release_date': 1_600_000_000,
        'category': 0,
        'cover_image_id': 'cover123',
        'rating_count': 0,
        'developers': [],
        'publishers': [],
        'genres': [],
        'platforms': [],
        'game_modes': [],
    }
    defaults.update(overrides)

    def _ensure_json(value):
        if isinstance(value, str):
            return value
        if isinstance(value, (list, tuple)):
            return json.dumps(list(value))
        return json.dumps([])

    with app_module.db_lock:
        with app_module.db:
            app_module.db.execute(
                f'''
                INSERT INTO {app_module.IGDB_CACHE_TABLE} (
                    igdb_id, name, summary, updated_at, first_release_date,
                    category, cover_image_id, rating_count, developers,
                    publishers, genres, platforms, game_modes, cached_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(igdb_id) DO UPDATE SET
                    name=excluded.name,
                    summary=excluded.summary,
                    updated_at=excluded.updated_at,
                    first_release_date=excluded.first_release_date,
                    category=excluded.category,
                    cover_image_id=excluded.cover_image_id,
                    rating_count=excluded.rating_count,
                    developers=excluded.developers,
                    publishers=excluded.publishers,
                    genres=excluded.genres,
                    platforms=excluded.platforms,
                    game_modes=excluded.game_modes,
                    cached_at=excluded.cached_at
                ''',
                (
                    int(igdb_id),
                    defaults['name'],
                    defaults['summary'],
                    defaults['updated_at'],
                    defaults['first_release_date'],
                    defaults['category'],
                    defaults['cover_image_id'],
                    defaults['rating_count'],
                    _ensure_json(defaults['developers']),
                    _ensure_json(defaults['publishers']),
                    _ensure_json(defaults['genres']),
                    _ensure_json(defaults['platforms']),
                    _ensure_json(defaults['game_modes']),
                    app_module.now_utc_iso(),
                ),
            )


def rebuild_updates_cache(app_module):
    if hasattr(app_module, 'rebuild_updates_list_cache'):
        app_module.rebuild_updates_list_cache()


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


class StubJobManager:
    def __init__(self, active_job=None, history=None):
        self._active_job = active_job
        self._history = list(history or [])

    def get_active_job(self, job_type):
        if job_type != 'refresh_updates':
            return None
        return self._active_job

    def list_jobs(self, job_type=None):
        if job_type and job_type != 'refresh_updates':
            return []
        return list(self._history)


def test_updates_refresh_returns_progress(tmp_path):
    app_module = load_app(tmp_path)
    client = app_module.app.test_client()
    authenticate(client)

    recorded: dict[str, Any] = {}

    def fake_refresh(access_token, client_id, offset, limit, **kwargs):
        recorded['access_token'] = access_token
        recorded['client_id'] = client_id
        recorded['offset'] = offset
        recorded['limit'] = limit
        recorded['conn'] = kwargs.get('conn')
        recorded['db_lock'] = kwargs.get('db_lock')
        return {
            'status': 'ok',
            'total': 500,
            'processed': offset + 3,
            'inserted': 2,
            'updated': 1,
            'unchanged': 0,
            'done': False,
            'next_offset': offset + 3,
        }

    app_module.routes_updates._context['exchange_twitch_credentials'] = (
        lambda: (lambda: ('token', 'client'))
    )

    with patch.object(app_module.routes_updates, 'refresh_igdb_cache', side_effect=fake_refresh):
        response = client.post('/api/updates/refresh?offset=7&limit=120')

    assert response.status_code == 200
    assert response.get_json() == {
        'total': 500,
        'processed': 10,
        'inserted': 2,
        'updated': 1,
        'unchanged': 0,
        'done': False,
        'next_offset': 10,
    }
    assert recorded['access_token'] == 'token'
    assert recorded['client_id'] == 'client'
    assert recorded['offset'] == 7
    assert recorded['limit'] == 120
    assert recorded['conn'] is not None
    assert recorded['db_lock'] is app_module.db_lock


def test_updates_refresh_defaults_query_params(tmp_path):
    app_module = load_app(tmp_path)
    client = app_module.app.test_client()
    authenticate(client)

    captured: dict[str, Any] = {}

    def fake_refresh(access_token, client_id, offset, limit, **_kwargs):
        captured['offset'] = offset
        captured['limit'] = limit
        return {
            'status': 'ok',
            'total': 0,
            'processed': 0,
            'inserted': 0,
            'updated': 0,
            'unchanged': 0,
            'done': True,
            'next_offset': 0,
        }

    app_module.routes_updates._context['exchange_twitch_credentials'] = (
        lambda: (lambda: ('token', 'client'))
    )

    with patch.object(app_module.routes_updates, 'refresh_igdb_cache', side_effect=fake_refresh):
        response = client.post('/api/updates/refresh')

    assert response.status_code == 200
    assert response.get_json() == {
        'total': 0,
        'processed': 0,
        'inserted': 0,
        'updated': 0,
        'unchanged': 0,
        'done': True,
        'next_offset': 0,
    }
    assert captured == {'offset': 0, 'limit': 200}


def test_updates_refresh_handles_runtime_error(tmp_path):
    app_module = load_app(tmp_path)
    client = app_module.app.test_client()
    authenticate(client)

    app_module.routes_updates._context['exchange_twitch_credentials'] = (
        lambda: (lambda: ('token', 'client'))
    )

    def fail_refresh(*_args, **_kwargs):
        raise RuntimeError('network unavailable')

    with patch.object(app_module.routes_updates, 'refresh_igdb_cache', side_effect=fail_refresh):
        response = client.post('/api/updates/refresh?offset=3&limit=10')

    assert response.status_code == 502
    assert response.get_json() == {'error': 'network unavailable'}


def test_updates_compare_runs_sync(tmp_path):
    app_module = load_app(tmp_path)
    client = app_module.app.test_client()
    authenticate(client)

    recorded: dict[str, Any] = {}

    def fake_compare_job(update_progress, **kwargs):
        recorded['called'] = True
        recorded['kwargs'] = kwargs
        update_progress(message='Comparing…', data={'phase': 'diffs'})
        return {'status': 'ok', 'message': 'Comparison complete.', 'toast_type': 'success'}

    app_module.routes_updates._context['compare_updates_job'] = fake_compare_job

    response = client.post('/api/updates/compare?sync=1')
    assert response.status_code == 200
    assert response.get_json() == {
        'status': 'ok',
        'message': 'Comparison complete.',
        'toast_type': 'success',
    }
    assert recorded['called'] is True
    assert recorded['kwargs']['igdb_ids'] == []


def test_updates_compare_filters_cache_ids(tmp_path):
    app_module = load_app(tmp_path)
    client = app_module.app.test_client()
    authenticate(client)

    insert_processed_game(app_module, ID=1, **{"Source Index": "0"}, igdb_id="100", Name="Game 100")
    insert_processed_game(app_module, ID=2, **{"Source Index": "1"}, igdb_id="abc", Name="Game ABC")
    insert_processed_game(app_module, ID=3, **{"Source Index": "2"}, igdb_id="200", Name="Game 200")

    insert_igdb_cache_entry(app_module, 100)
    insert_igdb_cache_entry(app_module, 200)
    insert_igdb_cache_entry(app_module, 300)

    captured: dict[str, Any] = {}

    def fake_compare_job(update_progress, **kwargs):
        captured['ids'] = kwargs.get('igdb_ids')
        update_progress(message='Comparing…', data={'phase': 'diffs'})
        return {'status': 'ok', 'message': 'Done', 'toast_type': 'success'}

    app_module.routes_updates._context['compare_updates_job'] = fake_compare_job

    response = client.post('/api/updates/compare?sync=1')
    assert response.status_code == 200
    assert sorted(captured['ids']) == [100, 200]


def test_refresh_creates_update_records(tmp_path):
    app_module = load_app(tmp_path)
    insert_processed_game(app_module)
    insert_igdb_cache_entry(
        app_module,
        100,
        summary='Remote summary',
        genres=['Action', 'Adventure'],
        platforms=['PC', 'Switch'],
    )
    client = app_module.app.test_client()
    authenticate(client)

    result = app_module._execute_refresh_job(lambda **_kwargs: None)
    assert result['status'] == 'ok'
    assert result['updated'] == 1
    assert result['missing'] == []

    rebuild_updates_cache(app_module)
    listing = client.get('/api/updates')
    assert listing.status_code == 200
    listing_data = listing.get_json()
    assert listing_data['total'] == 1
    assert listing_data['limit'] == 100
    assert listing_data['has_more'] is False
    assert 'next_cursor' not in listing_data
    assert listing_data['items']
    entry = listing_data['items'][0]
    assert entry['processed_game_id'] == 1
    assert entry['igdb_id'] == '100'
    assert entry['name'] == 'Local Game'
    assert entry['has_diff'] is True
    assert entry['update_type'] == 'mismatch'
    assert entry['detail_available'] is True
    assert entry['cover'] is None
    assert entry['cover_available'] is False
    assert entry['cover_url'] == '/static/no-image.jpg'


def test_updates_list_includes_duplicates(tmp_path):
    app_module = load_app(tmp_path)
    clear_processed_tables(app_module)

    set_games_dataframe(
        app_module,
        pd.DataFrame(
            [
                {'Source Index': '0', 'Name': 'Unique Game', 'id': 10},
                {'Source Index': '1', 'Name': 'Duplicated Game', 'id': 20},
                {'Source Index': '2', 'Name': 'Duplicated Game', 'id': 20},
            ]
        ),
        rebuild_metadata=False,
        rebuild_navigator=True,
    )
    app_module.catalog_state.set_navigator(
        app_module.GameNavigator(app_module.catalog_state.total_games)
    )

    insert_processed_game(
        app_module,
        ID=1,
        **{'Source Index': '0', 'Name': 'Unique Game', 'igdb_id': '10', 'Summary': 'Done', 'Cover Path': 'cover.jpg'},
    )
    insert_processed_game(
        app_module,
        ID=2,
        **{'Source Index': '1', 'Name': 'Duplicated Game', 'igdb_id': '20', 'Summary': 'Done', 'Cover Path': 'cover2.jpg'},
    )
    insert_processed_game(
        app_module,
        ID=3,
        **{'Source Index': '2', 'Name': 'Duplicated Game', 'igdb_id': '20', 'Summary': '', 'Cover Path': ''},
    )

    client = app_module.app.test_client()
    authenticate(client)

    rebuild_updates_cache(app_module)
    listing = client.get('/api/updates')
    assert listing.status_code == 200
    data = listing.get_json()
    assert data['total'] == 1
    assert data['items']
    duplicate_entry = data['items'][0]
    assert duplicate_entry['processed_game_id'] == 3
    assert duplicate_entry['update_type'] == 'duplicate'
    assert duplicate_entry['detail_available'] is False
    assert duplicate_entry['has_diff'] is False
    assert duplicate_entry['cover'] is None
    assert duplicate_entry['cover_available'] is False
    assert duplicate_entry['cover_url'] == '/static/no-image.jpg'


def test_updates_list_returns_processed_cover_url(tmp_path):
    app_module = load_app(tmp_path)
    clear_processed_tables(app_module)

    cover_path = tmp_path / 'cover.png'
    Image.new('RGB', (4, 4), color='blue').save(cover_path, format='PNG')

    insert_processed_game(
        app_module,
        ID=1,
        **{
            'Source Index': '0',
            'Name': 'Cover Game',
            'igdb_id': '10',
            'Summary': 'Done',
            'Cover Path': str(cover_path),
        },
    )

    _insert_igdb_update(
        app_module,
        1,
        '2024-05-01T00:00:00+00:00',
        has_diff=1,
    )

    client = app_module.app.test_client()
    authenticate(client)

    rebuild_updates_cache(app_module)
    listing = client.get('/api/updates')
    assert listing.status_code == 200
    payload = listing.get_json()
    assert payload['items']
    entry = payload['items'][0]
    assert entry['cover_available'] is True
    assert entry['cover_url'].startswith('data:image/jpeg;base64,')


def test_updates_list_respects_cursor_pagination(tmp_path):
    app_module = load_app(tmp_path)
    clear_processed_tables(app_module)

    for index in range(1, 4):
        insert_processed_game(
            app_module,
            ID=index,
            **{
                'Source Index': str(index - 1),
                'Name': f'Game {index}',
                'igdb_id': str(100 + index),
                'Summary': 'Summary',
            },
        )
        _insert_igdb_update(
            app_module,
            index,
            f'2024-05-0{index}T00:00:00+00:00',
            has_diff=1 if index % 2 else 0,
        )

    client = app_module.app.test_client()
    authenticate(client)

    rebuild_updates_cache(app_module)
    first_page = client.get('/api/updates?limit=1')
    assert first_page.status_code == 200
    payload = first_page.get_json()
    assert payload['total'] == 3
    assert payload['limit'] == 1
    assert len(payload['items']) == 1
    assert payload['has_more'] is True
    first_id = payload['items'][0]['processed_game_id']
    assert first_id in {1, 2, 3}
    assert payload['next_cursor']

    second_page = client.get(f"/api/updates?cursor={payload['next_cursor']}&limit=1")
    assert second_page.status_code == 200
    second_data = second_page.get_json()
    assert second_data['limit'] == 1
    assert len(second_data['items']) == 1
    assert second_data['items'][0]['processed_game_id'] != first_id
    assert second_data['has_more'] is True
    assert second_data['next_cursor']

    third_page = client.get(f"/api/updates?cursor={second_data['next_cursor']}&limit=1")
    assert third_page.status_code == 200
    third_data = third_page.get_json()
    assert len(third_data['items']) == 1
    remaining_id = third_data['items'][0]['processed_game_id']
    assert remaining_id not in {first_id, second_data['items'][0]['processed_game_id']}
    assert third_data['has_more'] is False


def test_updates_list_since_filters_new_entries(tmp_path):
    app_module = load_app(tmp_path)
    clear_processed_tables(app_module)

    insert_processed_game(
        app_module,
        ID=1,
        **{
            'Source Index': '0',
            'Name': 'First Game',
            'igdb_id': '100',
            'Summary': 'Summary',
        },
    )
    insert_processed_game(
        app_module,
        ID=2,
        **{
            'Source Index': '1',
            'Name': 'Second Game',
            'igdb_id': '101',
            'Summary': 'Summary',
        },
    )
    insert_processed_game(
        app_module,
        ID=3,
        **{
            'Source Index': '2',
            'Name': 'Third Game',
            'igdb_id': '102',
            'Summary': 'Summary',
        },
    )

    _insert_igdb_update(app_module, 1, '2024-05-01T00:00:00+00:00', has_diff=1)
    _insert_igdb_update(app_module, 2, '2024-05-02T00:00:00+00:00', has_diff=1)
    _insert_igdb_update(app_module, 3, '2024-05-03T12:34:56+00:00', has_diff=1)

    client = app_module.app.test_client()
    authenticate(client)

    rebuild_updates_cache(app_module)
    baseline = client.get('/api/updates')
    assert baseline.status_code == 200
    baseline_payload = baseline.get_json()
    assert len(baseline_payload['items']) == 3

    response = client.get('/api/updates?since=2024-05-02T00:00:00+00:00')
    assert response.status_code == 200
    payload = response.get_json()
    assert isinstance(payload['items'], list)
    assert len(payload['items']) == 1
    entry = payload['items'][0]
    assert entry['processed_game_id'] == 3
    assert entry['updated_at'] == '2024-05-03T12:34:56+00:00'
    assert payload['nextAfter'] is None


def test_updates_list_responds_with_etag_and_304(tmp_path):
    app_module = load_app(tmp_path)
    clear_processed_tables(app_module)

    insert_processed_game(
        app_module,
        ID=1,
        **{
            'Source Index': '0',
            'Name': 'Cached Game',
            'igdb_id': '200',
            'Summary': 'Summary',
        },
    )
    _insert_igdb_update(app_module, 1, '2024-05-05T10:00:00+00:00', has_diff=1)

    client = app_module.app.test_client()
    authenticate(client)

    rebuild_updates_cache(app_module)
    first = client.get('/api/updates')
    assert first.status_code == 200
    etag = first.headers.get('ETag')
    assert etag
    assert first.headers.get('Cache-Control') == 'max-age=30, stale-while-revalidate=120'
    payload = first.get_json()
    assert isinstance(payload, dict)
    assert payload['items']

    second = client.get('/api/updates', headers={'If-None-Match': etag})
    assert second.status_code == 304
    assert second.headers.get('ETag') == etag
    assert second.headers.get('Cache-Control') == 'max-age=30, stale-while-revalidate=120'


def test_cache_refresh_creates_entries(tmp_path):
    app_module = load_app(tmp_path)
    client = app_module.app.test_client()
    authenticate(client)

    app_module.exchange_twitch_credentials = lambda: ('token', 'client')

    def fake_count(token, client_id):
        assert token == 'token'
        assert client_id == 'client'
        return 3

    app_module.download_igdb_game_count = fake_count

    def fake_download(_token, _client_id, offset, limit):
        if offset > 0:
            return []
        return [
            {
                'id': 100,
                'name': 'Cache Game',
                'summary': 'Summary',
                'genres': ['Action'],
                'platforms': ['PC'],
                'game_modes': ['Single player'],
                'developers': ['Dev Studio'],
                'publishers': ['Pub Co'],
            },
            {
                'id': 200,
                'name': 'Second Game',
                'summary': '',
                'genres': [],
                'platforms': ['Switch'],
                'game_modes': [],
                'developers': [],
                'publishers': [],
            },
        ]

    app_module.download_igdb_games = fake_download

    response = client.post('/api/igdb/cache?sync=1', json={'offset': 0, 'limit': 5})
    assert response.status_code == 200
    payload = response.get_json()
    assert payload == {
        'status': 'ok',
        'total': 3,
        'processed': 2,
        'inserted': 2,
        'updated': 0,
        'unchanged': 0,
        'done': True,
        'next_offset': 2,
        'message': 'Cached 2 IGDB records.',
        'toast_type': 'success',
        'offset': 0,
        'limit': 5,
    }

    with app_module.db_lock:
        rows = app_module.db.execute(
            f'SELECT igdb_id, name FROM {app_module.IGDB_CACHE_TABLE} ORDER BY igdb_id'
        ).fetchall()

    assert [(row['igdb_id'], row['name']) for row in rows] == [
        (100, 'Cache Game'),
        (200, 'Second Game'),
    ]


def test_refresh_parses_involved_companies(tmp_path):
    app_module = load_app(tmp_path)
    insert_processed_game(
        app_module,
        Developers='Local Dev',
        Publishers='Local Pub',
        Genres='Ação e Aventura, Tiro',
        **{'Game Modes': 'Single-player, Cooperativo (Co-op), Competitivo (PvP), Multiplayer local, Multiplayer online'},
    )
    insert_igdb_cache_entry(
        app_module,
        100,
        updated_at=1_700_000_000,
        genres=['Action', 'Adventure', 'Shooter'],
        platforms=['PC', 'Switch'],
        game_modes=[
            'Single player',
            'Multiplayer',
            'Split screen',
            'Battle Royale',
            'Online co-op',
        ],
        developers=['Remote Dev Co', 'Dual Role Co', 'String Company'],
        publishers=['Remote Pub Co', 'Dual Role Co'],
    )
    client = app_module.app.test_client()
    authenticate(client)

    result = app_module._execute_refresh_job(lambda **_kwargs: None)
    assert result['status'] == 'ok'
    assert result['updated'] == 1

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
    assert detail_payload['igdb_payload']['genres'] == [
        'Action',
        'Adventure',
        'Shooter',
    ]
    assert detail_payload['igdb_payload']['platforms'] == [
        'PC',
        'Switch',
    ]
    assert detail_payload['igdb_payload']['game_modes'] == [
        'Single player',
        'Multiplayer',
        'Split screen',
        'Battle Royale',
        'Online co-op',
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
    assert 'Genres' not in diff
    assert diff['Platforms']['added'] == ['Switch']
    assert diff['Platforms']['removed'] == []
    assert 'Game Modes' not in diff


def test_compare_updates_job_recomputes_diffs(tmp_path):
    app_module = load_app(tmp_path)
    insert_processed_game(app_module)
    insert_igdb_cache_entry(app_module, 100, name='Cache Game', summary='Updated summary')
    client = app_module.app.test_client()
    authenticate(client)

    result = app_module._execute_compare_updates_job(lambda **_kwargs: None)
    assert result['status'] == 'ok'
    assert result['updated'] == 1

    detail = client.get('/api/updates/1')
    assert detail.status_code == 200
    payload = detail.get_json()
    assert payload['igdb_id'] == '100'
    assert payload['diff']


def test_build_igdb_diff_formats_first_release_date(tmp_path):
    app_module = load_app(tmp_path)

    processed_row = {'First Launch Date': '2020-01-01'}
    igdb_payload = {'first_release_date': 1623715200}

    diff = app_module.build_igdb_diff(processed_row, igdb_payload)

    launch_date_diff = diff['First Launch Date']
    assert launch_date_diff['added'] == '2021-06-15'
    assert launch_date_diff['removed'] == '2020-01-01'
    assert launch_date_diff['added'] != str(igdb_payload['first_release_date'])


def test_refresh_surfaces_igdb_failure(tmp_path):
    app_module = load_app(tmp_path)
    insert_processed_game(app_module)
    client = app_module.app.test_client()
    authenticate(client)

    result = app_module._execute_refresh_job(lambda **_kwargs: None)
    assert result['status'] == 'ok'
    assert result['updated'] == 0
    assert result['missing'] == [1]


def test_download_igdb_metadata_sets_user_agent(tmp_path):
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
        result = app_module.download_igdb_metadata('token', 'client', ['100'])

    assert result == {}
    assert captured['request'] is captured['opened_request']
    assert (
        captured['request'].get_header('User-agent')
        == app_module.IGDB_USER_AGENT
    )


def test_download_igdb_metadata_batches_requests(tmp_path):
    app_module = load_app(tmp_path)
    app_module.IGDB_BATCH_SIZE = 5

    igdb_ids = [str(i) for i in range(1, 13)]

    def build_payload(start, end):
        return [
            {
                "id": idx,
                "involved_companies": [
                    {"company": {"name": f"Dev {idx}"}, "developer": True},
                    {"company": {"name": f"Pub {idx}"}, "publisher": True},
                ],
            }
            for idx in range(start, end)
        ]

    responses = [
        json.dumps(build_payload(1, 6)).encode('utf-8'),
        json.dumps(build_payload(6, 11)).encode('utf-8'),
        json.dumps(build_payload(11, 13)).encode('utf-8'),
    ]

    queries: list[str] = []

    class DummyResponse:
        def __init__(self, body: bytes):
            self._body = body

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return self._body

    def fake_urlopen(request):
        queries.append(request.data.decode('utf-8'))
        index = len(queries) - 1
        try:
            body = responses[index]
        except IndexError as exc:  # pragma: no cover - guard against unexpected calls
            raise AssertionError('Too many IGDB requests') from exc
        return DummyResponse(body)

    with patch.object(app_module, 'urlopen', new=fake_urlopen):
        results = app_module.download_igdb_metadata('token', 'client', igdb_ids)

    assert len(queries) == 3
    assert 'limit 5;' in queries[0]
    assert 'limit 5;' in queries[1]
    assert 'limit 2;' in queries[2]
    assert 'where id = (1, 2, 3, 4, 5);' in queries[0]
    assert 'where id = (6, 7, 8, 9, 10);' in queries[1]
    assert 'where id = (11, 12);' in queries[2]

    expected_ids = [str(i) for i in range(1, 13)]
    assert sorted(results.keys(), key=int) == expected_ids
    assert results['1']['id'] == 1
    assert results['1']['developers'] == ['Dev 1']
    assert results['1']['publishers'] == ['Pub 1']
    assert results['12']['id'] == 12
    assert results['12']['developers'] == ['Dev 12']
    assert results['12']['publishers'] == ['Pub 12']


def test_updates_detail_returns_diff(tmp_path):
    app_module = load_app(tmp_path)
    insert_processed_game(app_module)
    insert_igdb_cache_entry(
        app_module,
        100,
        summary='Remote summary',
        genres=['Action', 'Adventure'],
        platforms=['PC', 'Switch'],
    )
    client = app_module.app.test_client()
    authenticate(client)

    result = app_module._execute_refresh_job(lambda **_kwargs: None)
    assert result['status'] == 'ok'

    detail = client.get('/api/updates/1')
    assert detail.status_code == 200
    payload = detail.get_json()
    assert payload['igdb_payload']['summary'] == 'Remote summary'
    assert 'Summary' not in payload['diff']
    assert payload['diff']['Genres']['added'] == ['Ação e Aventura']
    assert payload['diff']['Genres']['removed'] == ['Action']
    assert 'Category' not in payload['diff']
    assert payload['processed_game_id'] == 1
    assert payload['igdb_id'] == '100'
    assert payload['cover_available'] is False
    assert payload['cover_url'] == '/static/no-image.jpg'


def test_updates_cover_endpoint_returns_image(tmp_path):
    app_module = load_app(tmp_path)
    cover_path = tmp_path / 'cover.png'
    Image.new('RGB', (4, 4), color='red').save(cover_path, format='PNG')
    insert_processed_game(
        app_module,
        ID=1,
        **{'Cover Path': str(cover_path)},
    )

    client = app_module.app.test_client()
    authenticate(client)

    response = client.get('/api/updates/1/cover')
    assert response.status_code == 200
    payload = response.get_json()
    assert payload['cover'].startswith('data:image/jpeg;base64,')


def test_updates_cover_endpoint_returns_404_when_missing(tmp_path):
    app_module = load_app(tmp_path)
    insert_processed_game(app_module, ID=1, **{'Cover Path': ''})

    client = app_module.app.test_client()
    authenticate(client)

    response = client.get('/api/updates/1/cover')
    assert response.status_code == 404


def test_fix_names_updates_batches(tmp_path):
    app_module = load_app(tmp_path)
    clear_processed_tables(app_module)
    insert_processed_game(
        app_module,
        ID=1,
        **{'Source Index': '1', 'Name': 'Wrong Name', 'igdb_id': '101'},
    )
    insert_processed_game(
        app_module,
        ID=2,
        **{'Source Index': '2', 'Name': 'Correct 202', 'igdb_id': '202'},
    )
    insert_processed_game(
        app_module,
        ID=3,
        **{'Source Index': '3', 'Name': 'Missing Remote', 'igdb_id': '303'},
    )
    insert_igdb_cache_entry(app_module, 101, name='Correct Name')
    insert_igdb_cache_entry(app_module, 202, name='Correct 202')
    insert_igdb_cache_entry(app_module, 303, name='')

    with app_module.db_lock:
        before_rows = app_module.db.execute(
            'SELECT "ID", last_edited_at FROM processed_games ORDER BY "ID"'
        ).fetchall()

    client = app_module.app.test_client()
    authenticate(client)

    first = client.post('/api/updates/fix-names', json={'limit': 2})
    assert first.status_code == 200
    first_data = first.get_json()
    assert first_data['total'] == 3
    assert first_data['processed'] == 2
    assert first_data['updated'] == 1
    assert first_data['unchanged'] == 1
    assert first_data['missing'] == []
    assert first_data['done'] is False
    assert first_data['next_offset'] == 2

    with app_module.db_lock:
        rows = app_module.db.execute(
            'SELECT "ID", "Name", last_edited_at FROM processed_games ORDER BY "ID"'
        ).fetchall()

    assert [row['Name'] for row in rows] == [
        'Correct Name',
        'Correct 202',
        'Missing Remote',
    ]
    assert rows[0]['last_edited_at'] != before_rows[0]['last_edited_at']
    assert rows[1]['last_edited_at'] == before_rows[1]['last_edited_at']

    second = client.post(
        '/api/updates/fix-names',
        json={'offset': first_data['next_offset'], 'limit': 2},
    )
    assert second.status_code == 200
    second_data = second.get_json()
    assert second_data['processed'] == 3
    assert second_data['done'] is True
    assert second_data['updated'] == 0
    assert second_data['missing'] == ['303']
    assert second_data['missing_name'] == ['303']
    assert second_data['unchanged'] == 0

    with app_module.db_lock:
        final_rows = app_module.db.execute(
            'SELECT "Name" FROM processed_games ORDER BY "ID"'
        ).fetchall()
    assert [row['Name'] for row in final_rows] == [
        'Correct Name',
        'Correct 202',
        'Missing Remote',
    ]


def test_fix_names_without_ids_skips_api(tmp_path):
    app_module = load_app(tmp_path)
    clear_processed_tables(app_module)
    insert_processed_game(
        app_module,
        ID=1,
        **{'Source Index': '10', 'Name': 'No Remote', 'igdb_id': None},
    )

    client = app_module.app.test_client()
    authenticate(client)

    def fail_exchange():  # pragma: no cover - ensures endpoint short-circuits
        raise AssertionError('exchange_twitch_credentials should not be called')

    app_module.exchange_twitch_credentials = fail_exchange
    app_module.fetch_igdb_metadata = lambda *_args, **_kwargs: {}

    response = client.post('/api/updates/fix-names')
    assert response.status_code == 200
    payload = response.get_json()
    assert payload['total'] == 0
    assert payload['processed'] == 0
    assert payload['done'] is True
    assert payload['updated'] == 0
    assert payload['invalid'] == 0


def test_updates_detail_missing_returns_404(tmp_path):
    app_module = load_app(tmp_path)
    client = app_module.app.test_client()
    authenticate(client)
    response = client.get('/api/updates/999')
    assert response.status_code == 404
    assert response.get_json()['error'] == 'not found'


def test_remove_duplicates_merges_duplicate_entries(tmp_path):
    app_module = load_app(tmp_path)
    clear_processed_tables(app_module)

    set_games_dataframe(
        app_module,
        pd.DataFrame(
            [
                {'Source Index': '0', 'Name': 'Unique Game', 'id': 10},
                {'Source Index': '1', 'Name': 'Duplicated Game', 'id': 20},
                {'Source Index': '2', 'Name': 'Duplicated Game', 'id': 20},
                {'Source Index': '3', 'Name': 'Tail Game', 'id': 30},
            ]
        ),
        rebuild_metadata=False,
        rebuild_navigator=True,
    )
    app_module.catalog_state.set_navigator(
        app_module.GameNavigator(app_module.catalog_state.total_games)
    )

    insert_processed_game(
        app_module,
        ID=1,
        **{'Source Index': '0', 'Name': 'Unique Game', 'igdb_id': '10', 'Summary': '', 'Cover Path': ''},
    )
    insert_processed_game(
        app_module,
        ID=2,
        **{
            'Source Index': '1',
            'Name': 'Duplicated Game',
            'igdb_id': '20',
            'Summary': 'Primary summary',
            'Cover Path': f"{app_module.PROCESSED_DIR}/2.jpg",
            'Developers': 'Canonical Dev',
            'Platforms': 'Base Platform',
        },
    )
    insert_processed_game(
        app_module,
        ID=3,
        **{
            'Source Index': '2',
            'Name': 'Duplicated Game',
            'igdb_id': '20',
            'Summary': '',
            'Cover Path': '',
            'Publishers': 'Extra Pub',
            'Platforms': 'Extra Platform',
        },
    )
    insert_processed_game(
        app_module,
        ID=4,
        **{'Source Index': '3', 'Name': 'Tail Game', 'igdb_id': '30', 'Summary': '', 'Cover Path': ''},
    )

    with app_module.db_lock:
        with app_module.db:
            app_module.db.executemany(
                'INSERT INTO igdb_updates (processed_game_id, igdb_id) VALUES (?, ?)',
                [(2, '20'), (3, '20')],
            )

    client = app_module.app.test_client()
    authenticate(client)

    response = client.post('/api/updates/remove-duplicates')
    assert response.status_code == 200
    payload = response.get_json()
    assert payload['status'] == 'ok'
    assert payload['removed'] == 1
    assert payload['duplicate_groups'] == 1
    assert payload['skipped'] == 0
    assert payload['remaining'] == 3

    with app_module.db_lock:
        rows = app_module.db.execute(
            'SELECT "ID", "Source Index", "Name", "Developers", "Publishers", "Platforms" '
            'FROM processed_games ORDER BY CAST("ID" AS INTEGER)'
        ).fetchall()
        updates = app_module.db.execute(
            'SELECT processed_game_id FROM igdb_updates ORDER BY processed_game_id'
        ).fetchall()
        developer_rows = app_module.db.execute(
            'SELECT processed_game_id, developer_id FROM processed_game_developers ORDER BY processed_game_id, developer_id'
        ).fetchall()
        publisher_rows = app_module.db.execute(
            'SELECT processed_game_id, publisher_id FROM processed_game_publishers ORDER BY processed_game_id, publisher_id'
        ).fetchall()
        platform_rows = app_module.db.execute(
            'SELECT processed_game_id, platform_id FROM processed_game_platforms ORDER BY processed_game_id, platform_id'
        ).fetchall()

    assert len(rows) == 3
    assert [row['Name'] for row in rows] == ['Unique Game', 'Duplicated Game', 'Tail Game']
    assert [row['Source Index'] for row in rows] == ['0', '1', '2']
    duplicate_row = next(row for row in rows if row['Name'] == 'Duplicated Game')
    assert 'Canonical Dev' in duplicate_row['Developers']
    assert 'Local Dev' in duplicate_row['Developers']
    assert 'Extra Pub' in duplicate_row['Publishers']
    assert 'Local Pub' in duplicate_row['Publishers']
    assert duplicate_row['Platforms'] == 'Base Platform, Extra Platform'
    assert [row['processed_game_id'] for row in updates] == [duplicate_row['ID']]
    developer_ids = {row['processed_game_id'] for row in developer_rows}
    publisher_ids = {row['processed_game_id'] for row in publisher_rows}
    platform_ids = {row['processed_game_id'] for row in platform_rows}
    assert duplicate_row['ID'] in developer_ids
    assert duplicate_row['ID'] in publisher_ids
    assert duplicate_row['ID'] in platform_ids
    assert 3 not in developer_ids
    assert 3 not in publisher_ids
    assert 3 not in platform_ids

    df = app_module.catalog_state.games_df
    assert len(df) == 3
    assert list(df['Source Index']) == ['0', '1', '2']
    assert app_module.catalog_state.total_games == 3
    assert app_module._ensure_navigator_dataframe(rebuild_state=False).total == 3


def test_remove_duplicate_endpoint_deletes_entry(tmp_path):
    app_module = load_app(tmp_path)
    clear_processed_tables(app_module)

    set_games_dataframe(
        app_module,
        pd.DataFrame(
            [
                {'Source Index': '0', 'Name': 'Unique Game', 'id': 10},
                {'Source Index': '1', 'Name': 'Duplicated Game', 'id': 20},
                {'Source Index': '2', 'Name': 'Duplicated Game', 'id': 20},
                {'Source Index': '3', 'Name': 'Tail Game', 'id': 30},
            ]
        ),
        rebuild_metadata=False,
        rebuild_navigator=True,
    )
    app_module.catalog_state.set_navigator(
        app_module.GameNavigator(app_module.catalog_state.total_games)
    )

    insert_processed_game(
        app_module,
        ID=1,
        **{'Source Index': '0', 'Name': 'Unique Game', 'igdb_id': '10', 'Summary': '', 'Cover Path': ''},
    )
    insert_processed_game(
        app_module,
        ID=2,
        **{
            'Source Index': '1',
            'Name': 'Duplicated Game',
            'igdb_id': '20',
            'Summary': '',
            'Cover Path': f"{app_module.PROCESSED_DIR}/2.jpg",
            'Developers': 'Canonical Dev',
            'Platforms': 'Base Platform',
        },
    )
    insert_processed_game(
        app_module,
        ID=3,
        **{
            'Source Index': '2',
            'Name': 'Duplicated Game',
            'igdb_id': '20',
            'Summary': 'Better summary',
            'Cover Path': '',
            'Publishers': 'Extra Pub',
        },
    )
    insert_processed_game(
        app_module,
        ID=4,
        **{'Source Index': '3', 'Name': 'Tail Game', 'igdb_id': '30', 'Summary': '', 'Cover Path': ''},
    )

    client = app_module.app.test_client()
    authenticate(client)

    response = client.post('/api/updates/remove-duplicate/3')
    assert response.status_code == 200
    payload = response.get_json()
    assert payload['status'] == 'ok'
    assert payload['removed'] == 1
    assert payload['removed_id'] == 3
    assert payload['remaining'] == 3
    assert payload['toast_type'] == 'success'

    with app_module.db_lock:
        rows = app_module.db.execute(
            'SELECT "ID", "Source Index", "Name", "Summary", "Publishers" '
            'FROM processed_games ORDER BY CAST("ID" AS INTEGER)'
        ).fetchall()
        publisher_rows = app_module.db.execute(
            'SELECT processed_game_id, publisher_id FROM processed_game_publishers ORDER BY processed_game_id, publisher_id'
        ).fetchall()

    assert len(rows) == 3
    assert [row['Source Index'] for row in rows] == ['0', '1', '2']
    names = [row['Name'] for row in rows]
    assert names.count('Duplicated Game') == 1
    assert 'Tail Game' in names
    duplicate_row = next(row for row in rows if row['Name'] == 'Duplicated Game')
    assert duplicate_row['Summary'] == 'Better summary'
    assert 'Extra Pub' in duplicate_row['Publishers']
    publisher_ids = {row['processed_game_id'] for row in publisher_rows}
    assert duplicate_row['ID'] in publisher_ids
    assert 3 not in publisher_ids
    df = app_module.catalog_state.games_df
    assert len(df) == 3
    assert list(df['Source Index']) == ['0', '1', '2']


def test_remove_duplicates_handles_all_duplicates(tmp_path):
    app_module = load_app(tmp_path)
    clear_processed_tables(app_module)

    set_games_dataframe(
        app_module,
        pd.DataFrame(
            [
                {'Source Index': '0', 'Name': 'Duplicate One', 'id': 50},
                {'Source Index': '1', 'Name': 'Duplicate One', 'id': 50},
            ]
        ),
        rebuild_metadata=False,
        rebuild_navigator=True,
    )
    app_module.catalog_state.set_navigator(
        app_module.GameNavigator(app_module.catalog_state.total_games)
    )

    insert_processed_game(
        app_module,
        ID=1,
        **{'Source Index': '0', 'Name': 'Duplicate One', 'igdb_id': '50', 'Summary': '', 'Cover Path': ''},
    )
    insert_processed_game(
        app_module,
        ID=2,
        **{'Source Index': '1', 'Name': 'Duplicate One', 'igdb_id': '50', 'Summary': '', 'Cover Path': ''},
    )

    client = app_module.app.test_client()
    authenticate(client)

    response = client.post('/api/updates/remove-duplicates')
    assert response.status_code == 200
    payload = response.get_json()
    assert payload['status'] == 'ok'
    assert payload['removed'] == 1
    assert payload['duplicate_groups'] == 1
    assert payload['skipped'] == 0
    assert payload['remaining'] == 1

    with app_module.db_lock:
        rows = app_module.db.execute(
            'SELECT "Source Index" FROM processed_games ORDER BY CAST("Source Index" AS INTEGER)'
        ).fetchall()

    assert [row['Source Index'] for row in rows] == ['0']
    df = app_module.catalog_state.games_df
    assert len(df) == 1
    assert list(df['Source Index']) == ['0']
    assert app_module.catalog_state.total_games == 1


def _insert_igdb_update(
    app_module, processed_id: int, refreshed_at: str, *, has_diff: int = 0
) -> None:
    with app_module.db_lock:
        with app_module.db:
            app_module.db.execute(
                '''INSERT INTO igdb_updates (
                        processed_game_id, igdb_id, igdb_updated_at,
                        igdb_payload, diff, local_last_edited_at, refreshed_at, has_diff
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                (
                    processed_id,
                    str(100 + processed_id),
                    '2024-01-01T00:00:00+00:00',
                    '{}',
                    '{}',
                    '2024-01-01T00:00:00+00:00',
                    refreshed_at,
                    has_diff,
                ),
            )


def test_updates_status_idle(tmp_path):
    app_module = load_app(tmp_path)
    insert_processed_game(app_module)
    refreshed_at = '2024-05-01T12:00:00+00:00'
    _insert_igdb_update(app_module, 1, refreshed_at)
    app_module.routes_updates._context['job_manager'] = StubJobManager()

    client = app_module.app.test_client()
    authenticate(client)
    response = client.get('/api/updates/status')

    assert response.status_code == 200
    payload = response.get_json()
    assert payload['phase'] == 'idle'
    assert payload['queued'] == 0
    assert payload['processed'] == 0
    assert payload['last_refreshed_at'] == refreshed_at
    assert payload['errors'] == []


def test_updates_status_reports_timeout_counts(tmp_path):
    app_module = load_app(tmp_path)
    insert_processed_game(app_module)
    refreshed_at = '2024-05-01T12:00:00+00:00'
    _insert_igdb_update(app_module, 1, refreshed_at)
    app_module.routes_updates._context['job_manager'] = StubJobManager()
    app_module.routes_updates._context['get_igdb_timeout_count'] = lambda: 3

    client = app_module.app.test_client()
    authenticate(client)
    response = client.get('/api/updates/status')

    assert response.status_code == 200
    payload = response.get_json()
    assert payload['phase'] == 'idle'
    assert payload['queued'] == 0
    assert payload['processed'] == 0
    assert payload['last_refreshed_at'] == refreshed_at
    assert 'IGDB timeouts: 3' in payload['errors']


def test_updates_status_running(tmp_path):
    app_module = load_app(tmp_path)
    active_job = {
        'status': 'running',
        'progress_current': 5,
        'progress_total': 20,
        'data': {'phase': 'diffs'},
        'result': {},
    }
    app_module.routes_updates._context['job_manager'] = StubJobManager(
        active_job=active_job
    )

    client = app_module.app.test_client()
    authenticate(client)
    response = client.get('/api/updates/status')

    assert response.status_code == 200
    payload = response.get_json()
    assert payload['phase'] == 'diffs'
    assert payload['processed'] == 5
    assert payload['queued'] == 15
    assert payload['last_refreshed_at'] is None
    assert payload['errors'] == []


def test_updates_status_after_completion(tmp_path):
    app_module = load_app(tmp_path)
    insert_processed_game(app_module)
    refreshed_at = '2024-06-15T08:00:00+00:00'
    _insert_igdb_update(app_module, 1, refreshed_at)
    history_job = {
        'status': 'error',
        'error': 'refresh failed',
        'result': {'errors': ['timeout']},
    }
    app_module.routes_updates._context['job_manager'] = StubJobManager(
        history=[history_job]
    )

    client = app_module.app.test_client()
    authenticate(client)
    response = client.get('/api/updates/status')

    assert response.status_code == 200
    payload = response.get_json()
    assert payload['phase'] == 'error'
    assert payload['queued'] == 0
    assert payload['processed'] == 0
    assert payload['last_refreshed_at'] == refreshed_at
    assert 'refresh failed' in payload['errors']
    assert 'timeout' in payload['errors']
