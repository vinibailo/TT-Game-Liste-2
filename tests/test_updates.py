"""Unit and integration tests for IGDB cache refresh utilities."""
from __future__ import annotations
import sqlite3
from threading import RLock
from typing import Any
from unittest.mock import patch

from igdb.cache import IGDB_CACHE_STATE_TABLE, IGDB_CACHE_TABLE
from updates.service import refresh_igdb_cache

from tests.app_helpers import load_app


def authenticate(client: Any) -> None:
    """Mark the Flask test client session as authenticated."""

    with client.session_transaction() as sess:
        sess['authenticated'] = True


def _create_payload(
    igdb_id: int,
    *,
    name: str = 'Example Game',
    summary: str = 'Summary',
    updated_at: int = 1_700_000_000,
    first_release_date: int = 1_600_000_000,
    category: int = 0,
    cover_image_id: str = 'cover123',
    rating_count: int = 10,
    developers: list[str] | None = None,
    publishers: list[str] | None = None,
    genres: list[str] | None = None,
    platforms: list[str] | None = None,
    game_modes: list[str] | None = None,
) -> dict[str, Any]:
    return {
        'id': igdb_id,
        'name': name,
        'summary': summary,
        'updated_at': updated_at,
        'first_release_date': first_release_date,
        'category': category,
        'cover': {'image_id': cover_image_id},
        'rating_count': rating_count,
        'developers': developers or ['Developer Studio'],
        'publishers': publishers or ['Publisher House'],
        'genres': genres or ['Action'],
        'platforms': platforms or ['PC'],
        'game_modes': game_modes or ['Single player'],
    }


def test_refresh_igdb_cache_inserts_and_updates() -> None:
    """Ensure ``refresh_igdb_cache`` performs inserts and updates correctly."""

    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row
    db_lock = RLock()

    def get_db() -> sqlite3.Connection:
        return conn

    initial_payloads = [
        _create_payload(100, summary='Original summary'),
        _create_payload(200, name='Second Game'),
    ]
    updated_payloads = [
        _create_payload(100, summary='Updated summary'),
        _create_payload(200, name='Second Game'),
    ]

    with patch('updates.service.client_download_igdb_game_count', autospec=True) as mock_count, patch(
        'updates.service.client_download_igdb_games', autospec=True
    ) as mock_games:
        mock_count.return_value = 3
        mock_games.side_effect = [initial_payloads, updated_payloads]

        result_insert = refresh_igdb_cache(
            'token',
            'client',
            0,
            2,
            db_lock=db_lock,
            get_db=get_db,
        )

        assert result_insert == {
            'status': 'ok',
            'total': 3,
            'processed': 2,
            'inserted': 2,
            'updated': 0,
            'unchanged': 0,
            'done': False,
            'next_offset': 2,
            'batch_count': 2,
        }

        with db_lock:
            rows = conn.execute(
                f'SELECT igdb_id, name, summary FROM {IGDB_CACHE_TABLE} ORDER BY igdb_id'
            ).fetchall()
        assert [(row['igdb_id'], row['name'], row['summary']) for row in rows] == [
            (100, 'Example Game', 'Original summary'),
            (200, 'Second Game', 'Summary'),
        ]

        result_update = refresh_igdb_cache(
            'token',
            'client',
            0,
            2,
            db_lock=db_lock,
            get_db=get_db,
        )

        assert result_update == {
            'status': 'ok',
            'total': 3,
            'processed': 2,
            'inserted': 0,
            'updated': 1,
            'unchanged': 1,
            'done': False,
            'next_offset': 2,
            'batch_count': 2,
        }

        with db_lock:
            updated_rows = conn.execute(
                f'SELECT igdb_id, name, summary FROM {IGDB_CACHE_TABLE} ORDER BY igdb_id'
            ).fetchall()
            cached_total = conn.execute(
                f'SELECT total_count FROM {IGDB_CACHE_STATE_TABLE} WHERE id = 1'
            ).fetchone()

    assert [(row['igdb_id'], row['name'], row['summary']) for row in updated_rows] == [
        (100, 'Example Game', 'Updated summary'),
        (200, 'Second Game', 'Summary'),
    ]
    assert cached_total['total_count'] == 3
    assert mock_count.call_count == 2
    assert mock_games.call_count == 2


def test_api_igdb_cache_refresh_batches_and_updates(tmp_path) -> None:
    """Integration test ensuring the API inserts and updates IGDB cache entries."""

    app_module = load_app(tmp_path)
    client = app_module.app.test_client()
    authenticate(client)

    batches = [
        [
            _create_payload(300, name='Cache Game', summary='Initial summary'),
            _create_payload(400, name='Another Game'),
        ],
        [
            _create_payload(300, name='Cache Game', summary='Updated via API'),
            _create_payload(400, name='Another Game'),
        ],
    ]

    download_calls: list[tuple[int, int]] = []

    def fake_exchange() -> tuple[str, str]:
        return 'token', 'client'

    def fake_count(token: str, client_id: str) -> int:
        assert token == 'token'
        assert client_id == 'client'
        return 3

    def fake_download(token: str, client_id: str, offset: int, limit: int):
        assert token == 'token'
        assert client_id == 'client'
        download_calls.append((offset, limit))
        if batches:
            return batches.pop(0)
        return []

    routes_updates = app_module.routes_updates

    with patch.dict(
        routes_updates._context,
        {
            'exchange_twitch_credentials': lambda: fake_exchange,
            'download_igdb_game_count': lambda: fake_count,
            'download_igdb_games': lambda: fake_download,
        },
        clear=False,
    ):
        first_response = client.post('/api/igdb/cache', json={'offset': 0, 'limit': 2})
        assert first_response.status_code == 200
        first_payload = first_response.get_json()
        assert first_payload == {
            'status': 'ok',
            'total': 3,
            'processed': 2,
            'inserted': 2,
            'updated': 0,
            'unchanged': 0,
            'done': False,
            'next_offset': 2,
            'batch_count': 2,
        }

        second_response = client.post('/api/igdb/cache', json={'offset': 0, 'limit': 2})
        assert second_response.status_code == 200
        second_payload = second_response.get_json()
        assert second_payload == {
            'status': 'ok',
            'total': 3,
            'processed': 2,
            'inserted': 0,
            'updated': 1,
            'unchanged': 1,
            'done': False,
            'next_offset': 2,
            'batch_count': 2,
        }

    with app_module.db_lock:
        rows = app_module.db.execute(
            f'SELECT igdb_id, name, summary FROM {app_module.IGDB_CACHE_TABLE} ORDER BY igdb_id'
        ).fetchall()

    assert download_calls == [(0, 2), (0, 2)]
    assert [(row['igdb_id'], row['name'], row['summary']) for row in rows] == [
        (300, 'Cache Game', 'Updated via API'),
        (400, 'Another Game', 'Summary'),
    ]


def test_api_igdb_cache_refresh_skips_when_offset_complete(tmp_path) -> None:
    """Integration test ensuring the API short-circuits when offset exceeds total."""

    app_module = load_app(tmp_path)
    client = app_module.app.test_client()
    authenticate(client)

    def fake_exchange() -> tuple[str, str]:
        return 'token', 'client'

    count_calls = []

    def fake_count(token: str, client_id: str) -> int:
        count_calls.append((token, client_id))
        return 2

    def fake_download(*_args: Any, **_kwargs: Any):
        raise AssertionError('download_igdb_games should not be called when offset >= total')

    routes_updates = app_module.routes_updates

    with patch.dict(
        routes_updates._context,
        {
            'exchange_twitch_credentials': lambda: fake_exchange,
            'download_igdb_game_count': lambda: fake_count,
            'download_igdb_games': lambda: fake_download,
        },
        clear=False,
    ):
        response = client.post('/api/igdb/cache', json={'offset': 5, 'limit': 50})
        assert response.status_code == 200
        payload = response.get_json()
        assert payload == {
            'status': 'ok',
            'total': 2,
            'processed': 2,
            'inserted': 0,
            'updated': 0,
            'unchanged': 0,
            'done': True,
            'next_offset': 2,
            'batch_count': 0,
        }

    assert count_calls == [('token', 'client')]

    with app_module.db_lock:
        has_games_table = app_module.db.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?",
            (app_module.IGDB_CACHE_TABLE,),
        ).fetchone()[0]
        if has_games_table:
            cache_entries = app_module.db.execute(
                f'SELECT COUNT(*) FROM {app_module.IGDB_CACHE_TABLE}'
            ).fetchone()[0]
        else:
            cache_entries = 0
    assert has_games_table in (0, 1)
    assert cache_entries == 0
