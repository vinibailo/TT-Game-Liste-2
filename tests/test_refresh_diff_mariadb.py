from types import SimpleNamespace

from sqlalchemy.engine.default import DefaultDialect

import pytest

import app


class DummyLock:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeMariaDBHandle:
    def __init__(self):
        self.calls: list[tuple[str, tuple]] = []
        dialect = SimpleNamespace(
            name='mariadb',
            identifier_preparer=DefaultDialect().identifier_preparer,
        )
        self.engine = SimpleNamespace(dialect=dialect)
        self.committed = False

    def execute(self, sql, params=None):
        self.calls.append((sql, tuple(params or ())))

    def commit(self):
        self.committed = True


def test_refresh_diff_phase_uses_mysql_upsert(monkeypatch):
    connection = FakeMariaDBHandle()

    monkeypatch.setattr(app, 'db_lock', DummyLock())
    monkeypatch.setattr(app, 'get_db', lambda: connection)
    monkeypatch.setattr(app.db_utils, '_fallback_connection', connection, raising=False)
    monkeypatch.setattr(app, '_collect_processed_games_with_igdb', lambda: [
        {'igdb_id': '123', 'ID': 1, 'last_edited_at': '2024-01-01T00:00:00Z'}
    ])
    monkeypatch.setattr(app, 'fetch_igdb_metadata', lambda _ids: {
        '123': {'updated_at': 1700000000}
    })
    monkeypatch.setattr(app, 'build_igdb_diff', lambda _row, _payload: {'diff': True})
    monkeypatch.setattr(app, '_populate_updates_list_locked', lambda _conn: None)
    monkeypatch.setattr(app, 'now_utc_iso', lambda: '2024-01-02T00:00:00Z')
    monkeypatch.setattr(app, 'mysql_insert', None)

    result = app._run_refresh_diff_phase(lambda **_kwargs: None)

    assert result['updated'] == 1
    assert connection.committed is True
    assert connection.calls, 'Expected an INSERT/UPDATE statement to be executed.'
    sql, parameters = connection.calls[0]
    normalized_sql = ' '.join(sql.upper().split())
    assert 'ON DUPLICATE KEY UPDATE' in normalized_sql
    assert 'ON CONFLICT' not in normalized_sql
    assert parameters[0] == 1
    assert parameters[1] == '123'
