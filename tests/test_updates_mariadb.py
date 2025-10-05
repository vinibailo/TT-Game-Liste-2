import json
from types import SimpleNamespace

import pytest
from flask import Flask
from sqlalchemy.exc import OperationalError

from igdb.cache import IGDB_CACHE_TABLE
from routes import updates as updates_routes


class DummyLock:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeRow:
    def __init__(self, value):
        self.value = value

    def __getitem__(self, key):
        if key == 'igdb_id' or key == 0:
            return self.value
        if key == 'count':
            return self.value
        raise KeyError(key)


class FakeCursor:
    def __init__(self, rows):
        self._rows = list(rows)

    def fetchall(self):
        return list(self._rows)

    def fetchone(self):
        return self._rows[0] if self._rows else None


class FakeMariaDBConnection:
    def __init__(self, processed_rows=None, cache_rows=None):
        self._processed_rows = list(processed_rows or [])
        self._cache_rows = list(cache_rows or [])
        self.calls: list[tuple[str, tuple[int, ...]]] = []
        self.engine = SimpleNamespace(dialect=SimpleNamespace(name='mariadb'))

    def execute(self, sql, params=None):
        parameters = tuple(params or ())
        self.calls.append((sql, parameters))
        normalized = ' '.join(sql.split()).lower()
        if 'from' in normalized and 'processed_games' in normalized:
            rows = [FakeRow(value) for value in self._processed_rows]
            return FakeCursor(rows)
        if 'from' in normalized and IGDB_CACHE_TABLE in normalized:
            if not parameters:
                rows = [FakeRow(value) for value in self._cache_rows]
            else:
                allowed = {int(item) for item in parameters}
                rows = [FakeRow(value) for value in self._cache_rows if int(value) in allowed]
            return FakeCursor(rows)
        if normalized.startswith('select count('):
            return FakeCursor([FakeRow(len(self._cache_rows))])
        if normalized.startswith('select total_count'):
            return FakeCursor([FakeRow(self._cache_rows)])
        raise AssertionError(f'Unexpected SQL: {sql}')


class FakeJobManager:
    def __init__(self, jobs=None, active=None):
        self._jobs = list(jobs or [])
        self._active = active

    def list_jobs(self, *_args, **_kwargs):
        return list(self._jobs)

    def get_active_job(self, *_args, **_kwargs):
        return self._active


@pytest.fixture(autouse=True)
def reset_updates_context():
    original = dict(updates_routes._context)
    try:
        yield
    finally:
        updates_routes._context.clear()
        updates_routes._context.update(original)


def _coerce_int(value):
    if value in (None, ''):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(str(value).strip())
        except (TypeError, ValueError):
            return None


def test_fetch_processed_cache_ids_with_mariadb(monkeypatch):
    connection = FakeMariaDBConnection(
        processed_rows=['100', 'ignore', 200, None],
        cache_rows=[100, 200, 300],
    )
    updates_routes._context.clear()
    updates_routes._context.update({
        'db_lock': DummyLock(),
        'get_db': lambda: connection,
    })

    result = updates_routes._fetch_processed_cache_ids()

    assert result == [100, 200]
    processed_sql = connection.calls[0][0].upper()
    assert 'REGEXP' in processed_sql
    cache_sql = connection.calls[1][0].upper()
    assert 'CAST' in cache_sql and 'UNSIGNED' in cache_sql
    assert connection.calls[1][1] == (100, 200)


def test_safe_set_cached_total_handles_lock(monkeypatch):
    updates_routes._context.clear()
    lock_error = OperationalError('update cache', {}, Exception('Lock wait timeout exceeded; try restarting transaction'))

    def failing_setter(_conn, _total):
        raise lock_error

    updates_routes._context.update({'_set_cached_igdb_total': failing_setter})

    updates_routes._safe_set_cached_total(object(), 5)

    other_error = OperationalError('update cache', {}, Exception('something else'))

    def other_setter(_conn, _total):
        raise other_error

    updates_routes._context['_set_cached_igdb_total'] = other_setter

    with pytest.raises(OperationalError):
        updates_routes._safe_set_cached_total(object(), 5)


def test_progress_endpoint_with_mariadb_context(monkeypatch):
    updates_routes._context.clear()
    updates_routes._context.update({
        'job_manager': FakeJobManager([
            {'id': 'job-1', 'status': 'running', 'result': {'value': 1}},
        ])
    })

    app = Flask(__name__)
    with app.test_request_context():
        response = updates_routes.api_progress_snapshot()

    payload = json.loads(response.get_data(as_text=True))
    assert payload['jobs'][0]['id'] == 'job-1'


def test_cache_status_endpoint_with_mariadb(monkeypatch):
    connection = FakeMariaDBConnection(processed_rows=[100], cache_rows=[100])
    updates_routes._context.clear()
    updates_routes._context.update({
        'db_lock': DummyLock(),
        'get_db': lambda: connection,
        'job_manager': FakeJobManager([
            {
                'result': {
                    'cache_summary': {
                        'inserted': '2',
                        'updated': '1',
                        'unchanged': '0',
                        'total': '5',
                        'processed': '3',
                        'message': 'done',
                        'finished_at': '2024-01-02T00:00:00Z',
                    }
                },
                'started_at': '2024-01-01T12:00:00Z',
            }
        ]),
        '_coerce_int': _coerce_int,
    })

    monkeypatch.setattr(updates_routes, 'cache_get_status', lambda _conn: {
        'cached_entries': '4',
        'remote_total': '5',
        'last_synced_at': '2024-01-01T00:00:00Z',
    })

    app = Flask(__name__)
    with app.test_request_context():
        response = updates_routes.api_updates_cache_status()

    payload = json.loads(response.get_data(as_text=True))
    assert payload['cached_entries'] == 4
    assert payload['remote_total'] == 5
    assert payload['last_synced_at'] == '2024-01-01T00:00:00Z'
    assert payload['last_refresh']['inserted'] == 2
    assert payload['last_refresh']['processed'] == 3
    assert payload['last_refresh']['message'] == 'done'
    assert payload['last_refresh']['started_at'] == '2024-01-01T12:00:00Z'
