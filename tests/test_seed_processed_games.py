import pandas as pd
from types import SimpleNamespace

from tests.app_helpers import load_app, set_games_dataframe


def test_seed_processed_games_respects_existing_keys(tmp_path):
    app_module = load_app(tmp_path)

    with app_module.db_lock:
        with app_module.db:
            app_module.db.execute('DELETE FROM processed_games')
            app_module.db.executemany(
                'INSERT INTO processed_games ("ID", "Source Index", "Name") VALUES (?, ?, ?)',
                [
                    (1, 'A-123 ', 'Stored A'),
                    (2, 'B-456', 'Stored B'),
                ],
            )

    set_games_dataframe(
        app_module,
        pd.DataFrame(
            [
                {'Source Index': 'A-123', 'Name': 'Alpha Updated'},
                {'Source Index': 'B-456', 'Name': 'Beta'},
                {'Source Index': None, 'Name': 'Should Skip'},
            ]
        ),
    )

    app_module.seed_processed_games_from_source()

    with app_module.db_lock:
        rows = app_module.db.execute(
            'SELECT "Source Index", "Name" FROM processed_games ORDER BY "Source Index"'
        ).fetchall()

    assert len(rows) == 2
    assert [row['Source Index'] for row in rows] == ['A-123', 'B-456']
    assert rows[0]['Name'] == 'Alpha Updated'
    assert rows[1]['Name'] == 'Beta'


def test_seed_processed_games_skips_rows_with_summary(tmp_path):
    app_module = load_app(tmp_path)

    with app_module.db_lock:
        with app_module.db:
            app_module.db.execute('DELETE FROM processed_games')
            app_module.db.execute(
                'INSERT INTO processed_games ("ID", "Source Index", "Name", "Summary", "Cover Path") '
                'VALUES (?, ?, ?, ?, ?)',
                (
                    5,
                    ' 00123 ',
                    'Stored Name',
                    'Manual summary',
                    f"{app_module.PROCESSED_DIR}/5.jpg",
                ),
            )

    set_games_dataframe(
        app_module,
        pd.DataFrame([
            {'Source Index': '00123', 'Name': 'Updated From IGDB'},
        ]),
    )

    app_module.seed_processed_games_from_source()

    with app_module.db_lock:
        row = app_module.db.execute(
            'SELECT "Source Index", "Name" FROM processed_games WHERE "ID"=?',
            (5,),
        ).fetchone()

    assert row['Source Index'] == ' 00123 '
    assert row['Name'] == 'Stored Name'


class _FakeRow(dict):
    def __getitem__(self, key):
        if isinstance(key, int):
            return list(self.values())[key]
        return super().__getitem__(key)


class _FakeResult:
    def __init__(self, rows=None):
        self._rows = [row if isinstance(row, _FakeRow) else _FakeRow(row) for row in (rows or [])]

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return list(self._rows)


class _FakeTransaction:
    def __init__(self, connection):
        self._connection = connection
        self.committed = False
        self.rolled_back = False

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True


class _FakeMariaDBConnection:
    def __init__(self):
        self.calls = []
        self.inserts = []

    def exec_driver_sql(self, sql, params=None):
        parameters = tuple(params or ())
        self.calls.append(('exec', sql, parameters))
        normalized = ' '.join(sql.split()).lower()
        if 'select max(cache_rank)' in normalized:
            return _FakeResult([{'max_rank': None}])
        if normalized.startswith('select "source index"') and 'from processed_games' in normalized:
            return _FakeResult([])
        if normalized.startswith('insert into processed_games'):
            self.inserts.append((sql, parameters))
            return _FakeResult([])
        return _FakeResult([])

    def execute(self, stmt, params=None):
        return self.exec_driver_sql(stmt, params)

    def begin(self):
        return _FakeTransaction(self)

    def in_transaction(self):
        return False


class _FakeHandle:
    def __init__(self, connection, dialect_name='mariadb'):
        self._connection = connection
        self.engine = SimpleNamespace(dialect=SimpleNamespace(name=dialect_name))

    def __enter__(self):
        return self._connection

    def __exit__(self, exc_type, exc, tb):
        return False

    def sa_connection(self):
        connection = self._connection

        class _Context:
            def __enter__(self_inner):
                return connection

            def __exit__(self_inner, exc_type, exc, tb):
                return False

        return _Context()


def test_seed_processed_games_with_mariadb_dialect(monkeypatch, tmp_path):
    app_module = load_app(tmp_path)

    fake_conn = _FakeMariaDBConnection()
    fake_handle = _FakeHandle(fake_conn)

    set_games_dataframe(
        app_module,
        pd.DataFrame([
            {'Source Index': 'A-001', 'Name': 'Example Game'},
        ]),
    )

    monkeypatch.setattr(app_module, 'mysql_insert', None)
    monkeypatch.setattr(app_module, '_load_lookup_tables', lambda _handle: None)

    def fake_get_columns(_conn=None):
        return {'Source Index', 'Name'}

    monkeypatch.setattr(app_module, 'get_processed_games_columns', fake_get_columns)
    monkeypatch.setattr(app_module, 'fetch_igdb_metadata', lambda _ids, conn=None: {})
    monkeypatch.setattr(app_module, 'get_db', lambda: fake_handle)

    def fake_with_lookup(conn, func, *, write=False):
        assert conn is fake_handle
        if write:
            txn = fake_conn.begin()
            try:
                result = func(fake_conn)
            except Exception:
                txn.rollback()
                raise
            else:
                txn.commit()
                return result
        return func(fake_conn)

    monkeypatch.setattr(app_module, '_with_lookup_connection', fake_with_lookup)

    app_module.seed_processed_games_from_source()

    assert fake_conn.inserts, 'Expected an insert for MariaDB dialect'
    sql, params = fake_conn.inserts[0]
    assert 'on duplicate key update' in sql.lower()
    assert params == ('A-001', 'Example Game')
