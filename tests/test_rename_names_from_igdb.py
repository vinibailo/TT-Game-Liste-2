import importlib
import sys

from tests.app_helpers import load_app


def _load_script(app_module):
    module_name = "scripts.rename_names_from_igdb"
    original_app = sys.modules.get("app")
    sys.modules["app"] = app_module
    try:
        if module_name in sys.modules:
            module = importlib.reload(sys.modules[module_name])
        else:
            module = importlib.import_module(module_name)
    finally:
        if original_app is None:
            sys.modules.pop("app", None)
        else:
            sys.modules["app"] = original_app
    return module


def test_rename_updates_names_from_igdb(tmp_path):
    app_module = load_app(tmp_path)
    script_module = _load_script(app_module)

    with app_module.db_lock:
        with app_module.db:
            app_module.db.execute('DELETE FROM processed_games')
            app_module.db.executemany(
                'INSERT INTO processed_games ("ID", "Source Index", "Name", "igdb_id") '
                'VALUES (?, ?, ?, ?)',
                [
                    (1, '10', 'Old Name', '100'),
                    (2, '20', 'Same Name', '200'),
                    (3, '30', 'Missing Remote', '300'),
                    (4, '40', 'No ID row', None),
                ],
            )

    captured_ids = []

    def fake_exchange():
        return 'token', 'client'

    def fake_fetch(token, client_id, igdb_ids):
        captured_ids.extend(list(igdb_ids))
        assert token == 'token'
        assert client_id == 'client'
        return {
            '100': {'name': 'New Name'},
            '200': {'name': 'Same Name'},
        }

    summary = script_module.rename_processed_games_from_igdb(
        conn=app_module.db,
        exchange_credentials=fake_exchange,
        metadata_loader=fake_fetch,
    )

    assert captured_ids == ['100', '200', '300']
    assert summary['updated'] == 1
    assert summary['unchanged'] == 1
    assert summary['missing_remote'] == ['300']
    assert summary['missing_name'] == []
    assert summary['missing_id'] == 1
    assert summary['rows_with_igdb_id'] == 3
    assert summary['total_rows'] == 4
    assert summary['renamed_rows'] == [
        {
            'id': 1,
            'source_index': '10',
            'igdb_id': '100',
            'old_name': 'Old Name',
            'new_name': 'New Name',
        }
    ]

    with app_module.db_lock:
        rows = app_module.db.execute(
            'SELECT "ID", "Name" FROM processed_games ORDER BY "ID"'
        ).fetchall()

    assert [row['Name'] for row in rows] == [
        'New Name',
        'Same Name',
        'Missing Remote',
        'No ID row',
    ]


def test_rename_skips_rows_without_remote_name(tmp_path):
    app_module = load_app(tmp_path)
    script_module = _load_script(app_module)

    with app_module.db_lock:
        with app_module.db:
            app_module.db.execute('DELETE FROM processed_games')
            app_module.db.execute(
                'INSERT INTO processed_games ("ID", "Source Index", "Name", "igdb_id") '
                'VALUES (?, ?, ?, ?)',
                (1, '55', 'Keep Original', '555'),
            )

    def fake_exchange():
        return 'token', 'client'

    def fake_fetch(token, client_id, igdb_ids):
        assert list(igdb_ids) == ['555']
        return {'555': {'name': '   '}}

    summary = script_module.rename_processed_games_from_igdb(
        conn=app_module.db,
        exchange_credentials=fake_exchange,
        metadata_loader=fake_fetch,
    )

    assert summary['updated'] == 0
    assert summary['missing_name'] == ['555']
    assert summary['missing_remote'] == []

    with app_module.db_lock:
        name = app_module.db.execute(
            'SELECT "Name" FROM processed_games WHERE "ID"=?',
            (1,),
        ).fetchone()['Name']

    assert name == 'Keep Original'


def test_rename_without_ids_avoids_api_calls(tmp_path):
    app_module = load_app(tmp_path)
    script_module = _load_script(app_module)

    with app_module.db_lock:
        with app_module.db:
            app_module.db.execute('DELETE FROM processed_games')
            app_module.db.execute(
                'INSERT INTO processed_games ("ID", "Source Index", "Name", "igdb_id") '
                'VALUES (?, ?, ?, ?)',
                (1, '77', 'No IGDB', None),
            )

    def fail_exchange():  # pragma: no cover - should not be called
        raise AssertionError('exchange_twitch_credentials should not be invoked')

    def fail_fetch(*_args):  # pragma: no cover - should not be called
        raise AssertionError('fetch_igdb_metadata should not be invoked')

    summary = script_module.rename_processed_games_from_igdb(
        conn=app_module.db,
        exchange_credentials=fail_exchange,
        metadata_loader=fail_fetch,
    )

    assert summary['total_rows'] == 1
    assert summary['rows_with_igdb_id'] == 0
    assert summary['updated'] == 0
    assert summary['missing_id'] == 1
