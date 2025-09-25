import pandas as pd

from tests.app_helpers import load_app, set_games_dataframe


def test_backfill_igdb_ids_skips_rows_with_summary(tmp_path):
    app_module = load_app(tmp_path)

    with app_module.db_lock:
        with app_module.db:
            app_module.db.execute('DELETE FROM processed_games')
            app_module.db.executemany(
                'INSERT INTO processed_games ("ID", "Source Index", "Name", "Summary", "Cover Path") '
                'VALUES (?, ?, ?, ?, ?)',
                [
                    (1, '0', 'First Game', '', None),
                    (2, '1', 'Second Game', 'Existing summary', f"{app_module.PROCESSED_DIR}/2.jpg"),
                ],
            )

    set_games_dataframe(
        app_module,
        pd.DataFrame(
            [
                {'Source Index': '0', 'id': 101},
                {'Source Index': '1', 'id': 202},
            ]
        ),
    )

    app_module.backfill_igdb_ids()

    with app_module.db_lock:
        rows = app_module.db.execute(
            'SELECT "Source Index", "igdb_id" FROM processed_games ORDER BY "Source Index"'
        ).fetchall()

    assert rows[0]['Source Index'] == '0'
    assert rows[0]['igdb_id'] == '101'
    assert rows[1]['Source Index'] == '1'
    assert rows[1]['igdb_id'] is None
