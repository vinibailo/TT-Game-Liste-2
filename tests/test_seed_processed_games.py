import pandas as pd

import pandas as pd

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
