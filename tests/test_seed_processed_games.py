import pandas as pd

from tests.app_helpers import load_app


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

    app_module.games_df = pd.DataFrame(
        [
            {'Source Index': 'A-123', 'Name': 'Alpha Updated'},
            {'Source Index': 'B-456', 'Name': 'Beta'},
            {'Source Index': None, 'Name': 'Should Skip'},
        ]
    )
    app_module.total_games = len(app_module.games_df)

    app_module.seed_processed_games_from_source()

    with app_module.db_lock:
        rows = app_module.db.execute(
            'SELECT "Source Index", "Name" FROM processed_games ORDER BY "Source Index"'
        ).fetchall()

    assert len(rows) == 2
    assert [row['Source Index'] for row in rows] == ['A-123', 'B-456']
    assert rows[0]['Name'] == 'Alpha Updated'
    assert rows[1]['Name'] == 'Beta'
