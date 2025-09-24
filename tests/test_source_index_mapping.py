import pandas as pd

from tests.app_helpers import load_app


def authenticate(client):
    with client.session_transaction() as sess:
        sess['authenticated'] = True


def test_non_zero_source_index_flow(tmp_path):
    app = load_app(tmp_path)
    app.games_df = pd.DataFrame(
        [
            {
                'Source Index': '100',
                'Name': 'First Game',
                'Summary': 'First summary',
                'id': 101,
            },
            {
                'Source Index': '200',
                'Name': 'Second Game',
                'Summary': 'Second summary',
                'id': 202,
            },
            {
                'Source Index': '300',
                'Name': 'Third Game',
                'Summary': 'Third summary',
                'id': 303,
            },
        ]
    )
    if hasattr(app, 'reset_source_index_cache'):
        app.reset_source_index_cache()
    app.total_games = len(app.games_df)
    with app.db_lock:
        with app.db:
            app.db.execute('DELETE FROM processed_games')
            app.db.execute('DELETE FROM navigator_state')
            app.db.execute(
                '''
                INSERT INTO processed_games (
                    "ID", "Source Index", "Name", "Summary", last_edited_at
                ) VALUES (?, ?, ?, ?, ?)
                ''',
                (5, '200', 'Stored Name', 'Stored summary', '2024-01-01T00:00:00Z'),
            )
    app.navigator = app.GameNavigator(app.total_games)

    assert app.get_source_index_for_position(1) == '200'
    assert app.get_position_for_source_index('200') == 1
    assert app.navigator.processed_total == 1

    payload = app.build_game_payload(
        1, app.navigator.seq_index, app.navigator.processed_total + 1
    )
    assert payload['id'] == 5
    assert payload['game']['Name'] == 'Stored Name'

    client = app.app.test_client()
    authenticate(client)

    app.navigator.current_index = 1
    response = client.post(
        '/api/save',
        json={
            'index': 1,
            'id': '5',
            'fields': {'Name': 'Updated Name', 'Summary': 'Updated Summary'},
        },
    )
    assert response.status_code == 200
    with app.db_lock:
        updated_row = app.db.execute(
            'SELECT "ID", "Source Index", "Name", "Summary" '
            'FROM processed_games WHERE "ID"=?',
            (5,),
        ).fetchone()
    assert updated_row['Source Index'] == '200'
    assert updated_row['Name'] == 'Updated Name'
    assert updated_row['Summary'] == 'Updated Summary'
    assert app.navigator.seq_index == 6

    app.navigator.current_index = 2
    new_id = app.navigator.seq_index
    response = client.post(
        '/api/save',
        json={
            'index': 2,
            'id': str(new_id),
            'fields': {'Name': 'Third Processed'},
        },
    )
    assert response.status_code == 200
    assert app.navigator.seq_index == new_id + 1
    assert app.navigator.processed_total == 2
    with app.db_lock:
        new_row = app.db.execute(
            'SELECT "ID", "Source Index", "Name" '
            'FROM processed_games WHERE "ID"=?',
            (new_id,),
        ).fetchone()
    assert new_row['Source Index'] == '300'
    assert new_row['Name'] == 'Third Processed'
    assert app.get_position_for_source_index('300') == 2
