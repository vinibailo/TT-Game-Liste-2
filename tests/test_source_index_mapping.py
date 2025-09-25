import base64
import io

import pandas as pd
from PIL import Image

from tests.app_helpers import load_app, set_games_dataframe


def authenticate(client):
    with client.session_transaction() as sess:
        sess['authenticated'] = True


def generate_image_data_url() -> str:
    img = Image.new('RGB', (10, 10), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()


def test_non_zero_source_index_flow(tmp_path):
    app = load_app(tmp_path)
    set_games_dataframe(
        app,
        pd.DataFrame(
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
        ),
    )
    navigator = app._ensure_navigator_dataframe(rebuild_state=True)
    with app.db_lock:
        with app.db:
            app.db.execute('DELETE FROM processed_games')
            app.db.execute('DELETE FROM navigator_state')
            app.db.execute(
                '''
                INSERT INTO processed_games (
                    "ID", "Source Index", "Name", "Summary", "Cover Path", last_edited_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ''',
                (
                    5,
                    '200',
                    'Stored Name',
                    'Stored summary',
                    f"{app.PROCESSED_DIR}/5.jpg",
                    '2024-01-01T00:00:00Z',
                ),
            )
    app.catalog_state.set_navigator(app.GameNavigator(app.catalog_state.total_games))
    navigator = app._ensure_navigator_dataframe(rebuild_state=False)

    assert app.get_source_index_for_position(1) == '200'
    assert app.get_position_for_source_index('200') == 1
    assert navigator.processed_total == 1

    payload = app.build_game_payload(
        1, navigator.seq_index, navigator.processed_total + 1
    )
    assert payload['id'] == 5
    assert payload['game']['Name'] == 'Stored Name'

    client = app.app.test_client()
    authenticate(client)

    navigator.current_index = 1
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
    assert navigator.seq_index == 6

    navigator.current_index = 2
    new_id = navigator.seq_index
    response = client.post(
        '/api/save',
        json={
            'index': 2,
            'id': str(new_id),
            'fields': {'Name': 'Third Processed', 'Summary': 'Third summary'},
            'image': generate_image_data_url(),
        },
    )
    assert response.status_code == 200
    navigator = app._ensure_navigator_dataframe(rebuild_state=False)
    assert navigator.seq_index == new_id + 1
    assert navigator.processed_total == 2
    with app.db_lock:
        new_row = app.db.execute(
            'SELECT "ID", "Source Index", "Name" '
            'FROM processed_games WHERE "ID"=?',
            (new_id,),
        ).fetchone()
    assert new_row['Source Index'] == '300'
    assert new_row['Name'] == 'Third Processed'
    assert app.get_position_for_source_index('300') == 2
