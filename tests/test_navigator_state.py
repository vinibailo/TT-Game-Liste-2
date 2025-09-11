import os
import json
import uuid
import importlib.util
from pathlib import Path

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def load_app(tmp_path):
    os.chdir(tmp_path)
    module_name = f"app_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, APP_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def populate_db(app_module, count):
    with app_module.db_lock:
        with app_module.db:
            for i in range(count):
                app_module.db.execute(
                    'INSERT INTO processed_games ("ID", "Source Index") VALUES (?, ?)',
                    (str(i), str(i)),
                )


def read_state(app_module):
    with app_module.db_lock:
        cur = app_module.db.execute(
            'SELECT current_index, seq_index, skip_queue FROM navigator_state WHERE id=1'
        )
        return cur.fetchone()


def test_load_ignores_stale_state(tmp_path):
    app = load_app(tmp_path)
    populate_db(app, 5)
    with app.db_lock:
        with app.db:
            app.db.execute('DELETE FROM navigator_state')
            app.db.execute(
                'INSERT INTO navigator_state (id, current_index, seq_index, skip_queue) VALUES (1, 2, 3, "[]")'
            )
    nav = app.GameNavigator(10)
    assert nav.current_index == 5
    assert nav.seq_index == 5
    state = read_state(app)
    assert state['current_index'] == 5
    assert state['seq_index'] == 5


def test_load_ignores_corrupted_state(tmp_path):
    app = load_app(tmp_path)
    populate_db(app, 5)
    with app.db_lock:
        with app.db:
            app.db.execute('DELETE FROM navigator_state')
            app.db.execute(
                'INSERT INTO navigator_state (id, current_index, seq_index, skip_queue) VALUES (1, 5, 10, "[]")'
            )
    nav = app.GameNavigator(10)
    assert nav.current_index == 5
    assert nav.seq_index == 5
    state = read_state(app)
    assert state['current_index'] == 5
    assert state['seq_index'] == 5


def test_load_resumes_from_first_unprocessed_index(tmp_path):
    app = load_app(tmp_path)
    populate_db(app, 3)
    with app.db_lock:
        with app.db:
            app.db.execute('DELETE FROM processed_games WHERE "Source Index"=?', ('1',))
            app.db.execute('DELETE FROM navigator_state')
    nav = app.GameNavigator(5)
    assert nav.current_index == 1
    assert nav.seq_index == 3


def test_navigation_reloads_state_from_db(tmp_path):
    app = load_app(tmp_path)
    populate_db(app, 5)  # processed indices 0-4
    with app.db_lock:
        with app.db:
            app.db.execute('DELETE FROM navigator_state')
    nav = app.GameNavigator(10)
    assert nav.current_index == 5
    # simulate another worker processing index 5 and advancing state
    with app.db_lock:
        with app.db:
            app.db.execute(
                'INSERT INTO processed_games ("ID", "Source Index") VALUES (?, ?)',
                ('5', '5'),
            )
            app.db.execute(
                'UPDATE navigator_state SET current_index=?, seq_index=?', (6, 6)
            )
    assert nav.next() == 7
    state = read_state(app)
    assert state['current_index'] == 7
    # simulate worker resetting state to 2 and database accordingly
    with app.db_lock:
        with app.db:
            app.db.execute('DELETE FROM processed_games WHERE CAST("Source Index" AS INTEGER) >= 2')
            app.db.execute(
                'UPDATE navigator_state SET current_index=?, seq_index=?, skip_queue=?',
                (2, 2, '[]'),
            )
    assert nav.current() == 2
    # simulate worker moving state to 4 with matching processed entries
    with app.db_lock:
        with app.db:
            app.db.execute('DELETE FROM processed_games')
            for i in range(4):
                app.db.execute(
                    'INSERT INTO processed_games ("ID", "Source Index") VALUES (?, ?)',
                    (str(i), str(i)),
                )
            app.db.execute(
                'UPDATE navigator_state SET current_index=?, seq_index=?, skip_queue=?',
                (4, 4, '[]'),
            )
    assert nav.back() == 3
    # simulate worker setting state to 3 and add skip
    with app.db_lock:
        with app.db:
            app.db.execute('DELETE FROM processed_games WHERE "Source Index"=?', ('3',))
            app.db.execute(
                'UPDATE navigator_state SET current_index=?, seq_index=?, skip_queue=?',
                (3, 3, '[]'),
            )
    nav.skip(8)
    state = read_state(app)
    assert state['current_index'] == 3
    assert json.loads(state['skip_queue']) == [{'index': 8, 'countdown': 30}]


def test_sequential_navigation_moves_multiple_steps(tmp_path):
    app = load_app(tmp_path)
    populate_db(app, 5)  # processed indices 0-4
    with app.db_lock:
        with app.db:
            app.db.execute('DELETE FROM navigator_state')
    nav = app.GameNavigator(10)
    assert nav.current_index == 5
    assert nav.next() == 6
    assert nav.next() == 7
    assert nav.back() == 6
    assert nav.back() == 5


def test_next_after_save_advances_once(tmp_path):
    app = load_app(tmp_path)
    populate_db(app, 5)  # processed indices 0-4
    with app.db_lock:
        with app.db:
            app.db.execute('DELETE FROM navigator_state')
    nav = app.GameNavigator(10)
    assert nav.current_index == 5
    seq_id = f"{nav.seq_index:07d}"
    with app.db_lock:
        with app.db:
            app.db.execute(
                'INSERT INTO processed_games ("ID", "Source Index") VALUES (?, ?)',
                (seq_id, str(nav.current_index)),
            )
    nav.seq_index += 1
    nav._save()
    assert nav.next() == 6
    assert nav.current_index == 6
    assert nav.skip_queue == []

