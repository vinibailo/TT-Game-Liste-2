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


def test_load_ignores_stale_progress(tmp_path):
    app = load_app(tmp_path)
    populate_db(app, 5)
    progress_file = tmp_path / app.PROGRESS_JSON
    progress_file.write_text(json.dumps({'current_index': 2, 'seq_index': 3, 'skip_queue': []}))
    nav = app.GameNavigator(10)
    assert nav.current_index == 5
    assert nav.seq_index == 6
    data = json.loads(progress_file.read_text())
    assert data['current_index'] == 5
    assert data['seq_index'] == 6


def test_load_ignores_corrupted_progress(tmp_path):
    app = load_app(tmp_path)
    populate_db(app, 5)
    progress_file = tmp_path / app.PROGRESS_JSON
    progress_file.write_text(json.dumps({'current_index': 5, 'seq_index': 10, 'skip_queue': []}))
    nav = app.GameNavigator(10)
    assert nav.current_index == 5
    assert nav.seq_index == 6
    data = json.loads(progress_file.read_text())
    assert data['current_index'] == 5
    assert data['seq_index'] == 6
