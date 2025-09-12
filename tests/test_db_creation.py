import os
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

def test_db_created_without_migration(tmp_path):
    assert not (tmp_path / "processed_games.xlsx").exists()
    load_app(tmp_path)
    assert (tmp_path / "processed_games.db").exists()
