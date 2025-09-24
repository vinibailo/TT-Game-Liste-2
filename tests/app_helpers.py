"""Shared testing helpers for loading the Flask app without live IGDB calls."""

from __future__ import annotations

import importlib.util
import os
import uuid
from pathlib import Path

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def load_app(tmp_path: Path) -> object:
    """Import the application module using the on-disk processed database."""

    os.chdir(tmp_path)
    module_name = f"app_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, APP_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load app module specification")
    module = importlib.util.module_from_spec(spec)

    env_vars = {key: os.environ.get(key) for key in ("TWITCH_CLIENT_ID", "TWITCH_CLIENT_SECRET")}
    for key in env_vars:
        os.environ.pop(key, None)

    try:
        spec.loader.exec_module(module)
    finally:
        for key, value in env_vars.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    module.games_df = module.games_df.copy()
    if hasattr(module, "reset_source_index_cache"):
        module.reset_source_index_cache()
    module.total_games = len(module.games_df)
    if hasattr(module, "navigator"):
        module.navigator.total = module.total_games

    if hasattr(module, "app"):
        module.app.config['TESTING'] = True
        module.app.testing = True

    return module
