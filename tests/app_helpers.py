"""Shared testing helpers for loading the Flask app without live IGDB calls."""

from __future__ import annotations

import importlib.util
import os
import uuid
from pathlib import Path

import pandas as pd

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

    if hasattr(module, "catalog_state"):
        module.catalog_state.set_games_dataframe(
            module.catalog_state.games_df.copy(),
            rebuild_metadata=False,
            rebuild_navigator=True,
        )
    if hasattr(module, "reset_source_index_cache"):
        module.reset_source_index_cache()

    if hasattr(module, "app"):
        module.app.config['TESTING'] = True
        module.app.testing = True

    if hasattr(module, "routes_updates"):
        module.routes_updates._context['validate_igdb_credentials'] = lambda: True

    if hasattr(module, "exchange_twitch_credentials"):
        module.exchange_twitch_credentials = lambda: ('token', 'client')

    if hasattr(module, "igdb_api_client"):
        module.igdb_api_client.exchange_twitch_credentials = (
            lambda **_kwargs: ('token', 'client')
        )

    if hasattr(module, "_recreate_lookup_join_tables") and hasattr(module, "db"):
        with module.db_lock:
            module._recreate_lookup_join_tables(module.db)

    if hasattr(module, "_ensure_lookup_id_columns") and hasattr(module, "db"):
        with module.db_lock:
            module._ensure_lookup_id_columns(module.db)

    return module


def set_games_dataframe(
    module: object,
    df: pd.DataFrame,
    *,
    rebuild_metadata: bool = True,
    rebuild_navigator: bool = True,
) -> None:
    """Helper to update the app's games DataFrame during tests."""

    if hasattr(module, "_set_games_dataframe"):
        module._set_games_dataframe(
            df,
            rebuild_metadata=rebuild_metadata,
            rebuild_navigator=rebuild_navigator,
        )
    elif hasattr(module, "catalog_state"):
        module.catalog_state.set_games_dataframe(
            df,
            rebuild_metadata=rebuild_metadata,
            rebuild_navigator=rebuild_navigator,
        )
