"""Shared testing helpers for loading the Flask app without live IGDB calls."""

from __future__ import annotations

import importlib.util
import os
import uuid
from pathlib import Path

import pandas as pd
from flask import request

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def load_app(tmp_path: Path) -> object:
    """Import the application module using the on-disk processed database."""

    os.chdir(tmp_path)
    module_name = f"app_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, APP_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load app module specification")
    module = importlib.util.module_from_spec(spec)

    env_vars = {
        key: os.environ.get(key)
        for key in ("TWITCH_CLIENT_ID", "TWITCH_CLIENT_SECRET", "DB_LEGACY_SQLITE")
    }
    for key in env_vars:
        os.environ.pop(key, None)

    os.environ["DB_LEGACY_SQLITE"] = "1"

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

    run_migrations = bool(getattr(module, "RUN_DB_MIGRATIONS", False))

    if hasattr(module, "routes_updates"):
        module.routes_updates._context['validate_igdb_credentials'] = lambda: True
        def _format_refresh_response(payload, *, offset, limit):
            sync_arg = request.args.get('sync') if request else None
            if sync_arg not in (None, '', '0', 'false', 'False'):
                return None
            return {
                'status': payload.get('status'),
                'total': payload.get('total'),
                'processed': payload.get('processed'),
                'inserted': payload.get('inserted'),
                'updated': payload.get('updated'),
                'unchanged': payload.get('unchanged'),
                'done': bool(payload.get('done')),
                'next_offset': payload.get('next_offset'),
                'batch_count': payload.get('batch_count'),
            }

        module.routes_updates._context['format_refresh_response'] = _format_refresh_response

    if hasattr(module, "exchange_twitch_credentials"):
        module.exchange_twitch_credentials = lambda: ('token', 'client')

    if hasattr(module, "igdb_api_client"):
        module.igdb_api_client.exchange_twitch_credentials = (
            lambda **_kwargs: ('token', 'client')
        )

    if hasattr(module, "_ensure_lookup_join_tables") and hasattr(module, "db"):
        with module.db_lock, module.db.connection() as conn:
            module._ensure_lookup_join_tables(conn)

    if (
        run_migrations
        and hasattr(module, "_load_lookup_tables")
        and hasattr(module, "db")
    ):
        with module.db_lock, module.db.connection() as conn:
            module._load_lookup_tables(conn)

    if (
        run_migrations
        and hasattr(module, "_recreate_lookup_join_tables")
        and hasattr(module, "db")
    ):
        with module.db_lock, module.db.connection() as conn:
            module._recreate_lookup_join_tables(conn)

    if (
        run_migrations
        and hasattr(module, "_backfill_lookup_relations")
        and hasattr(module, "db")
    ):
        with module.db_lock, module.db.connection() as conn:
            module._backfill_lookup_relations(conn)

    if (
        run_migrations
        and hasattr(module, "_ensure_lookup_id_columns")
        and hasattr(module, "db")
    ):
        with module.db_lock, module.db.connection() as conn:
            module._ensure_lookup_id_columns(conn)

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
