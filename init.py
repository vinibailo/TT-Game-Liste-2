"""Application startup orchestration helpers."""

from __future__ import annotations

import logging
from typing import Callable

import pandas as pd

from config import RUN_DB_MIGRATIONS
from db import utils as db_utils

logger = logging.getLogger(__name__)


def initialize_app(
    *,
    ensure_dirs: Callable[[], None],
    init_db: Callable[..., None],
    load_games: Callable[..., pd.DataFrame],
    set_games_dataframe: Callable[..., None],
    connection_factory: Callable[[], db_utils.DatabaseEngine | db_utils.DatabaseHandle],
    run_migrations: bool = RUN_DB_MIGRATIONS,
    rebuild_metadata: bool = True,
    rebuild_navigator: bool = True,
    prefer_cache: bool = False,
) -> db_utils.DatabaseEngine | db_utils.DatabaseHandle:
    """Perform the core startup tasks required for the application.

    The initializer ensures filesystem directories exist, prepares the processed
    games database (including migrations and lookup seeding), establishes the
    fallback SQLite connection or SQLAlchemy engine, and loads the source games
    workbook into the in-memory navigator state. The database handle produced by
    ``connection_factory`` is returned unchanged so callers can reuse the
    SQLAlchemy-oriented interface directly.

    Parameters mirror the existing helper functions in :mod:`app` so that the
    orchestration can remain testable and reusable from scripts.
    """

    ensure_dirs()

    init_db(run_migrations=run_migrations)

    connection = connection_factory()
    db_utils.set_fallback_connection(connection)

    games_df = load_games(prefer_cache=prefer_cache)

    try:
        set_games_dataframe(
            games_df,
            rebuild_metadata=rebuild_metadata,
            rebuild_navigator=rebuild_navigator,
        )
    except Exception:
        logger.exception("Failed to configure navigator state during startup")
        raise

    return connection


__all__ = ["initialize_app"]

