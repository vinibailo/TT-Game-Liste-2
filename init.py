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
    connection_factory: Callable[[], db_utils.DatabaseHandle | db_utils.DatabaseEngine],
    run_migrations: bool = RUN_DB_MIGRATIONS,
    rebuild_metadata: bool = True,
    rebuild_navigator: bool = True,
    prefer_cache: bool = False,
) -> db_utils.DatabaseHandle:
    """Perform the core startup tasks required for the application.

    The initializer ensures filesystem directories exist, prepares the processed
    games database (including migrations and lookup seeding), establishes the
    fallback SQLite connection, and loads the source games workbook into the
    in-memory navigator state.

    Parameters mirror the existing helper functions in :mod:`app` so that the
    orchestration can remain testable and reusable from scripts.
    """

    ensure_dirs()

    init_db(run_migrations=run_migrations)

    connection = connection_factory()
    handle = (
        connection
        if isinstance(connection, db_utils.DatabaseHandle)
        else db_utils.DatabaseHandle(connection)
    )
    db_utils.set_fallback_connection(handle)

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

    return handle


__all__ = ["initialize_app"]

