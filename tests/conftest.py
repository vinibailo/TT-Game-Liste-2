"""Pytest fixtures shared across the test suite."""

import os

os.environ.setdefault('DB_LEGACY_SQLITE', '1')

import pytest
from sqlalchemy import inspect, text

from config import DB_DSN, DB_LEGACY_SQLITE
from db import utils as db_utils


@pytest.fixture(autouse=True)
def enable_db_migrations(monkeypatch):
    """Ensure database migrations run for tests that rely on migrated schema."""

    monkeypatch.setenv('RUN_DB_MIGRATIONS', '1')
    yield
    monkeypatch.delenv('RUN_DB_MIGRATIONS', raising=False)


@pytest.fixture(autouse=True)
def reset_database_state():
    """Reset cached handles and clean MariaDB schemas between tests."""

    db_utils.set_fallback_connection(None)
    db_utils.clear_processed_games_columns_cache()

    if not DB_LEGACY_SQLITE:
        engine_wrapper = db_utils.build_engine_from_dsn(DB_DSN)
        engine = engine_wrapper.engine
        with engine.connect() as conn:
            with conn.begin():
                inspector = inspect(conn)
                tables = inspector.get_table_names()
                if tables:
                    conn.execute(text('SET FOREIGN_KEY_CHECKS = 0'))
                    preparer = conn.dialect.identifier_preparer
                    for table in tables:
                        conn.execute(text(f"DROP TABLE IF EXISTS {preparer.quote(table)}"))
                    conn.execute(text('SET FOREIGN_KEY_CHECKS = 1'))
        engine_wrapper.dispose()

    yield

    db_utils.set_fallback_connection(None)
    db_utils.clear_processed_games_columns_cache()
