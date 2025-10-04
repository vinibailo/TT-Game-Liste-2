import os

import pytest


os.environ.setdefault('DB_LEGACY_SQLITE', '1')


@pytest.fixture(autouse=True)
def enable_db_migrations(monkeypatch):
    """Ensure database migrations run for tests that rely on migrated schema."""

    monkeypatch.setenv('RUN_DB_MIGRATIONS', '1')
    yield
    monkeypatch.delenv('RUN_DB_MIGRATIONS', raising=False)
