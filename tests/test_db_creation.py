"""Smoke tests covering processed-games database initialization."""

from db import utils as db_utils

from tests.app_helpers import load_app, using_legacy_sqlite


def test_db_created_without_migration(tmp_path):
    """The application should create the processed games schema on startup."""

    assert not (tmp_path / "processed_games.xlsx").exists()

    app = load_app(tmp_path)

    with app.db_lock:
        db_utils.clear_processed_games_columns_cache()
        columns = db_utils.get_processed_games_columns(handle=app.db)

    assert 'ID' in columns
    assert 'Source Index' in columns

    if using_legacy_sqlite():
        assert (tmp_path / "processed_games.db").exists()
