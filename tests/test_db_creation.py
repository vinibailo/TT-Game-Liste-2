from tests.app_helpers import load_app

def test_db_created_without_migration(tmp_path):
    assert not (tmp_path / "processed_games.xlsx").exists()
    load_app(tmp_path)
    assert (tmp_path / "processed_games.db").exists()
