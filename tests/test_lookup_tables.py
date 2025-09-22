import os
import sqlite3
import uuid
import importlib.util
from pathlib import Path

import pandas as pd

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def load_app(tmp_path):
    os.chdir(tmp_path)
    module_name = f"app_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, APP_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_lookup_workbooks(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"Developer": ["Foo Studio"]}).to_excel(
        directory / "Developers_unique.xlsx", index=False
    )
    pd.DataFrame({"Publisher": ["Bar Publishing"]}).to_excel(
        directory / "Publishers_unique.xlsx", index=False
    )
    pd.DataFrame({"Genre": ["Action"]}).to_excel(
        directory / "Genres_unique.xlsx", index=False
    )
    pd.DataFrame({"GameMode": ["Single-player"]}).to_excel(
        directory / "GameModes_unique.xlsx", index=False
    )
    pd.DataFrame({"Platform": ["PC"]}).to_excel(
        directory / "Platforms_unique.xlsx", index=False
    )


def initialize_legacy_db(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    try:
        with conn:
            conn.execute(
                '''
                CREATE TABLE processed_games (
                    "ID" INTEGER PRIMARY KEY,
                    "Source Index" TEXT UNIQUE,
                    "Name" TEXT,
                    "Developers" TEXT,
                    "Publishers" TEXT,
                    "Genres" TEXT,
                    "Game Modes" TEXT,
                    "Platforms" TEXT
                )
                '''
            )
            conn.execute(
                '''
                INSERT INTO processed_games (
                    "ID", "Source Index", "Name", "Developers", "Publishers",
                    "Genres", "Game Modes", "Platforms"
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''',
                (
                    1,
                    "0",
                    "Sample Game",
                    "foo studio",
                    "bar publishing",
                    "action",
                    "single-player",
                    "pc",
                ),
            )
    finally:
        conn.close()


def test_lookup_tables_backfilled(tmp_path):
    lookup_dir = tmp_path / "lookup"
    write_lookup_workbooks(lookup_dir)

    db_path = tmp_path / "processed_games.db"
    initialize_legacy_db(db_path)

    os.environ['LOOKUP_DATA_DIR'] = str(lookup_dir)
    try:
        app = load_app(tmp_path)
    finally:
        os.environ.pop('LOOKUP_DATA_DIR', None)

    with app.db_lock:
        dev_rows = app.db.execute('SELECT name FROM developers').fetchall()
        assert {row['name'] for row in dev_rows} == {"Foo Studio"}

        pub_rows = app.db.execute('SELECT name FROM publishers').fetchall()
        assert {row['name'] for row in pub_rows} == {"Bar Publishing"}

        genre_rows = app.db.execute('SELECT name FROM genres').fetchall()
        assert {row['name'] for row in genre_rows} == {"Action"}

        mode_rows = app.db.execute('SELECT name FROM game_modes').fetchall()
        assert {row['name'] for row in mode_rows} == {"Single-player"}

        platform_rows = app.db.execute('SELECT name FROM platforms').fetchall()
        assert {row['name'] for row in platform_rows} == {"PC"}

        developer_link = app.db.execute(
            'SELECT processed_game_id, developer_id FROM processed_game_developers'
        ).fetchone()
        assert developer_link['processed_game_id'] == 1
        developer_name = app.db.execute(
            'SELECT name FROM developers WHERE id=?',
            (developer_link['developer_id'],),
        ).fetchone()
        assert developer_name['name'] == "Foo Studio"

        publisher_link = app.db.execute(
            'SELECT processed_game_id, publisher_id FROM processed_game_publishers'
        ).fetchone()
        assert publisher_link['processed_game_id'] == 1
        publisher_name = app.db.execute(
            'SELECT name FROM publishers WHERE id=?',
            (publisher_link['publisher_id'],),
        ).fetchone()
        assert publisher_name['name'] == "Bar Publishing"

        genre_link = app.db.execute(
            'SELECT processed_game_id, genre_id FROM processed_game_genres'
        ).fetchone()
        assert genre_link['processed_game_id'] == 1
        genre_name = app.db.execute(
            'SELECT name FROM genres WHERE id=?',
            (genre_link['genre_id'],),
        ).fetchone()
        assert genre_name['name'] == "Action"

        mode_link = app.db.execute(
            'SELECT processed_game_id, game_mode_id FROM processed_game_game_modes'
        ).fetchone()
        assert mode_link['processed_game_id'] == 1
        mode_name = app.db.execute(
            'SELECT name FROM game_modes WHERE id=?',
            (mode_link['game_mode_id'],),
        ).fetchone()
        assert mode_name['name'] == "Single-player"

        platform_link = app.db.execute(
            'SELECT processed_game_id, platform_id FROM processed_game_platforms'
        ).fetchone()
        assert platform_link['processed_game_id'] == 1
        platform_name = app.db.execute(
            'SELECT name FROM platforms WHERE id=?',
            (platform_link['platform_id'],),
        ).fetchone()
        assert platform_name['name'] == "PC"
