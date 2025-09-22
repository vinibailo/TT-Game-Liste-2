import os
import sqlite3
import uuid
import json
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
        processed_row = app.db.execute(
            '''SELECT developers_ids, publishers_ids, genres_ids,
                      game_modes_ids, platforms_ids
               FROM processed_games WHERE "ID"=?''',
            (1,),
        ).fetchone()
        assert processed_row is not None

        developer_ids = json.loads(processed_row['developers_ids'])
        assert developer_ids
        developer_name = app.db.execute(
            'SELECT name FROM developers WHERE id=?',
            (developer_ids[0],),
        ).fetchone()
        assert developer_name['name'] == "Foo Studio"

        publisher_ids = json.loads(processed_row['publishers_ids'])
        assert publisher_ids
        publisher_name = app.db.execute(
            'SELECT name FROM publishers WHERE id=?',
            (publisher_ids[0],),
        ).fetchone()
        assert publisher_name['name'] == "Bar Publishing"

        genre_ids = json.loads(processed_row['genres_ids'])
        assert genre_ids
        genre_name = app.db.execute(
            'SELECT name FROM genres WHERE id=?',
            (genre_ids[0],),
        ).fetchone()
        assert genre_name['name'] == "Action"

        mode_ids = json.loads(processed_row['game_modes_ids'])
        assert mode_ids
        mode_name = app.db.execute(
            'SELECT name FROM game_modes WHERE id=?',
            (mode_ids[0],),
        ).fetchone()
        assert mode_name['name'] == "Single-player"

        platform_ids = json.loads(processed_row['platforms_ids'])
        assert platform_ids
        platform_name = app.db.execute(
            'SELECT name FROM platforms WHERE id=?',
            (platform_ids[0],),
        ).fetchone()
        assert platform_name['name'] == "PC"
