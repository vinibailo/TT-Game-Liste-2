import os
import json
import sqlite3
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from tests.app_helpers import load_app


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

        columns = {
            row['name'] if isinstance(row, Mapping) else row[1]
            for row in app.db.execute('PRAGMA table_info(processed_games)')
        }
        assert 'developers_ids' in columns
        assert 'publishers_ids' in columns
        assert 'genres_ids' in columns
        assert 'game_modes_ids' in columns
        assert 'platforms_ids' in columns
        assert 'cache_rank' in columns

        def first_lookup_id(query: str, processed_game_id: int) -> int:
            row = app.db.execute(query, (processed_game_id,)).fetchone()
            assert row is not None
            if isinstance(row, Mapping):
                return row[0]
            return row[0]

        developer_id = first_lookup_id(
            'SELECT developer_id FROM processed_game_developers WHERE processed_game_id=?',
            1,
        )
        developer_name = app.db.execute(
            'SELECT name FROM developers WHERE id=?',
            (developer_id,),
        ).fetchone()
        assert developer_name['name'] == "Foo Studio"

        publisher_id = first_lookup_id(
            'SELECT publisher_id FROM processed_game_publishers WHERE processed_game_id=?',
            1,
        )
        publisher_name = app.db.execute(
            'SELECT name FROM publishers WHERE id=?',
            (publisher_id,),
        ).fetchone()
        assert publisher_name['name'] == "Bar Publishing"

        genre_id = first_lookup_id(
            'SELECT genre_id FROM processed_game_genres WHERE processed_game_id=?',
            1,
        )
        genre_name = app.db.execute(
            'SELECT name FROM genres WHERE id=?',
            (genre_id,),
        ).fetchone()
        assert genre_name['name'] == "Action"

        mode_id = first_lookup_id(
            'SELECT game_mode_id FROM processed_game_game_modes WHERE processed_game_id=?',
            1,
        )
        mode_name = app.db.execute(
            'SELECT name FROM game_modes WHERE id=?',
            (mode_id,),
        ).fetchone()
        assert mode_name['name'] == "Single-player"

        platform_id = first_lookup_id(
            'SELECT platform_id FROM processed_game_platforms WHERE processed_game_id=?',
            1,
        )
        platform_name = app.db.execute(
            'SELECT name FROM platforms WHERE id=?',
            (platform_id,),
        ).fetchone()
        assert platform_name['name'] == "PC"

        processed_row = app.db.execute(
            'SELECT developers_ids, publishers_ids, genres_ids, '
            'game_modes_ids, platforms_ids FROM processed_games WHERE "ID"=?',
            (1,),
        ).fetchone()
        assert json.loads(processed_row['developers_ids']) == [developer_id]
        assert json.loads(processed_row['publishers_ids']) == [publisher_id]
        assert json.loads(processed_row['genres_ids']) == [genre_id]
        assert json.loads(processed_row['game_modes_ids']) == [mode_id]
        assert json.loads(processed_row['platforms_ids']) == [platform_id]
