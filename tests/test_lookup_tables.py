import os
import json
from collections.abc import Mapping
from pathlib import Path

import pandas as pd
from sqlalchemy import Column, Integer, MetaData, String, Table, Text

from db import utils as db_utils
from tests.app_helpers import get_test_db_engine, load_app


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


def initialize_legacy_db(tmp_path: Path) -> None:
    engine_wrapper = get_test_db_engine(tmp_path)
    engine = engine_wrapper.engine
    metadata = MetaData()
    processed_games = Table(
        'processed_games',
        metadata,
        Column('ID', Integer, primary_key=True),
        Column('Source Index', String(255), unique=True),
        Column('Name', Text),
        Column('Summary', Text),
        Column('Developers', Text),
        Column('Publishers', Text),
        Column('Genres', Text),
        Column('Game Modes', Text),
        Column('Platforms', Text),
        Column('Cover Path', Text),
        mysql_engine='InnoDB',
    )

    metadata.drop_all(engine, tables=[processed_games])
    metadata.create_all(engine, tables=[processed_games])

    with engine.begin() as conn:
        conn.execute(
            processed_games.insert(),
            {
                'ID': 1,
                'Source Index': '0',
                'Name': 'Sample Game',
                'Developers': 'foo studio',
                'Publishers': 'bar publishing',
                'Genres': 'action',
                'Game Modes': 'single-player',
                'Platforms': 'pc',
            },
        )

    engine_wrapper.dispose()


def test_lookup_tables_backfilled(tmp_path):
    lookup_dir = tmp_path / "lookup"
    write_lookup_workbooks(lookup_dir)

    initialize_legacy_db(tmp_path)

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

        db_utils.clear_processed_games_columns_cache()
        columns = db_utils.get_processed_games_columns(handle=app.db)
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
