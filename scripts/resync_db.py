#!/usr/bin/env python3
"""Utility to align processed_games.db with the live IGDB source list."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any
import sys

from sqlalchemy import text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from db import utils as db_utils

from app import (
    load_games,
    normalize_processed_games,
    get_db_sa_connection,
    db_lock,
    extract_igdb_id,
    coerce_igdb_id,
    is_processed_game_done,
)


def _iter_source_rows() -> list[tuple[str, str]]:
    games_df = load_games()
    if games_df.empty:
        return []
    return [
        (
            (str(row.get("Source Index", index)).strip() or str(index)),
            extract_igdb_id(row, allow_generic_id=True),
        )
        for index, row in games_df.iterrows()
    ]


def _normalize_existing_id(row: Any) -> str:
    if row is None:
        return ""
    if isinstance(row, Mapping):
        value = row.get("igdb_id")
    else:
        try:
            value = row[0]
        except Exception:
            value = row
    return coerce_igdb_id(value)


def main() -> None:
    sources = _iter_source_rows()
    if not sources:
        print("No games returned from IGDB; nothing to resync.")
        return

    processed_games_table = db_utils._quote_identifier("processed_games")
    source_index_column = db_utils._quote_identifier("Source Index")
    igdb_id_column = db_utils._quote_identifier("igdb_id")
    summary_column = db_utils._quote_identifier("Summary")
    cover_path_column = db_utils._quote_identifier("Cover Path")

    select_sql = text(
        f"SELECT {igdb_id_column} AS igdb_id, "
        f"{summary_column} AS summary_value, "
        f"{cover_path_column} AS cover_path_value "
        f"FROM {processed_games_table} "
        f"WHERE {source_index_column} = :source_index"
    )

    update_sql = text(
        f"UPDATE {processed_games_table} "
        f"SET {igdb_id_column} = :igdb_id "
        f"WHERE {source_index_column} = :source_index"
    )

    with db_lock:
        with get_db_sa_connection() as sa_conn:
            dialect_name = sa_conn.dialect.name.lower()
            if dialect_name in {"mysql", "mariadb"}:
                insert_sql = text(
                    f"INSERT INTO {processed_games_table} "
                    f"({source_index_column}, {igdb_id_column}) "
                    "VALUES (:source_index, :igdb_id) "
                    f"ON DUPLICATE KEY UPDATE {igdb_id_column} = VALUES({igdb_id_column})"
                )
            else:
                insert_sql = text(
                    f"INSERT INTO {processed_games_table} "
                    f"({source_index_column}, {igdb_id_column}) "
                    "VALUES (:source_index, :igdb_id)"
                )
            with sa_conn.begin():
                for src_index, igdb_id in sources:
                    row = (
                        sa_conn.execute(
                            select_sql,
                            {"source_index": src_index},
                        )
                        .mappings()
                        .first()
                    )
                    if row is None:
                        sa_conn.execute(
                            insert_sql,
                            {
                                "source_index": src_index,
                                "igdb_id": igdb_id or None,
                            },
                        )
                        continue

                    if not igdb_id:
                        continue

                    summary_value = row.get("summary_value")
                    cover_value = row.get("cover_path_value")
                    if is_processed_game_done(summary_value, cover_value):
                        continue

                    existing_id = _normalize_existing_id(row)
                    if not existing_id:
                        sa_conn.execute(
                            update_sql,
                            {
                                "igdb_id": igdb_id,
                                "source_index": src_index,
                            },
                        )

    normalize_processed_games()


if __name__ == "__main__":
    main()
