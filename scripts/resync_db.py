#!/usr/bin/env python3
"""Utility to align processed_games.db with the live IGDB source list."""

from typing import Any

from app import (
    load_games,
    normalize_processed_games,
    get_db,
    db_lock,
    extract_igdb_id,
    coerce_igdb_id,
)


def _iter_source_rows() -> list[tuple[str, str]]:
    games_df = load_games()
    if games_df.empty:
        return []
    return [
        (str(index), extract_igdb_id(row, allow_generic_id=True))
        for index, row in games_df.iterrows()
    ]


def _normalize_existing_id(row: Any) -> str:
    if row is None:
        return ""
    if isinstance(row, dict):
        value = row.get("igdb_id")
    else:
        try:
            value = row[0]
        except Exception:
            value = None
    return coerce_igdb_id(value)


def main() -> None:
    sources = _iter_source_rows()
    if not sources:
        print("No games returned from IGDB; nothing to resync.")
        return

    conn = get_db()
    with db_lock:
        with conn:
            for src_index, igdb_id in sources:
                cur = conn.execute(
                    'SELECT "igdb_id" FROM processed_games WHERE "Source Index"=?',
                    (src_index,),
                )
                row = cur.fetchone()
                if row is None:
                    conn.execute(
                        'INSERT INTO processed_games ("Source Index", "igdb_id") VALUES (?, ?)',
                        (src_index, igdb_id or None),
                    )
                    continue

                if not igdb_id:
                    continue

                existing_id = _normalize_existing_id(row)
                if not existing_id:
                    conn.execute(
                        'UPDATE processed_games SET "igdb_id"=? WHERE "Source Index"=?',
                        (igdb_id, src_index),
                    )

    normalize_processed_games()
    conn.close()


if __name__ == "__main__":
    main()
