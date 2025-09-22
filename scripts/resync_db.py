#!/usr/bin/env python3
"""Utility to resync processed_games.db with igdb_all_games.xlsx."""

from app import (
    load_games,
    normalize_processed_games,
    get_db,
    db_lock,
    extract_igdb_id,
)


def main() -> None:
    df = load_games()
    if df.empty:
        print("No games loaded; nothing to resync.")
        return

    conn = get_db()
    with db_lock:
        with conn:
            for idx in df.index:
                src_index = str(idx)
                row = df.iloc[idx]
                igdb_id = extract_igdb_id(row, allow_generic_id=True)
                cur = conn.execute(
                    'SELECT 1 FROM processed_games WHERE "Source Index"=?',
                    (src_index,),
                )
                if cur.fetchone() is None:
                    conn.execute(
                        'INSERT INTO processed_games ("Source Index", "igdb_id") VALUES (?, ?)',
                        (src_index, igdb_id or None),
                    )
                elif igdb_id:
                    conn.execute(
                        """UPDATE processed_games SET "igdb_id"=? WHERE "Source Index"=? AND ("igdb_id" IS NULL OR "igdb_id" = '')""",
                        (igdb_id, src_index),
                    )

    normalize_processed_games()
    conn.close()


if __name__ == "__main__":
    main()
