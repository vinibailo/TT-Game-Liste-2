#!/usr/bin/env python3
"""Utility to resync processed_games.db with igdb_all_games.xlsx."""

from app import load_games, normalize_processed_games, get_db, db_lock


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
                cur = conn.execute(
                    'SELECT 1 FROM processed_games WHERE "Source Index"=?',
                    (src_index,),
                )
                if cur.fetchone() is None:
                    conn.execute(
                        'INSERT INTO processed_games ("Source Index") VALUES (?)',
                        (src_index,),
                    )

    normalize_processed_games()
    conn.close()


if __name__ == "__main__":
    main()
