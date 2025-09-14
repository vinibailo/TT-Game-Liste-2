# TT-Game-Liste-2

Browser-based editor for reviewing games from `igdb_all_games.xlsx`. It lets you clean up metadata and produce uniform 1080×1080 cover images while saving progress to an SQLite database (`processed_games.db`).

Use the **Back** button to return to the previously processed game if you need to correct an earlier entry.

See [INSTALL.md](INSTALL.md) for installation instructions.

Run `python scripts/resync_db.py` to repair a misaligned database if processed entries fall out of sync with `igdb_all_games.xlsx`.
