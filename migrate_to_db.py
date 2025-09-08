import os
import sqlite3
import pandas as pd

PROCESSED_XLSX = 'processed_games.xlsx'
PROCESSED_DB = 'processed_games.db'

SCHEMA = '''CREATE TABLE IF NOT EXISTS processed_games (
    "ID" TEXT PRIMARY KEY,
    "Source Index" TEXT UNIQUE,
    "Name" TEXT,
    "Summary" TEXT,
    "First Launch Date" TEXT,
    "Developers" TEXT,
    "Publishers" TEXT,
    "Genres" TEXT,
    "Game Modes" TEXT,
    "Cover Path" TEXT,
    "Width" INTEGER,
    "Height" INTEGER
)'''

def as_int(value):
    try:
        return int(float(value))
    except Exception:
        return 0

def migrate():
    if not os.path.exists(PROCESSED_XLSX):
        print(f'{PROCESSED_XLSX} not found.')
        return
    conn = sqlite3.connect(PROCESSED_DB)
    conn.row_factory = sqlite3.Row
    with conn:
        conn.execute(SCHEMA)
    df = pd.read_excel(PROCESSED_XLSX, dtype=str).fillna('')

    # Ensure rows are ordered by numeric ID and reassign IDs/Source Index sequentially
    if 'ID' in df.columns:
        df['__id_int'] = df['ID'].apply(as_int)
        df = df.sort_values('__id_int').drop(columns=['__id_int']).reset_index(drop=True)
    df['ID'] = df.index.map(lambda i: f"{i + 1:07d}")
    df['Source Index'] = df.index.astype(str)

    with conn:
        for _, r in df.iterrows():
            conn.execute(
                '''INSERT OR REPLACE INTO processed_games (
                    "ID","Source Index","Name","Summary",
                    "First Launch Date","Developers","Publishers",
                    "Genres","Game Modes","Cover Path","Width","Height"
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)''',
                (
                    r.get('ID', ''),
                    r.get('Source Index', ''),
                    r.get('Name', ''),
                    r.get('Summary', ''),
                    r.get('First Launch Date', ''),
                    r.get('Developers', ''),
                    r.get('Publishers', ''),
                    r.get('Genres', ''),
                    r.get('Game Modes', ''),
                    r.get('Cover Path', ''),
                    as_int(r.get('Width', 0)),
                    as_int(r.get('Height', 0)),
                ),
            )
    conn.close()
    print('Migration complete.')

if __name__ == '__main__':
    migrate()
