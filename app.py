import os
import json
import uuid
import base64
import io
import sqlite3
import numbers
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from threading import Lock
import logging

from flask import (
    Flask,
    request,
    jsonify,
    render_template,
    session,
    redirect,
    url_for,
    g,
    has_app_context,
)
from werkzeug.exceptions import RequestEntityTooLarge
from PIL import Image, ExifTags
import pandas as pd
from openai import OpenAI
from urllib.parse import urlparse, urlencode
from urllib.request import urlopen, Request
from urllib.error import HTTPError

logger = logging.getLogger(__name__)

INPUT_XLSX = 'igdb_all_games.xlsx'
PROCESSED_DB = 'processed_games.db'
UPLOAD_DIR = 'uploaded_sources'
PROCESSED_DIR = 'processed_covers'
COVERS_DIR = 'covers_out'

BASE_DIR = Path(__file__).resolve().parent
LOOKUP_DATA_DIR = Path(os.environ.get('LOOKUP_DATA_DIR', BASE_DIR))
LOOKUP_TABLES = (
    {
        'table': 'developers',
        'column': 'Developer',
        'filename': 'Developers_unique.xlsx',
    },
    {
        'table': 'publishers',
        'column': 'Publisher',
        'filename': 'Publishers_unique.xlsx',
    },
    {
        'table': 'genres',
        'column': 'Genre',
        'filename': 'Genres_unique.xlsx',
    },
    {
        'table': 'game_modes',
        'column': 'GameMode',
        'filename': 'GameModes_unique.xlsx',
    },
    {
        'table': 'platforms',
        'column': 'Platform',
        'filename': 'Platforms_unique.xlsx',
    },
)
LOOKUP_RELATIONS = (
    {
        'processed_column': 'Developers',
        'lookup_table': 'developers',
        'response_key': 'Developers',
        'id_column': 'developers_ids',
        'join_table': 'processed_game_developers',
        'join_column': 'developer_id',
    },
    {
        'processed_column': 'Publishers',
        'lookup_table': 'publishers',
        'response_key': 'Publishers',
        'id_column': 'publishers_ids',
        'join_table': 'processed_game_publishers',
        'join_column': 'publisher_id',
    },
    {
        'processed_column': 'Genres',
        'lookup_table': 'genres',
        'response_key': 'Genres',
        'id_column': 'genres_ids',
        'join_table': 'processed_game_genres',
        'join_column': 'genre_id',
    },
    {
        'processed_column': 'Game Modes',
        'lookup_table': 'game_modes',
        'response_key': 'GameModes',
        'id_column': 'game_modes_ids',
        'join_table': 'processed_game_game_modes',
        'join_column': 'game_mode_id',
    },
    {
        'processed_column': 'Platforms',
        'lookup_table': 'platforms',
        'response_key': 'Platforms',
        'id_column': 'platforms_ids',
        'join_table': 'processed_game_platforms',
        'join_column': 'platform_id',
    },
)

LOOKUP_RELATIONS_BY_COLUMN = {
    relation['processed_column']: relation for relation in LOOKUP_RELATIONS
}

LOOKUP_RELATIONS_BY_KEY = {
    relation['response_key']: relation for relation in LOOKUP_RELATIONS
}

LOOKUP_TABLES_BY_NAME = {
    table_config['table']: table_config for table_config in LOOKUP_TABLES
}

LOOKUP_ENDPOINT_MAP = {
    'developers': 'developers',
    'publishers': 'publishers',
    'genres': 'genres',
    'game_modes': 'game_modes',
    'game-modes': 'game_modes',
    'gamemodes': 'game_modes',
    'platforms': 'platforms',
}

LOOKUP_SOURCE_KEYS = {
    'Developers': ['Developers', 'Developer'],
    'Publishers': ['Publishers', 'Publisher'],
    'Genres': ['Genres', 'Genre'],
    'Game Modes': ['Game Modes', 'Mode'],
    'Platforms': ['Platforms', 'Platform'],
}

app = Flask(__name__)
app.secret_key = os.environ.get('APP_SECRET_KEY', 'dev-secret')
APP_PASSWORD = os.environ.get('APP_PASSWORD', 'password')
DEFAULT_IGDB_USER_AGENT = 'TT-Game-Liste/1.0 (support@example.com)'
IGDB_USER_AGENT = os.environ.get('IGDB_USER_AGENT') or DEFAULT_IGDB_USER_AGENT
IGDB_BATCH_SIZE = 500
logging.basicConfig(level=logging.DEBUG)
app.logger.setLevel(logging.DEBUG)
logger.setLevel(logging.DEBUG)
# Configure OpenAI using API key from environment
client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY', ''))

# SQLite setup for processed games
db_lock = Lock()


def get_db():
    if not has_app_context():
        return db
    if 'db' not in g:
        g.db = sqlite3.connect(PROCESSED_DB)
        g.db.row_factory = sqlite3.Row
    return g.db


def _quote_identifier(identifier: str) -> str:
    return '"' + str(identifier).replace('"', '""') + '"'


def _parse_iterable(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [v.strip() for v in value.split(',') if v.strip()]
    if isinstance(value, numbers.Number):
        return [str(value)]
    items: list[str] = []
    try:
        iterator = iter(value)
    except TypeError:
        return [str(value)]
    for element in iterator:
        if isinstance(element, Mapping):
            name = element.get('name')
            if isinstance(name, str) and name.strip():
                items.append(name.strip())
            else:
                items.append(str(element).strip())
        else:
            items.append(str(element).strip())
    return [item for item in items if item]


def _parse_company_names(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        cleaned = value.strip()
        return [cleaned] if cleaned else []
    if isinstance(value, numbers.Number):
        return [str(value)]
    names: list[str] = []
    try:
        iterator = iter(value)
    except TypeError:
        text = str(value).strip()
        return [text] if text else []
    for element in iterator:
        name_value: Any = None
        if isinstance(element, Mapping):
            if isinstance(element.get('name'), str):
                name_value = element['name']
            else:
                company_obj = element.get('company')
                if isinstance(company_obj, Mapping) and isinstance(
                    company_obj.get('name'), str
                ):
                    name_value = company_obj['name']
                elif isinstance(company_obj, str):
                    name_value = company_obj
        else:
            name_value = element
        text = _normalize_lookup_name(name_value)
        if text:
            names.append(text)
    return names


def _normalize_lookup_name(value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, str):
        return value.strip()
    try:
        if pd.isna(value):
            return ''
    except Exception:
        pass
    return str(value).strip()


def _row_value(row: sqlite3.Row | tuple[Any, ...], key: str, index: int) -> Any:
    if isinstance(row, sqlite3.Row):
        return row[key]
    return row[index]


def _decode_lookup_id_list(raw_value: Any) -> list[int]:
    if raw_value is None:
        return []

    candidates: list[Any] = []
    if isinstance(raw_value, str):
        stripped = raw_value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            candidates.extend(part.strip() for part in stripped.split(',') if part.strip())
        else:
            if isinstance(parsed, list):
                candidates.extend(parsed)
            else:
                candidates.append(parsed)
    elif isinstance(raw_value, numbers.Number):
        candidates.append(raw_value)
    elif isinstance(raw_value, (list, tuple, set)):
        candidates.extend(raw_value)
    else:
        candidates.append(raw_value)

    normalized: list[int] = []
    seen: set[int] = set()
    for candidate in candidates:
        if isinstance(candidate, str):
            try:
                coerced = int(candidate.strip())
            except (TypeError, ValueError):
                continue
        else:
            try:
                coerced = int(candidate)
            except (TypeError, ValueError):
                continue
        if coerced in seen:
            continue
        seen.add(coerced)
        normalized.append(coerced)
    return normalized


def _fetch_lookup_entries_for_game(
    conn: sqlite3.Connection, processed_game_id: int
) -> dict[str, list[dict[str, Any]]]:
    selections: dict[str, list[dict[str, Any]]] = {}
    try:
        cur = conn.execute(
            'SELECT * FROM processed_games WHERE "ID"=?',
            (processed_game_id,),
        )
        processed_row = cur.fetchone()
    except sqlite3.OperationalError:
        processed_row = None

    processed_mapping = dict(processed_row) if processed_row is not None else {}
    for relation in LOOKUP_RELATIONS:
        response_key = relation['response_key']
        lookup_table = relation['lookup_table']
        join_table = relation['join_table']
        join_column = relation['join_column']
        entries: list[dict[str, Any]] = []
        join_rows: list[sqlite3.Row | tuple[Any, ...]] = []
        try:
            cur_lookup = conn.execute(
                f'''
                    SELECT j.{join_column} AS lookup_id, l.name
                    FROM {join_table} j
                    LEFT JOIN {lookup_table} l ON l.id = j.{join_column}
                    WHERE j.processed_game_id = ?
                    ORDER BY j.rowid
                ''',
                (processed_game_id,),
            )
            join_rows = cur_lookup.fetchall()
        except sqlite3.OperationalError:
            join_rows = []

        seen_ids: set[int] = set()
        for join_row in join_rows:
            raw_id = _row_value(join_row, 'lookup_id', 0)
            try:
                lookup_id = int(raw_id)
            except (TypeError, ValueError):
                continue
            if lookup_id in seen_ids:
                continue
            seen_ids.add(lookup_id)
            lookup_name = _normalize_lookup_name(_row_value(join_row, 'name', 1))
            if not lookup_name:
                lookup_name = _lookup_name_for_id(conn, lookup_table, lookup_id)
            normalized_name = _normalize_lookup_name(lookup_name)
            entries.append({'id': lookup_id, 'name': normalized_name})
        if not entries and processed_mapping:
            raw_value = processed_mapping.get(relation['processed_column'])
            for name in _parse_iterable(raw_value):
                normalized = _normalize_lookup_name(name)
                if normalized:
                    entries.append({'id': None, 'name': normalized})
        selections[response_key] = entries
    return selections


def _parse_lookup_entries_from_source(
    source_row: pd.Series | Mapping[str, Any], processed_column: str
) -> list[dict[str, Any]]:
    keys = LOOKUP_SOURCE_KEYS.get(processed_column, [processed_column])
    values: list[str] = []
    for key in keys:
        try:
            if isinstance(source_row, Mapping):
                value = source_row.get(key)
            else:
                value = source_row[key] if key in source_row else None
        except Exception:
            value = None
        if value is None:
            continue
        values.extend(_parse_iterable(value))
        if values:
            break
    entries: list[dict[str, Any]] = []
    for name in values:
        normalized = _normalize_lookup_name(name)
        if normalized:
            entries.append({'id': None, 'name': normalized})
    return entries


def _format_lookup_response(entries: list[dict[str, Any]]) -> dict[str, Any]:
    formatted: list[dict[str, Any]] = []
    names: list[str] = []
    ids: list[int] = []
    seen_ids: set[int] = set()
    seen_names: set[str] = set()
    for entry in entries:
        entry_id = entry.get('id')
        if entry_id is not None:
            try:
                entry_id = int(entry_id)
            except (TypeError, ValueError):
                entry_id = None
        entry_name = _normalize_lookup_name(entry.get('name'))
        if entry_id is None and not entry_name:
            continue
        if entry_id is not None:
            if entry_id in seen_ids:
                continue
            seen_ids.add(entry_id)
            ids.append(entry_id)
        else:
            fingerprint = entry_name.casefold()
            if fingerprint in seen_names:
                continue
            seen_names.add(fingerprint)
        if entry_name:
            names.append(entry_name)
        formatted.append({'id': entry_id, 'name': entry_name})
    return {
        'selected': formatted,
        'names': names,
        'ids': ids,
    }


def _lookup_display_text(names: list[str]) -> str:
    return ', '.join(name for name in names if name)


def _lookup_name_for_id(
    conn: sqlite3.Connection, table_name: str, lookup_id: int
) -> str:
    cur = conn.execute(
        f'SELECT name FROM {table_name} WHERE id = ?',
        (lookup_id,),
    )
    row = cur.fetchone()
    if row is None:
        return ''
    return _normalize_lookup_name(_row_value(row, 'name', 0))


def _iter_lookup_payload(raw_value: Any) -> list[dict[str, Any]]:
    if raw_value is None:
        return []
    if isinstance(raw_value, Mapping):
        if 'selected' in raw_value:
            return _iter_lookup_payload(raw_value['selected'])
        if 'entries' in raw_value:
            return _iter_lookup_payload(raw_value['entries'])
        ids_value = raw_value.get('ids')
        names_value = raw_value.get('names')
        results: list[dict[str, Any]] = []
        if isinstance(ids_value, (list, tuple)):
            for idx, entry_id in enumerate(ids_value):
                entry_name = None
                if isinstance(names_value, (list, tuple)) and idx < len(names_value):
                    entry_name = names_value[idx]
                results.append({'id': entry_id, 'name': entry_name})
            return results
        if isinstance(names_value, (list, tuple)):
            for name in names_value:
                results.append({'id': None, 'name': name})
            return results
        entry_id = raw_value.get('id')
        entry_name = raw_value.get('name')
        if entry_id is not None or entry_name is not None:
            return [{'id': entry_id, 'name': entry_name}]
        return []
    if isinstance(raw_value, str):
        text = _normalize_lookup_name(raw_value)
        return [{'id': None, 'name': text}] if text else []
    if isinstance(raw_value, numbers.Number):
        try:
            return [{'id': int(raw_value), 'name': None}]
        except (TypeError, ValueError):
            return []
    try:
        iterator = iter(raw_value)
    except TypeError:
        text = _normalize_lookup_name(raw_value)
        return [{'id': None, 'name': text}] if text else []
    results: list[dict[str, Any]] = []
    for item in iterator:
        results.extend(_iter_lookup_payload(item))
    return results


def _resolve_lookup_selection(
    conn: sqlite3.Connection,
    relation: Mapping[str, Any],
    raw_value: Any,
) -> dict[str, Any]:
    lookup_table = relation['lookup_table']
    entries = _iter_lookup_payload(raw_value)
    resolved: list[dict[str, Any]] = []
    for entry in entries:
        entry_id = entry.get('id')
        entry_name = _normalize_lookup_name(entry.get('name'))
        coerced_id: int | None = None
        if entry_id is not None:
            try:
                coerced_id = int(entry_id)
            except (TypeError, ValueError):
                coerced_id = None
        if coerced_id is not None:
            lookup_name = _lookup_name_for_id(conn, lookup_table, coerced_id)
            if lookup_name:
                resolved.append({'id': coerced_id, 'name': lookup_name})
                continue
            coerced_id = None
        if not entry_name:
            continue
        lookup_id = _get_or_create_lookup_id(conn, lookup_table, entry_name)
        if lookup_id is None:
            continue
        lookup_name = _lookup_name_for_id(conn, lookup_table, lookup_id) or entry_name
        resolved.append({'id': lookup_id, 'name': lookup_name})
    deduped: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    for entry in resolved:
        entry_id = entry['id']
        if entry_id in seen_ids:
            continue
        seen_ids.add(entry_id)
        deduped.append(entry)
    names = [entry['name'] for entry in deduped if entry['name']]
    ids: list[int] = []
    for entry in deduped:
        entry_id = entry.get('id')
        if entry_id is None:
            continue
        try:
            coerced = int(entry_id)
        except (TypeError, ValueError):
            continue
        ids.append(coerced)
    return {'entries': deduped, 'names': names, 'ids': ids}


def _load_lookup_tables(conn: sqlite3.Connection) -> None:
    for table_config in LOOKUP_TABLES:
        path = LOOKUP_DATA_DIR / table_config['filename']
        if not path.exists():
            continue
        try:
            df = pd.read_excel(path)
        except Exception:
            logger.exception('Failed to load lookup workbook %s', path)
            continue
        column_name = table_config['column']
        if column_name not in df.columns:
            logger.warning(
                'Workbook %s missing expected column %s', path, column_name
            )
            continue
        series = df[column_name].dropna()
        for raw_value in series:
            name = _normalize_lookup_name(raw_value)
            if not name:
                continue
            conn.execute(
                f'''
                INSERT INTO {table_config['table']} (name)
                VALUES (?)
                ON CONFLICT(name) DO UPDATE SET name=excluded.name
                ''',
                (name,),
            )


def _get_or_create_lookup_id(
    conn: sqlite3.Connection, table_name: str, raw_name: Any
) -> int | None:
    name = _normalize_lookup_name(raw_name)
    if not name:
        return None
    cur = conn.execute(
        f'SELECT id FROM {table_name} WHERE name = ? COLLATE NOCASE', (name,)
    )
    row = cur.fetchone()
    if row is not None:
        return row[0]
    cur = conn.execute(
        f'INSERT INTO {table_name} (name) VALUES (?)',
        (name,),
    )
    return cur.lastrowid


def _recreate_lookup_join_tables(conn: sqlite3.Connection) -> None:
    for relation in LOOKUP_RELATIONS:
        join_table = relation['join_table']
        join_column = relation['join_column']
        lookup_table = relation['lookup_table']
        try:
            conn.execute(f'DROP TABLE IF EXISTS {join_table}')
            conn.execute(
                f'''
                    CREATE TABLE {join_table} (
                        processed_game_id INTEGER NOT NULL,
                        {join_column} INTEGER NOT NULL,
                        PRIMARY KEY (processed_game_id, {join_column}),
                        FOREIGN KEY(processed_game_id)
                            REFERENCES processed_games("ID") ON DELETE CASCADE,
                        FOREIGN KEY({join_column})
                            REFERENCES {lookup_table}(id) ON DELETE CASCADE
                    )
                '''
            )
            conn.execute(
                f'''
                    CREATE INDEX IF NOT EXISTS {join_table}_processed_game_idx
                    ON {join_table} (processed_game_id)
                '''
            )
        except sqlite3.OperationalError:
            logger.exception('Failed to recreate join table %s', join_table)


def _drop_lookup_id_columns(conn: sqlite3.Connection) -> None:
    try:
        cur = conn.execute('PRAGMA table_info(processed_games)')
    except sqlite3.OperationalError:
        return

    rows = cur.fetchall()
    existing_columns = {row[1] for row in rows}
    legacy_columns = {
        relation['id_column']
        for relation in LOOKUP_RELATIONS
        if relation.get('id_column')
    }
    if not (existing_columns & legacy_columns):
        return

    desired_columns: list[tuple[str, str]] = [
        ('ID', 'INTEGER PRIMARY KEY'),
        ('Source Index', 'TEXT UNIQUE'),
        ('Name', 'TEXT'),
        ('Summary', 'TEXT'),
        ('First Launch Date', 'TEXT'),
        ('Developers', 'TEXT'),
        ('Publishers', 'TEXT'),
        ('Genres', 'TEXT'),
        ('Game Modes', 'TEXT'),
        ('Category', 'TEXT'),
        ('Platforms', 'TEXT'),
        ('igdb_id', 'TEXT'),
        ('Cover Path', 'TEXT'),
        ('Width', 'INTEGER'),
        ('Height', 'INTEGER'),
        ('last_edited_at', 'TEXT'),
    ]

    column_defs = ',\n                    '.join(
        f'{_quote_identifier(name)} {col_type}' for name, col_type in desired_columns
    )
    insert_columns = [
        _quote_identifier(name) for name, _ in desired_columns
    ]
    select_columns: list[str] = []
    for name, _ in desired_columns:
        if name in existing_columns:
            select_columns.append(_quote_identifier(name))
        else:
            select_columns.append('NULL')

    insert_sql = ', '.join(insert_columns)
    select_sql = ', '.join(select_columns)

    try:
        conn.execute('PRAGMA foreign_keys = OFF')
        conn.executescript(
            f'''
                ALTER TABLE processed_games RENAME TO processed_games_old;
                CREATE TABLE processed_games (
                    {column_defs}
                );
                INSERT INTO processed_games ({insert_sql})
                SELECT {select_sql} FROM processed_games_old;
                DROP TABLE processed_games_old;
            '''
        )
    finally:
        conn.execute('PRAGMA foreign_keys = ON')


def _replace_lookup_relations(
    conn: sqlite3.Connection,
    join_table: str,
    join_column: str,
    processed_game_id: int,
    lookup_ids: list[int],
) -> None:
    try:
        conn.execute(
            f'DELETE FROM {join_table} WHERE processed_game_id = ?',
            (processed_game_id,),
        )
    except sqlite3.OperationalError:
        return
    to_insert: list[tuple[int, int]] = []
    for value in lookup_ids:
        try:
            coerced = int(value)
        except (TypeError, ValueError):
            continue
        to_insert.append((processed_game_id, coerced))
    if not to_insert:
        return
    try:
        conn.executemany(
            f'INSERT OR IGNORE INTO {join_table} '
            f'(processed_game_id, {join_column}) VALUES (?, ?)',
            to_insert,
        )
    except sqlite3.OperationalError:
        return


def _persist_lookup_relations(
    conn: sqlite3.Connection,
    processed_game_id: int,
    selections: Mapping[str, Mapping[str, Any]] | Mapping[str, Any],
) -> None:
    for relation in LOOKUP_RELATIONS:
        response_key = relation['response_key']
        join_table = relation['join_table']
        join_column = relation['join_column']
        ids: list[int] = []
        selection: Any = None
        if isinstance(selections, Mapping):
            selection = selections.get(response_key)
        if isinstance(selection, Mapping):
            raw_ids = selection.get('ids')
            if isinstance(raw_ids, (list, tuple)):
                ids = [value for value in raw_ids if value is not None]
        _replace_lookup_relations(conn, join_table, join_column, processed_game_id, ids)


def _backfill_lookup_relations(conn: sqlite3.Connection) -> None:
    try:
        cur = conn.execute('SELECT * FROM processed_games')
    except sqlite3.OperationalError:
        return

    rows = cur.fetchall()
    column_names = [desc[0] for desc in cur.description] if cur.description else []

    for row in rows:
        if isinstance(row, sqlite3.Row):
            row_dict = dict(row)
        else:
            row_dict = {
                column_names[idx]: value
                for idx, value in enumerate(row)
                if idx < len(column_names)
            }
        try:
            game_id = int(row_dict.get('ID'))
        except (TypeError, ValueError):
            continue

        for relation in LOOKUP_RELATIONS:
            lookup_table = relation['lookup_table']
            processed_column = relation['processed_column']
            join_table = relation['join_table']
            join_column = relation['join_column']
            id_column = relation.get('id_column')

            existing_ids: list[int] = []
            try:
                cur_existing = conn.execute(
                    f'SELECT {join_column} FROM {join_table} '
                    'WHERE processed_game_id = ? ORDER BY rowid',
                    (game_id,),
                )
            except sqlite3.OperationalError:
                existing_ids = []
            else:
                for existing_row in cur_existing.fetchall():
                    try:
                        existing_id = int(
                            _row_value(existing_row, join_column, 0)
                        )
                    except (TypeError, ValueError):
                        continue
                    existing_ids.append(existing_id)

            candidate_ids: list[int] = list(existing_ids)
            if not candidate_ids and id_column in row_dict:
                candidate_ids.extend(
                    _decode_lookup_id_list(row_dict.get(id_column))
                )
            if not candidate_ids:
                raw_value = row_dict.get(processed_column)
                names = _parse_iterable(raw_value)
                seen_names: set[str] = set()
                for name in names:
                    normalized = _normalize_lookup_name(name)
                    if not normalized:
                        continue
                    fingerprint = normalized.casefold()
                    if fingerprint in seen_names:
                        continue
                    seen_names.add(fingerprint)
                    lookup_id = _get_or_create_lookup_id(
                        conn, lookup_table, normalized
                    )
                    if lookup_id is None:
                        continue
                    candidate_ids.append(int(lookup_id))

            deduped_ids: list[int] = []
            seen_ids: set[int] = set()
            for value in candidate_ids:
                try:
                    coerced = int(value)
                except (TypeError, ValueError):
                    continue
                if coerced in seen_ids:
                    continue
                seen_ids.add(coerced)
                deduped_ids.append(coerced)

            if existing_ids and deduped_ids == existing_ids:
                continue

            _replace_lookup_relations(
                conn, join_table, join_column, game_id, deduped_ids
            )


def _migrate_id_column(conn: sqlite3.Connection) -> None:
    cur = conn.execute("PRAGMA table_info(processed_games)")
    cols = cur.fetchall()
    id_col = next((c for c in cols if c[1] == "ID"), None)
    if id_col and id_col[2].upper() != "INTEGER":
        conn.executescript(
            '''
            ALTER TABLE processed_games RENAME TO processed_games_old;
            CREATE TABLE processed_games (
                "ID" INTEGER PRIMARY KEY,
                "Source Index" TEXT UNIQUE,
                "Name" TEXT,
                "Summary" TEXT,
                "First Launch Date" TEXT,
                "Developers" TEXT,
                "Publishers" TEXT,
                "Genres" TEXT,
                "Game Modes" TEXT,
                "Category" TEXT,
                "Platforms" TEXT,
                "igdb_id" TEXT,
                "Cover Path" TEXT,
                "Width" INTEGER,
                "Height" INTEGER
            );
            INSERT INTO processed_games (
                "ID", "Source Index", "Name", "Summary", "First Launch Date",
                "Developers", "Publishers", "Genres", "Game Modes", "Category",
                "Platforms", "igdb_id", "Cover Path", "Width", "Height"
            )
            SELECT CAST("ID" AS INTEGER), "Source Index", "Name", "Summary",
                   "First Launch Date", "Developers", "Publishers", "Genres",
                   "Game Modes", '', '', '', "Cover Path", "Width", "Height"
            FROM processed_games_old;
            DROP TABLE processed_games_old;
            '''
        )


def _init_db() -> None:
    conn = sqlite3.connect(PROCESSED_DB)
    try:
        with conn:
            conn.execute('PRAGMA foreign_keys = ON')
            conn.execute(
                '''CREATE TABLE IF NOT EXISTS processed_games (
                    "ID" INTEGER PRIMARY KEY,
                    "Source Index" TEXT UNIQUE,
                    "Name" TEXT,
                    "Summary" TEXT,
                    "First Launch Date" TEXT,
                    "Developers" TEXT,
                    "Publishers" TEXT,
                    "Genres" TEXT,
                    "Game Modes" TEXT,
                    "Category" TEXT,
                    "Platforms" TEXT,
                    "igdb_id" TEXT,
                    "Cover Path" TEXT,
                    "Width" INTEGER,
                    "Height" INTEGER,
                    last_edited_at TEXT
                )'''
            )
            conn.execute(
                '''CREATE TABLE IF NOT EXISTS navigator_state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    current_index INTEGER,
                    seq_index INTEGER,
                    skip_queue TEXT
                )'''
            )
            _migrate_id_column(conn)
            try:
                conn.execute('ALTER TABLE processed_games ADD COLUMN "Category" TEXT')
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute('ALTER TABLE processed_games ADD COLUMN "Platforms" TEXT')
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute('ALTER TABLE processed_games ADD COLUMN "igdb_id" TEXT')
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute('ALTER TABLE processed_games ADD COLUMN last_edited_at TEXT')
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute(
                    '''CREATE TABLE IF NOT EXISTS igdb_updates (
                        processed_game_id INTEGER PRIMARY KEY,
                        igdb_id TEXT,
                        igdb_updated_at TEXT,
                        igdb_payload TEXT,
                        diff TEXT,
                        local_last_edited_at TEXT,
                        refreshed_at TEXT,
                        FOREIGN KEY(processed_game_id) REFERENCES processed_games("ID")
                    )'''
                )
            except sqlite3.OperationalError:
                pass

            for table_config in LOOKUP_TABLES:
                conn.execute(
                    f'''
                    CREATE TABLE IF NOT EXISTS {table_config['table']} (
                        id INTEGER PRIMARY KEY,
                        name TEXT NOT NULL COLLATE NOCASE UNIQUE
                    )
                    '''
                )

            _load_lookup_tables(conn)
            _recreate_lookup_join_tables(conn)
            _backfill_lookup_relations(conn)
            _drop_lookup_id_columns(conn)

            cur = conn.execute('SELECT "ID", last_edited_at FROM processed_games')
            rows = cur.fetchall()
            for game_id, last_edit in rows:
                if not last_edit:
                    conn.execute(
                        'UPDATE processed_games SET last_edited_at=? WHERE "ID"=?',
                        (
                            datetime.now(timezone.utc).isoformat(),
                            game_id,
                        ),
                    )

            cur = conn.execute(
                'SELECT "ID", "igdb_id", last_edited_at FROM processed_games'
            )
            for game_id, igdb_id_value, last_edit in cur.fetchall():
                if not igdb_id_value:
                    continue
                conn.execute(
                    '''INSERT OR IGNORE INTO igdb_updates (
                        processed_game_id, igdb_id, local_last_edited_at
                    ) VALUES (?, ?, ?)''',
                    (
                        game_id,
                        str(igdb_id_value),
                        last_edit,
                    ),
                )
    finally:
        conn.close()


_init_db()

# Expose a module-level connection for tests and utilities
db = sqlite3.connect(PROCESSED_DB)
db.row_factory = sqlite3.Row

@app.teardown_appcontext
def close_db(exc):
    db = g.pop('db', None)
    if db is not None:
        db.close()


@app.errorhandler(Exception)
def handle_exception(e):
    app.logger.exception("Unhandled exception")
    if request.path.startswith('/api/'):
        return jsonify({'error': 'internal server error'}), 500
    return "Internal Server Error", 500


@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    app.logger.warning("File upload too large for path %s", request.path)
    return jsonify({'error': 'file too large'}), 413


def open_image_auto_rotate(source: Any) -> Image.Image:
    """Open image from path or file-like and auto-rotate using EXIF."""
    img = Image.open(source) if not isinstance(source, Image.Image) else source
    try:
        exif = img._getexif()
        if exif:
            orientation_tag = next(
                k for k, v in ExifTags.TAGS.items() if v == 'Orientation'
            )
            orientation = exif.get(orientation_tag)
            if orientation == 3:
                img = img.rotate(180, expand=True)
            elif orientation == 6:
                img = img.rotate(270, expand=True)
            elif orientation == 8:
                img = img.rotate(90, expand=True)
    except Exception:
        pass
    return img.convert('RGB')


def load_games() -> pd.DataFrame:
    if not os.path.exists(INPUT_XLSX):
        logger.warning("%s not found", INPUT_XLSX)
        return pd.DataFrame()
    try:
        df = pd.read_excel(INPUT_XLSX)
    except Exception:
        logger.exception("Failed to read %s", INPUT_XLSX)
        return pd.DataFrame()
    df = df.dropna(how='all')
    if 'Name' in df.columns:
        df = df[df['Name'].notna()]
    if 'Rating Count' in df.columns:
        df['Rating Count'] = pd.to_numeric(df['Rating Count'], errors='coerce').fillna(0)
        df = df.sort_values(by='Rating Count', ascending=False, kind='mergesort')
    df = df.reset_index(drop=True)
    return df




def find_cover(row: pd.Series) -> str | None:
    url = str(row.get('Large Cover Image (URL)', ''))
    if not url:
        return None
    parsed_path = urlparse(url).path
    base = os.path.splitext(os.path.basename(parsed_path))[0]
    for ext in ('.jpg', '.jpeg', '.png'):
        path = os.path.join(COVERS_DIR, base + ext)
        if os.path.exists(path):
            img = open_image_auto_rotate(path)
            buf = io.BytesIO()
            img.save(buf, format='JPEG')
            return 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()
    try:
        with urlopen(url) as resp:
            img = open_image_auto_rotate(resp)
            buf = io.BytesIO()
            img.save(buf, format='JPEG')
            return 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        app.logger.warning("No cover found for URL %s", url)
    return None


def ensure_dirs() -> None:
    for d in (UPLOAD_DIR, PROCESSED_DIR, COVERS_DIR):
        os.makedirs(d, exist_ok=True)


def normalize_processed_games() -> None:
    """Resequence IDs based on the order of ``Source Index``."""
    with db_lock:
        conn = get_db()
        with conn:
            cur = conn.execute(
                'SELECT "Source Index" FROM processed_games '
                'ORDER BY CAST("Source Index" AS INTEGER)'
            )
            rows = [r["Source Index"] for r in cur.fetchall()]
            for new_id, src_index in enumerate(rows, start=1):
                conn.execute(
                    'UPDATE processed_games SET "ID"=? WHERE "Source Index"=?',
                    (-new_id, src_index),
                )
            conn.execute('UPDATE processed_games SET "ID" = -"ID"')


def backfill_igdb_ids() -> None:
    if 'games_df' not in globals():
        return
    if games_df.empty:
        return
    with db_lock:
        conn = get_db()
        with conn:
            cur = conn.execute(
                'SELECT "Source Index", "igdb_id" FROM processed_games'
            )
            rows = cur.fetchall()
            for row in rows:
                igdb_id_value = row['igdb_id']
                if igdb_id_value:
                    continue
                src_index = row['Source Index']
                try:
                    idx = int(str(src_index))
                except (TypeError, ValueError):
                    continue
                if idx < 0 or idx >= len(games_df):
                    continue
                candidate = extract_igdb_id(
                    games_df.iloc[idx], allow_generic_id=True
                )
                if candidate:
                    conn.execute(
                        'UPDATE processed_games SET "igdb_id"=? WHERE "Source Index"=?',
                        (candidate, src_index),
                    )


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def exchange_twitch_credentials() -> tuple[str, str]:
    client_id = os.environ.get('TWITCH_CLIENT_ID')
    client_secret = os.environ.get('TWITCH_CLIENT_SECRET')
    if not client_id or not client_secret:
        raise RuntimeError('missing twitch client credentials')

    payload = urlencode(
        {
            'client_id': client_id,
            'client_secret': client_secret,
            'grant_type': 'client_credentials',
        }
    ).encode('utf-8')

    request = Request(
        'https://id.twitch.tv/oauth2/token',
        data=payload,
        method='POST',
    )
    request.add_header('Content-Type', 'application/x-www-form-urlencoded')

    try:
        with urlopen(request) as response:
            data = json.loads(response.read().decode('utf-8'))
    except Exception as exc:  # pragma: no cover - network failures surfaced
        raise RuntimeError(f'failed to obtain twitch token: {exc}') from exc

    token = data.get('access_token')
    if not token:
        raise RuntimeError('missing access token in twitch response')
    return token, client_id


def fetch_igdb_metadata(
    access_token: str, client_id: str, igdb_ids: list[str]
) -> dict[str, dict[str, Any]]:
    if not igdb_ids:
        return {}

    numeric_ids: list[int] = []
    for value in igdb_ids:
        try:
            numeric_ids.append(int(str(value).strip()))
        except (TypeError, ValueError):
            logger.warning('Skipping invalid IGDB id %s', value)
    if not numeric_ids:
        return {}

    results: dict[str, dict[str, Any]] = {}
    batch_size = IGDB_BATCH_SIZE if isinstance(IGDB_BATCH_SIZE, int) and IGDB_BATCH_SIZE > 0 else 500
    for start in range(0, len(numeric_ids), batch_size):
        chunk = numeric_ids[start : start + batch_size]
        if not chunk:
            continue
        query = (
            'fields '
            'id,name,summary,updated_at,first_release_date,'
            'genres,platforms,game_modes,category,'
            'involved_companies.company.name,'
            'involved_companies.developer,'
            'involved_companies.publisher; '
            f"where id = ({', '.join(str(v) for v in chunk)}); "
            f'limit {len(chunk)};'
        )
        request = Request(
            'https://api.igdb.com/v4/games',
            data=query.encode('utf-8'),
            method='POST',
        )
        request.add_header('Client-ID', client_id)
        request.add_header('Authorization', f'Bearer {access_token}')
        request.add_header('Accept', 'application/json')
        request.add_header('User-Agent', IGDB_USER_AGENT)

        try:
            with urlopen(request) as response:
                payload = json.loads(response.read().decode('utf-8'))
        except HTTPError as exc:
            error_message = ''
            try:
                error_body = exc.read()
            except Exception:  # pragma: no cover - best effort to capture error body
                error_body = b''
            if error_body:
                try:
                    error_message = error_body.decode('utf-8', errors='replace').strip()
                except Exception:  # pragma: no cover - unexpected decoding failures
                    error_message = ''
            if not error_message and exc.reason:
                error_message = str(exc.reason)
            message = f"IGDB request failed: {exc.code}"
            if error_message:
                message = f"{message} {error_message}"
            raise RuntimeError(message) from exc
        except Exception as exc:  # pragma: no cover - network failures surfaced
            logger.warning('Failed to query IGDB: %s', exc)
            return {}

        for item in payload or []:
            if not isinstance(item, dict):
                continue
            igdb_id = item.get('id')
            if igdb_id is None:
                continue
            parsed_item = dict(item)
            involved_companies = item.get('involved_companies')
            developer_names: list[str] = []
            publisher_names: list[str] = []
            if isinstance(involved_companies, list):
                for company in involved_companies:
                    if not isinstance(company, Mapping):
                        continue
                    company_obj = company.get('company')
                    company_name: str | None = None
                    if isinstance(company_obj, Mapping):
                        name_value = company_obj.get('name')
                        if isinstance(name_value, str):
                            company_name = name_value.strip()
                    elif isinstance(company_obj, str):
                        company_name = company_obj.strip()
                    if not company_name:
                        continue
                    if company.get('developer'):
                        developer_names.append(company_name)
                    if company.get('publisher'):
                        publisher_names.append(company_name)
            parsed_item['developers'] = developer_names
            parsed_item['publishers'] = publisher_names
            results[str(igdb_id)] = parsed_item
    return results


def _normalize_text(value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, numbers.Number):
        return str(value)
    return str(value).strip()


def _normalize_timestamp(value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, numbers.Number):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc).date().isoformat()
        except Exception:
            return ''
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return ''
        try:
            return datetime.fromtimestamp(float(stripped), tz=timezone.utc).date().isoformat()
        except Exception:
            return stripped
    return _normalize_text(value)


IGDB_DIFF_FIELDS = {
    'name': ('Name', 'text'),
    'summary': ('Summary', 'text'),
    'first_release_date': ('First Launch Date', 'timestamp'),
    'genres': ('Genres', 'list'),
    'platforms': ('Platforms', 'list'),
    'game_modes': ('Game Modes', 'list'),
    'developers': ('Developers', 'company_list'),
    'publishers': ('Publishers', 'company_list'),
    'category': ('Category', 'text'),
}


def _extract_lookup_names_from_processed(
    processed_row: Mapping[str, Any], local_field: str
) -> list[str]:
    relation = LOOKUP_RELATIONS_BY_COLUMN.get(local_field)
    response_key = None
    if relation:
        response_key = relation['response_key']
    elif local_field in LOOKUP_RELATIONS_BY_KEY:
        response_key = local_field
    lookups_obj = processed_row.get('_lookup_entries') or processed_row.get('Lookups')
    names: list[str] = []
    if response_key and isinstance(lookups_obj, Mapping):
        data = lookups_obj.get(response_key)
        if isinstance(data, Mapping):
            raw_names = data.get('names')
            if isinstance(raw_names, (list, tuple)):
                for item in raw_names:
                    normalized = _normalize_lookup_name(item)
                    if normalized:
                        names.append(normalized)
            selected = data.get('selected')
            if not names and isinstance(selected, (list, tuple)):
                for entry in selected:
                    if isinstance(entry, Mapping):
                        normalized = _normalize_lookup_name(entry.get('name'))
                        if normalized:
                            names.append(normalized)
        elif isinstance(data, (list, tuple)):
            for entry in data:
                if isinstance(entry, Mapping):
                    normalized = _normalize_lookup_name(entry.get('name'))
                else:
                    normalized = _normalize_lookup_name(entry)
                if normalized:
                    names.append(normalized)
    if names:
        return names
    return _parse_iterable(processed_row.get(local_field))


def build_igdb_diff(
    processed_row: Mapping[str, Any], igdb_payload: Mapping[str, Any]
) -> dict[str, Any]:
    diff: dict[str, Any] = {}
    for igdb_field, (local_field, field_type) in IGDB_DIFF_FIELDS.items():
        remote_value = igdb_payload.get(igdb_field)
        local_value = processed_row.get(local_field)
        if field_type == 'list':
            remote_set = set(_parse_iterable(remote_value))
            local_set = set(
                _extract_lookup_names_from_processed(processed_row, local_field)
            )
            added = sorted(remote_set - local_set)
            removed = sorted(local_set - remote_set)
            if added or removed:
                diff[local_field] = {
                    'added': added,
                    'removed': removed,
                }
        elif field_type == 'company_list':
            remote_set = set(_parse_company_names(remote_value))
            local_set = set(
                _extract_lookup_names_from_processed(processed_row, local_field)
            )
            added = sorted(remote_set - local_set)
            removed = sorted(local_set - remote_set)
            if added or removed:
                diff[local_field] = {
                    'added': added,
                    'removed': removed,
                }
        else:
            if field_type == 'timestamp':
                remote_text = _normalize_timestamp(remote_value)
                local_text = _normalize_timestamp(local_value)
            else:
                remote_text = _normalize_text(remote_value)
                local_text = _normalize_text(local_value)
            if remote_text != local_text:
                entry: dict[str, Any] = {}
                if remote_text:
                    entry['added'] = remote_text
                if local_text:
                    entry['removed'] = local_text
                diff[local_field] = entry
    return diff


class GameNavigator:
    """Thread-safe helper to navigate game list and track progress."""

    def __init__(self, total_rows: int):
        self.lock = Lock()
        self.total = total_rows
        self.current_index = 0
        self.seq_index = 1
        self.skip_queue: list[dict[str, int]] = []
        self._load_initial()

    def _load_initial(self) -> None:
        with db_lock:
            conn = get_db()
            cur = conn.execute('SELECT current_index, seq_index, skip_queue FROM navigator_state WHERE id=1')
            state_row = cur.fetchone()
            cur = conn.execute('SELECT "Source Index", "ID" FROM processed_games')
            rows = cur.fetchall()
        processed = {int(r['Source Index']) for r in rows if str(r['Source Index']).isdigit()}
        max_seq = max((r['ID'] for r in rows), default=0)
        next_index = next((i for i in range(self.total) if i not in processed), self.total)
        expected_seq = max_seq + 1
        if state_row is not None:
            try:
                file_current = int(state_row['current_index'])
                file_seq = int(state_row['seq_index'])
                file_skip = json.loads(state_row['skip_queue'] or '[]')
                if file_current == next_index and file_seq == expected_seq:
                    self.current_index = file_current
                    self.seq_index = file_seq
                    self.skip_queue = file_skip
                    logger.debug(
                        "Loaded progress: current_index=%s seq_index=%s skip_queue=%s",
                        self.current_index,
                        self.seq_index,
                        self.skip_queue,
                    )
                    return
                logger.warning("Navigator state out of sync with database; rebuilding")
            except Exception as e:
                logger.warning("Failed to load navigator state: %s", e)
        self.current_index = next_index
        self.seq_index = expected_seq
        self.skip_queue = []
        logger.debug(
            "Loaded progress: current_index=%s seq_index=%s skip_queue=%s",
            self.current_index,
            self.seq_index,
            self.skip_queue,
        )
        self._save()

    def _load(self) -> None:
        with db_lock:
            conn = get_db()
            cur = conn.execute('SELECT current_index, seq_index, skip_queue FROM navigator_state WHERE id=1')
            state_row = cur.fetchone()
        if state_row is not None:
            try:
                self.current_index = int(state_row['current_index'])
                self.seq_index = int(state_row['seq_index'])
                self.skip_queue = json.loads(state_row['skip_queue'] or '[]')
                logger.debug(
                    "Loaded progress: current_index=%s seq_index=%s skip_queue=%s",
                    self.current_index,
                    self.seq_index,
                    self.skip_queue,
                )
                return
            except Exception as e:
                logger.warning("Failed to load navigator state: %s", e)
        # fallback: rebuild from processed_games
        self._load_initial()

    def _save(self) -> None:
        try:
            with db_lock:
                conn = get_db()
                conn.execute(
                    'REPLACE INTO navigator_state (id, current_index, seq_index, skip_queue) VALUES (1, ?, ?, ?)',
                    (
                        self.current_index,
                        self.seq_index,
                        json.dumps(self.skip_queue),
                    ),
                )
                conn.commit()
        except Exception as e:
            logger.warning("Failed to save navigator state: %s", e)

    def _process_skip_queue(self) -> None:
        logger.debug(
            "Processing skip queue: index=%s queue=%s",
            self.current_index,
            self.skip_queue,
        )
        for item in self.skip_queue:
            item['countdown'] -= 1
        for i, item in enumerate(self.skip_queue):
            if item['countdown'] <= 0:
                old_index = self.current_index
                self.current_index = item['index']
                del self.skip_queue[i]
                logger.debug(
                    "Skip queue hit: index from %s to %s",
                    old_index,
                    self.current_index,
                )
                break
        logger.debug(
            "After processing skip queue: index=%s queue=%s",
            self.current_index,
            self.skip_queue,
        )

    def current(self) -> int:
        with self.lock:
            self._load()
            self._process_skip_queue()
            self._save()
            return self.current_index

    def next(self) -> int:
        with self.lock:
            self._load()
            before = self.current_index
            logger.debug("next() before skip queue: index=%s", before)
            self._process_skip_queue()
            logger.debug("next() after skip queue: index=%s", self.current_index)
            if self.current_index < self.total:
                self.current_index += 1
            logger.debug("next() after increment: index=%s", self.current_index)
            self._save()
            return self.current_index

    def back(self) -> int:
        with self.lock:
            self._load()
            before = self.current_index
            logger.debug("back() before skip queue: index=%s", before)
            self._process_skip_queue()
            logger.debug("back() after skip queue: index=%s", self.current_index)
            if self.current_index > 0:
                self.current_index -= 1
            logger.debug("back() after decrement: index=%s", self.current_index)
            self._save()
            return self.current_index

    def skip(self, index: int) -> None:
        with self.lock:
            self._load()
            self.skip_queue = [s for s in self.skip_queue if s['index'] != index]
            self.skip_queue.append({'index': index, 'countdown': 30})
            if index == self.current_index:
                self.current_index += 1
            self._save()

    def set_index(self, index: int) -> None:
        with self.lock:
            self._load()
            if 0 <= index <= self.total:
                self.current_index = index
            self._save()


def extract_list(row: pd.Series, keys: list[str]) -> list[str]:
    """Return a list of comma-separated values from the first matching key."""
    for key in keys:
        if key in row.index:
            val = row.get(key, '')
            if pd.isna(val):
                return []
            return [g.strip() for g in str(val).split(',') if g.strip()]
    return []


def _normalize_column_name(name: str) -> str:
    return ''.join(ch.lower() for ch in str(name) if ch.isalnum())


def coerce_igdb_id(value: Any) -> str:
    if value is None:
        return ''
    try:
        if pd.isna(value):
            return ''
    except Exception:
        pass
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() == 'nan':
            return ''
        if text.endswith('.0') and text[:-2].isdigit():
            return text[:-2]
        return text
    if isinstance(value, numbers.Integral):
        return str(int(value))
    if isinstance(value, numbers.Real):
        if float(value).is_integer():
            return str(int(value))
        return str(value)
    text = str(value).strip()
    if not text or text.lower() == 'nan':
        return ''
    return text


def extract_igdb_id(row: pd.Series, allow_generic_id: bool = False) -> str:
    for key in row.index:
        normalized = _normalize_column_name(key)
        if 'igdb' in normalized and 'id' in normalized:
            value = coerce_igdb_id(row.get(key))
            if value:
                return value
    if allow_generic_id:
        for key in row.index:
            key_str = str(key)
            if key_str.lower() == 'id':
                value = coerce_igdb_id(row.get(key))
                if value:
                    return value
    return ''


def get_cell(row: pd.Series, key: str, missing: list[str]) -> str:
    """Return the cell value or an empty string if missing, tracking missing fields."""
    val = row.get(key, '')
    if pd.isna(val) or val is None:
        missing.append(key)
        return ''
    return val


def generate_pt_summary(game_name: str) -> str:
    """Generate a simple spoiler-free Portuguese summary for a game by name."""
    if not game_name:
        raise ValueError("game_name is required")
    if not os.environ.get('OPENAI_API_KEY'):
        raise RuntimeError("OPENAI_API_KEY not set")
    response = client.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=[
            {
                'role': 'system',
                'content': (
                    'Você é um assistente que cria sinopses curtas de jogos '
                    'em português do Brasil sem revelar spoilers.'
                ),
            },
            {
                'role': 'user',
                'content': (
                    f"Escreva uma sinopse um pouco mais longa (3 a 5 frases) para o jogo '{game_name}'."
                ),
            },
        ],
        temperature=0.7,
        max_tokens=200,
    )
    return response.choices[0].message.content.strip()


# initial load
ensure_dirs()
games_df = load_games()
categories_list = sorted({
    str(c).strip()
    for c in games_df.get('Category', pd.Series(dtype=str)).dropna()
    if str(c).strip()
})
platforms_list = sorted({
    p.strip()
    for ps in games_df.get('Platforms', pd.Series(dtype=str)).dropna()
    for p in str(ps).split(',')
    if p.strip()
})
total_games = len(games_df)
with app.app_context():
    backfill_igdb_ids()
    normalize_processed_games()
    navigator = GameNavigator(total_games)

@app.before_request
def require_login():
    if request.endpoint in ('login', 'static'):
        return
    if not session.get('authenticated'):
        return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        if request.form.get('password') == APP_PASSWORD:
            session['authenticated'] = True
            return redirect(url_for('index'))
        error = 'Invalid password'
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/')
def index():
    return render_template(
        'index.html',
        total=total_games,
        categories=categories_list,
        platforms=platforms_list,
    )


@app.route('/api/lookups/<lookup_type>')
def api_lookup_options(lookup_type: str):
    normalized = lookup_type.strip().lower().replace('-', '_').replace(' ', '_')
    table_name = LOOKUP_ENDPOINT_MAP.get(normalized)
    if not table_name or table_name not in LOOKUP_TABLES_BY_NAME:
        return jsonify({'error': 'unknown lookup type'}), 404
    with db_lock:
        conn = get_db()
        cur = conn.execute(
            f'SELECT id, name FROM {table_name} ORDER BY name COLLATE NOCASE'
        )
        rows = cur.fetchall()
    items: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    for row in rows:
        raw_id = _row_value(row, 'id', 0)
        try:
            coerced_id = int(raw_id)
        except (TypeError, ValueError):
            coerced_id = None
        if coerced_id is None or coerced_id in seen_ids:
            continue
        name = _normalize_lookup_name(_row_value(row, 'name', 1))
        if not name:
            continue
        seen_ids.add(coerced_id)
        items.append({'id': coerced_id, 'name': name})
    return jsonify({'items': items, 'type': table_name})


@app.route('/updates')
def updates_page():
    return render_template('updates.html')


def build_game_payload(index: int, seq: int) -> dict:
    try:
        row = games_df.iloc[index]
    except Exception:
        raise IndexError('invalid index')
    processed_row = None
    processed_lookup_entries: dict[str, list[dict[str, Any]]] = {}
    with db_lock:
        conn = get_db()
        cur = conn.execute('SELECT * FROM processed_games WHERE "Source Index"=?', (str(index),))
        processed_row = cur.fetchone()
        if processed_row is not None:
            processed_lookup_entries = _fetch_lookup_entries_for_game(
                conn, processed_row['ID']
            )

    if processed_row is not None and processed_row['Cover Path']:
        cover_path = processed_row['Cover Path']
        if os.path.exists(cover_path):
            try:
                img = open_image_auto_rotate(cover_path)
                buf = io.BytesIO()
                img.save(buf, format='JPEG')
                cover_data = 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()
            except Exception:
                app.logger.warning("Failed to open cover path %s", cover_path)
                cover_data = find_cover(row)
        else:
            app.logger.warning("Cover path %s missing", cover_path)
            cover_data = find_cover(row)
    else:
        cover_data = find_cover(row)

    processed_mapping: Mapping[str, Any] = (
        dict(processed_row) if processed_row is not None else {}
    )
    source_row = pd.Series(processed_mapping) if processed_mapping else row
    igdb_id = extract_igdb_id(source_row)
    if not igdb_id:
        igdb_id = extract_igdb_id(row, allow_generic_id=True)
    lookup_payloads: dict[str, dict[str, Any]] = {}
    for relation in LOOKUP_RELATIONS:
        response_key = relation['response_key']
        processed_column = relation['processed_column']
        entries = list(processed_lookup_entries.get(response_key, []))
        if not entries and processed_mapping:
            entries = _parse_lookup_entries_from_source(
                processed_mapping, processed_column
            )
        if not entries:
            entries = _parse_lookup_entries_from_source(row, processed_column)
        formatted = _format_lookup_response(entries)
        formatted['display'] = _lookup_display_text(formatted['names'])
        lookup_payloads[response_key] = formatted

    developers_display = lookup_payloads['Developers']['display']
    publishers_display = lookup_payloads['Publishers']['display']
    genres = lookup_payloads['Genres']['names']
    modes = lookup_payloads['GameModes']['names']
    platforms = lookup_payloads['Platforms']['names']
    missing: list[str] = []
    game_fields = {
        'Name': get_cell(source_row, 'Name', missing),
        'Summary': get_cell(source_row, 'Summary', missing),
        'FirstLaunchDate': get_cell(source_row, 'First Launch Date', missing),
        'Developers': developers_display,
        'Publishers': publishers_display,
        'Genres': genres,
        'GameModes': modes,
        'Category': get_cell(source_row, 'Category', missing),
        'Platforms': platforms,
        'IGDBID': igdb_id or None,
        'Lookups': lookup_payloads,
    }

    game_id = processed_row['ID'] if processed_row is not None else str(seq)

    return {
        'index': int(index),
        'total': total_games,
        'game': game_fields,
        'cover': cover_data,
        'seq': seq,
        'id': game_id,
        'missing': missing,
    }


@app.route('/api/game')
def api_game():
    try:
        index = navigator.current()
        if index >= total_games:
            return jsonify({'done': True, 'message': 'Todos os jogos foram processados.'})
        data = build_game_payload(index, navigator.seq_index)
        return jsonify(data)
    except Exception as e:
        app.logger.exception("api_game failed")
        return jsonify({'error': str(e)}), 500


@app.route('/api/game/<int:index>/raw')
def api_game_raw(index: int):
    if index < 0 or index >= total_games:
        return jsonify({'error': 'invalid index'}), 404
    try:
        row = games_df.iloc[index]
    except Exception:
        app.logger.exception("api_game_raw failed")
        return jsonify({'error': 'invalid index'}), 404
    cover_data = find_cover(row)
    genres = extract_list(row, ['Genres', 'Genre'])
    modes = extract_list(row, ['Game Modes', 'Mode'])
    platforms = extract_list(row, ['Platforms', 'Platform'])
    dummy: list[str] = []
    igdb_id = extract_igdb_id(row, allow_generic_id=True)
    game_fields = {
        'Name': get_cell(row, 'Name', dummy),
        'Summary': get_cell(row, 'Summary', dummy),
        'FirstLaunchDate': get_cell(row, 'First Launch Date', dummy),
        'Developers': get_cell(row, 'Developers', dummy),
        'Publishers': get_cell(row, 'Publishers', dummy),
        'Genres': genres,
        'GameModes': modes,
        'Category': get_cell(row, 'Category', dummy),
        'Platforms': platforms,
        'IGDBID': igdb_id or None,
    }
    return jsonify({
        'index': int(index),
        'total': total_games,
        'game': game_fields,
        'cover': cover_data,
        'seq': navigator.seq_index,
    })


@app.route('/api/summary', methods=['POST'])
def api_summary():
    data = request.get_json(force=True)
    game_name = data.get('game_name', '')
    try:
        summary_pt = generate_pt_summary(game_name)
        return jsonify({'summary': summary_pt})
    except Exception as e:
        app.logger.exception("Summary generation failed")
        return jsonify({'error': str(e)}), 500


@app.route('/api/upload', methods=['POST'])
def api_upload():
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'no file'}), 400
    img = open_image_auto_rotate(file.stream)
    filename = f"{uuid.uuid4().hex}.jpg"
    path = os.path.join(UPLOAD_DIR, filename)
    img.save(path, format='JPEG')
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    data = 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()
    return jsonify({'filename': filename, 'data': data})


@app.route('/api/save', methods=['POST'])
def api_save():
    data = request.get_json(force=True)
    index = int(data.get('index', 0))
    expected_id = data.get('id')
    fields = data.get('fields', {})
    image_b64 = data.get('image')
    upload_name = data.get('upload_name')
    if expected_id is None:
        return jsonify({'error': 'missing id'}), 400
    expected_id = int(expected_id)
    try:
        with navigator.lock:
            if index != navigator.current_index:
                expected_index = navigator.current_index
                expected_seq_id = navigator.seq_index
                with db_lock:
                    conn = get_db()
                    cur = conn.execute(
                        'SELECT "ID" FROM processed_games WHERE "Source Index"=?',
                        (str(expected_index),),
                    )
                    row = cur.fetchone()
                    if row is not None:
                        expected_seq_id = row['ID']
                return (
                    jsonify(
                        {
                            'error': 'index mismatch',
                            'expected': expected_index,
                            'actual': index,
                            'expected_id': expected_seq_id,
                        }
                    ),
                    409,
                )
            with db_lock:
                conn = get_db()
                cur = conn.execute(
                    'SELECT "ID", "igdb_id" FROM processed_games WHERE "Source Index"=?',
                    (str(index),),
                )
                existing = cur.fetchone()
                if existing:
                    existing_id = existing['ID']
                    if existing_id != expected_id:
                        return (
                            jsonify(
                                {
                                    'error': 'id mismatch',
                                    'expected': existing_id,
                                    'actual': expected_id,
                                }
                            ),
                            409,
                        )
                    seq_id = existing_id
                    new_record = False
                else:
                    seq_id = navigator.seq_index
                    if expected_id != seq_id:
                        return (
                            jsonify(
                                {
                                    'error': 'id mismatch',
                                    'expected': seq_id,
                                    'actual': expected_id,
                                }
                            ),
                            409,
                        )
                    new_record = True

            cover_path = ''
            width = height = 0
            if image_b64:
                header, b64data = image_b64.split(',', 1)
                img = Image.open(io.BytesIO(base64.b64decode(b64data)))
                img = img.convert('RGB')
                if min(img.size) < 1080:
                    img = img.resize((1080, 1080))
                else:
                    img = img.resize((1080, 1080))
                cover_path = os.path.join(PROCESSED_DIR, f"{seq_id}.jpg")
                img.save(cover_path, format='JPEG', quality=90)
                width, height = img.size

            igdb_id_value = None
            if existing is not None:
                try:
                    existing_raw = existing['igdb_id']
                except (KeyError, IndexError):
                    existing_raw = None
                existing_coerced = coerce_igdb_id(existing_raw)
                if existing_coerced:
                    igdb_id_value = existing_coerced
            if igdb_id_value is None and 0 <= index < len(games_df):
                igdb_candidate = extract_igdb_id(
                    games_df.iloc[index], allow_generic_id=True
                )
                if igdb_candidate:
                    igdb_id_value = igdb_candidate

            last_edit_ts = now_utc_iso()
            row = {
                "ID": seq_id,
                "Source Index": str(index),
                "Name": fields.get('Name', ''),
                "Summary": fields.get('Summary', ''),
                "First Launch Date": fields.get('FirstLaunchDate', ''),
                "Category": fields.get('Category', ''),
                "igdb_id": igdb_id_value,
                "Cover Path": cover_path,
                "Width": width,
                "Height": height,
                'last_edited_at': last_edit_ts,
            }

            lookups_input = fields.get('Lookups') if isinstance(fields, Mapping) else {}

            with db_lock:
                conn = get_db()
                try:
                    normalized_lookups: dict[str, dict[str, Any]] = {}
                    for relation in LOOKUP_RELATIONS:
                        response_key = relation['response_key']
                        processed_column = relation['processed_column']
                        raw_value: Any = None
                        if isinstance(lookups_input, Mapping):
                            raw_value = lookups_input.get(response_key)
                            if raw_value is None:
                                raw_value = lookups_input.get(processed_column)
                        if raw_value is None:
                            raw_value = fields.get(response_key)
                            if raw_value is None:
                                raw_value = fields.get(processed_column)
                        selection = _resolve_lookup_selection(conn, relation, raw_value)
                        normalized_lookups[response_key] = selection
                        row[processed_column] = _lookup_display_text(selection['names'])

                    if existing:
                        conn.execute(
                            '''UPDATE processed_games SET
                                "Name"=?, "Summary"=?, "First Launch Date"=?,
                                "Developers"=?, "Publishers"=?, "Genres"=?,
                                "Game Modes"=?, "Category"=?, "Platforms"=?,
                                "igdb_id"=?, "Cover Path"=?, "Width"=?, "Height"=?,
                                last_edited_at=?
                               WHERE "ID"=?''',
                            (
                                row['Name'], row['Summary'], row['First Launch Date'],
                                row.get('Developers', ''), row.get('Publishers', ''), row.get('Genres', ''),
                                row.get('Game Modes', ''), row['Category'], row.get('Platforms', ''),
                                row['igdb_id'], row['Cover Path'], row['Width'], row['Height'],
                                row['last_edited_at'],
                                seq_id,
                            ),
                        )
                    else:
                        conn.execute(
                            '''INSERT INTO processed_games (
                                "ID", "Source Index", "Name", "Summary",
                                "First Launch Date", "Developers", "Publishers",
                                "Genres", "Game Modes", "Category", "Platforms",
                                "igdb_id", "Cover Path", "Width", "Height", last_edited_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                            (
                                seq_id,
                                row['Source Index'], row['Name'], row['Summary'],
                                row['First Launch Date'], row.get('Developers', ''), row.get('Publishers', ''),
                                row.get('Genres', ''), row.get('Game Modes', ''), row['Category'],
                                row.get('Platforms', ''), row['igdb_id'], row['Cover Path'],
                                row['Width'], row['Height'], row['last_edited_at'],
                            ),
                        )

                    _persist_lookup_relations(conn, seq_id, normalized_lookups)
                    conn.commit()
                    if new_record:
                        navigator.seq_index += 1
                except sqlite3.IntegrityError:
                    conn.rollback()
                    return jsonify({'error': 'conflict'}), 409

            if upload_name:
                up_path = os.path.join(UPLOAD_DIR, upload_name)
                if os.path.exists(up_path):
                    os.remove(up_path)
            navigator.skip_queue = [s for s in navigator.skip_queue if s['index'] != index]
            navigator._save()
        return jsonify({'status': 'ok'})
    except Exception as e:
        app.logger.exception("api_save failed")
        return jsonify({'error': str(e)}), 500


@app.route('/api/skip', methods=['POST'])
def api_skip():
    data = request.get_json(force=True)
    index = int(data.get('index', 0))
    upload_name = data.get('upload_name')

    if upload_name:
        up_path = os.path.join(UPLOAD_DIR, upload_name)
        if os.path.exists(up_path):
            os.remove(up_path)
    try:
        navigator.skip(index)
        return jsonify({'status': 'ok'})
    except Exception as e:
        app.logger.exception("api_skip failed")
        return jsonify({'error': str(e)}), 500


def _collect_processed_games_with_igdb() -> list[dict[str, Any]]:
    with db_lock:
        conn = get_db()
        cur = conn.execute(
            "SELECT * FROM processed_games WHERE COALESCE(\"igdb_id\", '') != ''"
        )
        rows = cur.fetchall()
        results: list[dict[str, Any]] = []
        for row in rows:
            row_dict = dict(row)
            try:
                processed_id = int(row_dict['ID'])
            except (KeyError, TypeError, ValueError):
                processed_id = None
            if processed_id is not None:
                row_dict['_lookup_entries'] = _fetch_lookup_entries_for_game(
                    conn, processed_id
                )
            else:
                row_dict['_lookup_entries'] = {}
            results.append(row_dict)
    return results


def fetch_cached_updates() -> list[dict[str, Any]]:
    with db_lock:
        conn = get_db()
        cur = conn.execute(
            '''SELECT
                   u.processed_game_id,
                   u.igdb_id,
                   u.igdb_updated_at,
                   u.local_last_edited_at,
                   u.refreshed_at,
                   u.diff,
                   p."Name" AS game_name
               FROM igdb_updates u
               LEFT JOIN processed_games p ON p."ID" = u.processed_game_id
               ORDER BY p."Name" COLLATE NOCASE
            '''
        )
        rows = cur.fetchall()

    updates: list[dict[str, Any]] = []
    for row in rows:
        updates.append(
            {
                'processed_game_id': row['processed_game_id'],
                'igdb_id': row['igdb_id'],
                'igdb_updated_at': row['igdb_updated_at'],
                'local_last_edited_at': row['local_last_edited_at'],
                'refreshed_at': row['refreshed_at'],
                'name': row['game_name'],
                'has_diff': bool(row['diff']),
            }
        )
    return updates


@app.route('/api/updates/refresh', methods=['POST'])
def api_updates_refresh():
    processed_rows = _collect_processed_games_with_igdb()
    if not processed_rows:
        return jsonify({'status': 'ok', 'updated': 0, 'missing': []})

    try:
        access_token, client_id = exchange_twitch_credentials()
    except RuntimeError as exc:
        return jsonify({'error': str(exc)}), 400

    try:
        igdb_payloads = fetch_igdb_metadata(
            access_token,
            client_id,
            [row.get('igdb_id') for row in processed_rows],
        )
    except RuntimeError as exc:
        return jsonify({'error': str(exc)}), 502

    updated = 0
    missing: list[int] = []
    refreshed_at = now_utc_iso()

    with db_lock:
        conn = get_db()
        for row in processed_rows:
            igdb_id_value = row.get('igdb_id')
            if not igdb_id_value:
                continue
            payload = igdb_payloads.get(str(igdb_id_value))
            if not payload:
                try:
                    missing.append(int(row.get('ID', 0)))
                except (TypeError, ValueError):
                    pass
                continue
            igdb_updated_at = _normalize_timestamp(payload.get('updated_at'))
            diff = build_igdb_diff(row, payload)
            conn.execute(
                '''INSERT INTO igdb_updates (
                        processed_game_id, igdb_id, igdb_updated_at,
                        igdb_payload, diff, local_last_edited_at, refreshed_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(processed_game_id) DO UPDATE SET
                        igdb_id=excluded.igdb_id,
                        igdb_updated_at=excluded.igdb_updated_at,
                        igdb_payload=excluded.igdb_payload,
                        diff=excluded.diff,
                        local_last_edited_at=excluded.local_last_edited_at,
                        refreshed_at=excluded.refreshed_at''',
                (
                    int(row['ID']),
                    str(igdb_id_value),
                    igdb_updated_at,
                    json.dumps(payload),
                    json.dumps(diff),
                    row.get('last_edited_at') or refreshed_at,
                    refreshed_at,
                ),
            )
            updated += 1
        conn.commit()

    return jsonify({'status': 'ok', 'updated': updated, 'missing': missing})


@app.route('/api/updates', methods=['GET'])
def api_updates_list():
    return jsonify({'updates': fetch_cached_updates()})


@app.route('/api/updates/<int:processed_game_id>', methods=['GET'])
def api_updates_detail(processed_game_id: int):
    with db_lock:
        conn = get_db()
        cur = conn.execute(
            '''SELECT
                   u.processed_game_id,
                   u.igdb_id,
                   u.igdb_updated_at,
                   u.igdb_payload,
                   u.diff,
                   u.local_last_edited_at,
                   u.refreshed_at,
                   p."Name" AS game_name
               FROM igdb_updates u
               LEFT JOIN processed_games p ON p."ID" = u.processed_game_id
               WHERE u.processed_game_id=?''',
            (processed_game_id,),
        )
        row = cur.fetchone()

    if row is None:
        return jsonify({'error': 'not found'}), 404

    payload = json.loads(row['igdb_payload']) if row['igdb_payload'] else None
    diff = json.loads(row['diff']) if row['diff'] else {}

    return jsonify(
        {
            'processed_game_id': row['processed_game_id'],
            'igdb_id': row['igdb_id'],
            'igdb_updated_at': row['igdb_updated_at'],
            'igdb_payload': payload,
            'diff': diff,
            'local_last_edited_at': row['local_last_edited_at'],
            'refreshed_at': row['refreshed_at'],
            'name': row['game_name'],
        }
    )


@app.route('/api/set_index', methods=['POST'])
def api_set_index():
    data = request.get_json(silent=True) or {}
    index = int(data.get('index', 0))
    upload_name = data.get('upload_name')
    if upload_name:
        up_path = os.path.join(UPLOAD_DIR, upload_name)
        if os.path.exists(up_path):
            os.remove(up_path)
    try:
        navigator.set_index(index)
        return jsonify({'status': 'ok'})
    except Exception as e:
        app.logger.exception("api_set_index failed")
        return jsonify({'error': str(e)}), 500


@app.route('/api/game_by_id', methods=['POST'])
def api_game_by_id():
    data = request.get_json(silent=True) or {}
    raw_id = data.get('id')
    if raw_id is None:
        return jsonify({'error': 'missing id'}), 400
    try:
        game_id = int(str(raw_id).strip())
    except (TypeError, ValueError):
        return jsonify({'error': 'invalid id'}), 400

    upload_name = data.get('upload_name')
    if upload_name:
        up_path = os.path.join(UPLOAD_DIR, upload_name)
        if os.path.exists(up_path):
            os.remove(up_path)

    with db_lock:
        conn = get_db()
        cur = conn.execute(
            'SELECT "Source Index" FROM processed_games WHERE "ID"=?',
            (game_id,),
        )
        row = cur.fetchone()

    if row is None:
        return jsonify({'error': 'id not found'}), 404

    try:
        index = int(str(row['Source Index']))
    except (TypeError, ValueError):
        app.logger.error(
            "Invalid source index for ID %s: %s", game_id, row['Source Index']
        )
        return jsonify({'error': 'invalid source index'}), 500

    try:
        navigator.set_index(index)
        payload = build_game_payload(index, navigator.seq_index)
        return jsonify(payload)
    except IndexError:
        return jsonify({'error': 'invalid index'}), 404
    except Exception as e:
        app.logger.exception("api_game_by_id failed")
        return jsonify({'error': str(e)}), 500


@app.route('/api/next', methods=['POST'])
def api_next():
    data = request.get_json(silent=True) or {}
    upload_name = data.get('upload_name')
    if upload_name:
        up_path = os.path.join(UPLOAD_DIR, upload_name)
        if os.path.exists(up_path):
            os.remove(up_path)
    try:
        index = navigator.next()
        if index >= total_games:
            return jsonify({'done': True, 'message': 'Todos os jogos foram processados.'})
        payload = build_game_payload(index, navigator.seq_index)
        return jsonify(payload)
    except Exception as e:
        app.logger.exception("api_next failed")
        return jsonify({'error': str(e)}), 500


@app.route('/api/back', methods=['POST'])
def api_back():
    data = request.get_json(silent=True) or {}
    upload_name = data.get('upload_name')
    if upload_name:
        up_path = os.path.join(UPLOAD_DIR, upload_name)
        if os.path.exists(up_path):
            os.remove(up_path)
    try:
        index = navigator.back()
        payload = build_game_payload(index, navigator.seq_index)
        return jsonify(payload)
    except Exception as e:
        app.logger.exception("api_back failed")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
