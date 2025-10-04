import os
import json
import base64
import io
import sqlite3
import numbers
import re
import time
import math
from functools import partial
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Callable, Collection, Iterable, Mapping, MutableMapping, Optional
from threading import Lock
import logging
import logging.config

from flask import (
    Flask,
    request,
    jsonify,
    render_template,
    session,
    redirect,
    url_for,
    g,
)
from werkzeug.exceptions import RequestEntityTooLarge
from PIL import Image, ExifTags
import pandas as pd
from openai import OpenAI
from urllib.parse import urlparse, urlencode
from urllib.request import urlopen, Request
from urllib.error import HTTPError

# Placeholder imports to establish the upcoming modular structure.
import config as app_config
from config import (
    APP_PASSWORD,
    APP_SECRET_KEY,
    COVERS_DIR,
    IGDB_BATCH_SIZE,
    IGDB_USER_AGENT,
    INPUT_XLSX,
    OPENAI_API_KEY,
    OPENAI_SUMMARY_ENABLED,
    PROCESSED_DB,
    PROCESSED_DIR,
    RUN_DB_MIGRATIONS,
    SQLITE_TIMEOUT_SECONDS,
    UPLOAD_DIR,
    get_lookup_data_dir,
    LOG_FILE,
)
from helpers import (
    _collect_company_names,
    _format_first_release_date,
    _format_name_list,
    _normalize_lookup_name,
    _parse_company_names,
    _parse_iterable,
    has_cover_path_value,
    has_summary_text,
)
from db import utils as db_utils
from igdb import cache as igdb_cache
from igdb import diff as igdb_diff
from igdb.client import (
    IGDBClient,
    IGDB_CATEGORY_LABELS,
    coerce_igdb_id,
    cover_url_from_cover,
    extract_igdb_id,
    get_igdb_timeout_count,
    map_igdb_genres,
    map_igdb_modes,
    resolve_igdb_page_size,
)
from ingestion import data_loader as ingestion_data_loader
from init import initialize_app
from jobs import manager as jobs_manager
from lookups import config as lookups_config
from lookups import service as lookups_service
from media import covers as media_covers
from processed import duplicates as processed_duplicates
from processed import catalog as processed_catalog
from processed import navigator as processed_navigator
from processed import source_index_cache as processed_source_index_cache
from routes import games as routes_games
from routes import lookups as routes_lookups
from routes import updates as routes_updates
from routes import web as routes_web
from services import summaries as services_summaries
from updates import service as updates_service
from web import app_factory as web_app_factory

_PLACEHOLDER_IMPORTS = (
    app_config,
    db_utils,
    igdb_cache,
    igdb_diff,
    ingestion_data_loader,
    jobs_manager,
    lookups_config,
    lookups_service,
    media_covers,
    processed_duplicates,
    processed_navigator,
    processed_source_index_cache,
    routes_games,
    routes_lookups,
    routes_updates,
    routes_web,
    services_summaries,
    updates_service,
    web_app_factory,
)

logger = logging.getLogger(__name__)

igdb_api_client = IGDBClient()


def _determine_log_level(flask_app: Flask) -> int:
    if flask_app.debug:
        return logging.DEBUG
    env_value = str(flask_app.config.get('ENV', '')).lower()
    if env_value == 'development':
        return logging.DEBUG
    if os.environ.get('FLASK_DEBUG', '').lower() in {'1', 'true', 'yes', 'on'}:
        return logging.DEBUG
    return logging.INFO


def _configure_logging(flask_app: Flask) -> None:
    log_level = _determine_log_level(flask_app)
    log_path = Path(LOG_FILE)
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass

    for handler in list(flask_app.logger.handlers):
        flask_app.logger.removeHandler(handler)

    logging.config.dictConfig(
        {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'standard': {
                    'format': '%(asctime)s %(levelname)s [%(name)s] %(message)s',
                    'datefmt': '%Y-%m-%d %H:%M:%S',
                }
            },
            'handlers': {
                'console': {
                    'class': 'logging.StreamHandler',
                    'formatter': 'standard',
                    'level': log_level,
                    'stream': 'ext://sys.stdout',
                },
                'file': {
                    'class': 'logging.handlers.RotatingFileHandler',
                    'formatter': 'standard',
                    'level': logging.DEBUG,
                    'filename': os.fspath(log_path),
                    'maxBytes': 5 * 1024 * 1024,
                    'backupCount': 5,
                    'encoding': 'utf-8',
                },
            },
            'root': {
                'level': log_level,
                'handlers': ['console', 'file'],
            },
        }
    )

    flask_app.logger = logging.getLogger(flask_app.import_name)
    flask_app.logger.setLevel(log_level)
    logger.setLevel(log_level)

def _db_connection_factory() -> sqlite3.Connection:
    return db_utils._create_sqlite_connection(
        PROCESSED_DB, timeout=SQLITE_TIMEOUT_SECONDS
    )

LOOKUP_DATA_DIR = get_lookup_data_dir()

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


def _normalize_seed_value(value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, str):
        return value.strip()
    try:
        if pd.isna(value):
            return ''
    except Exception:
        pass
    text = str(value).strip()
    return '' if text.lower() == 'nan' else text

LOOKUP_RELATIONS_BY_COLUMN = {
    relation['processed_column']: relation for relation in LOOKUP_RELATIONS
}

LOOKUP_RELATIONS_BY_KEY = {
    relation['response_key']: relation for relation in LOOKUP_RELATIONS
}


LOOKUP_RELATIONS_BY_TABLE = {
    relation['lookup_table']: relation for relation in LOOKUP_RELATIONS
}


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


def _encode_lookup_id_list(values: Iterable[int]) -> str:
    normalized: list[int] = []
    seen: set[int] = set()
    for value in values:
        try:
            coerced = int(value)
        except (TypeError, ValueError):
            continue
        if coerced in seen:
            continue
        seen.add(coerced)
        normalized.append(coerced)
    if not normalized:
        return ''
    return json.dumps(normalized)


DuplicateGroupResolution = processed_duplicates.DuplicateGroupResolution
_coerce_int = processed_duplicates.coerce_int
_RELATION_COUNT_COLUMNS = tuple(f"{relation['join_table']}_count" for relation in LOOKUP_RELATIONS)


_fetch_lookup_entries_for_game = partial(
    lookups_service.fetch_lookup_entries_for_game,
    relations=LOOKUP_RELATIONS,
    normalize_lookup_name=_normalize_lookup_name,
    decode_lookup_id_list=_decode_lookup_id_list,
    parse_iterable=_parse_iterable,
    row_value=_row_value,
)

_iter_lookup_payload = partial(
    lookups_service.iter_lookup_payload,
    normalize_lookup_name=_normalize_lookup_name,
)

_format_lookup_response = partial(
    lookups_service.format_lookup_response,
    normalize_lookup_name=_normalize_lookup_name,
)

_resolve_lookup_selection = partial(
    lookups_service.resolve_lookup_selection,
    normalize_lookup_name=_normalize_lookup_name,
)

_load_lookup_tables = partial(
    lookups_service.load_lookup_tables,
    tables=LOOKUP_TABLES,
    data_dir=LOOKUP_DATA_DIR,
    normalize_lookup_name=_normalize_lookup_name,
    log=logger,
)

_ensure_lookup_id_columns = partial(
    lookups_service.ensure_lookup_id_columns,
    relations=LOOKUP_RELATIONS,
    encode_lookup_id_list=_encode_lookup_id_list,
    decode_lookup_id_list=_decode_lookup_id_list,
    row_value=_row_value,
)

_get_or_create_lookup_id = partial(
    lookups_service.get_or_create_lookup_id,
    normalize_lookup_name=_normalize_lookup_name,
)

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


def _format_lookup_label(value: str) -> str:
    text = str(value or '').replace('_', ' ').strip()
    if not text:
        return ''
    spaced = re.sub(r'(?<!^)(?=[A-Z])', ' ', text)
    return spaced.strip().title()


def is_processed_game_done(summary_value: Any, cover_path_value: Any) -> bool:
    """Return ``True`` when a processed row has the required summary and cover."""

    return has_summary_text(summary_value) and has_cover_path_value(cover_path_value)

app = Flask(__name__)
app.secret_key = APP_SECRET_KEY
app.config.setdefault('OPENAI_SUMMARY_ENABLED', OPENAI_SUMMARY_ENABLED)

_configure_logging(app)

IGDB_CACHE_TABLE = 'igdb_games'
IGDB_CACHE_STATE_TABLE = 'igdb_cache_state'
UPDATES_LIST_TABLE = 'updates_list'

def _igdb_category_display(value: Any) -> str:
    return IGDBClient.translate_category(value)


def _coerce_rating_count(primary: Any, secondary: Any) -> int | None:
    for candidate in (primary, secondary):
        if candidate in (None, ''):
            continue
        if isinstance(candidate, bool):
            continue
        if isinstance(candidate, numbers.Integral):
            return int(candidate)
        if isinstance(candidate, numbers.Real):
            return int(float(candidate))
        try:
            text = str(candidate).strip()
            if not text:
                continue
            return int(float(text))
        except (TypeError, ValueError):
            continue
    return None


def _read_http_error(exc: HTTPError) -> str:
    message = ''
    try:
        data = exc.read()
    except Exception:  # pragma: no cover - best effort logging
        data = b''
    if data:
        try:
            message = data.decode('utf-8', errors='replace').strip()
        except Exception:  # pragma: no cover - unexpected decoding failure
            message = ''
    if not message and getattr(exc, 'reason', None):
        message = str(exc.reason)
    if message:
        return f"{exc.code} {message}".strip()
    return str(exc)
# Configure OpenAI using API key from environment
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# SQLite setup for processed games
db_lock = db_utils.db_lock

job_manager = jobs_manager.get_job_manager()


def get_db() -> sqlite3.Connection:
    return db_utils.get_db(_db_connection_factory)


def get_processed_games_columns(conn: sqlite3.Connection | None = None) -> set[str]:
    return db_utils.get_processed_games_columns(
        conn, connection_factory=_db_connection_factory
    )


def _navigator_factory() -> processed_navigator.GameNavigator:
    return processed_navigator.GameNavigator(
        db_lock=db_lock,
        get_db=get_db,
        is_processed_game_done=is_processed_game_done,
        logger=logger,
    )


catalog_state = processed_catalog.CatalogState(
    navigator_factory=_navigator_factory,
    category_labels=IGDB_CATEGORY_LABELS,
    logger=logger,
)


def _ensure_navigator_dataframe(rebuild_state: bool = False) -> processed_navigator.GameNavigator:
    """Sync navigator state with the current in-memory games DataFrame."""

    return catalog_state.ensure_navigator_dataframe(rebuild_state=rebuild_state)


def reset_source_index_cache() -> None:
    """Expose navigator cache reset for test helpers and background jobs."""

    catalog_state.reset_source_index_cache()


def get_source_index_for_position(position: int) -> str:
    """Delegate to the navigator instance for ``Source Index`` lookups."""

    navigator = _ensure_navigator_dataframe(rebuild_state=False)
    return navigator.get_source_index_for_position(position)


def get_position_for_source_index(value: Any) -> int | None:
    """Delegate reverse lookup of ``Source Index`` values to the navigator."""

    navigator = _ensure_navigator_dataframe(rebuild_state=False)
    return navigator.get_position_for_source_index(value)


class GameNavigator(processed_navigator.GameNavigator):
    """Compatibility wrapper exposing the legacy navigator constructor signature."""

    def __init__(self, total_rows: int):
        super().__init__(
            db_lock=db_lock,
            get_db=get_db,
            is_processed_game_done=is_processed_game_done,
            logger=logger,
        )
        df = catalog_state.games_df if catalog_state.games_df is not None else pd.DataFrame()
        self.set_games_df(df, rebuild_state=False)
        self.total = total_rows
        self._load_initial()


def _quote_identifier(identifier: str) -> str:
    return db_utils._quote_identifier(identifier)





LOOKUP_RELATIONS_BY_TABLE = {
    relation['lookup_table']: relation for relation in LOOKUP_RELATIONS
}


DuplicateGroupResolution = processed_duplicates.DuplicateGroupResolution
_coerce_int = processed_duplicates.coerce_int
_RELATION_COUNT_COLUMNS = tuple(f"{relation['join_table']}_count" for relation in LOOKUP_RELATIONS)


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


def _lookup_display_text(names: list[str]) -> str:
    return ', '.join(name for name in names if name)


def _lookup_name_for_id(
    conn: sqlite3.Connection, table_name: str, lookup_id: int
) -> str:
    name = lookups_service.lookup_name_for_id(
        conn,
        table_name,
        lookup_id,
        normalize_lookup_name=_normalize_lookup_name,
    )
    return name or ''


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


def _ensure_lookup_join_tables(conn: sqlite3.Connection) -> None:
    for relation in LOOKUP_RELATIONS:
        join_table = relation['join_table']
        join_column = relation['join_column']
        lookup_table = relation['lookup_table']
        try:
            conn.execute(
                f'''
                    CREATE TABLE IF NOT EXISTS {join_table} (
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
            logger.exception('Failed to ensure join table %s exists', join_table)


def _persist_lookup_relations(
    conn: sqlite3.Connection,
    processed_game_id: int,
    selections: Mapping[str, Mapping[str, Any]] | Mapping[str, Any],
) -> None:
    lookups_service.persist_relations(
        conn,
        processed_game_id,
        selections,
        LOOKUP_RELATIONS,
    )


def _lookup_entries_to_selection(
    entries: Mapping[str, list[dict[str, Any]]]
) -> dict[str, dict[str, list[int]]]:
    return lookups_service.lookup_entries_to_selection(entries, LOOKUP_RELATIONS)


def _apply_lookup_entries_to_processed_game(
    conn: sqlite3.Connection,
    processed_game_id: int,
    entries: Mapping[str, list[dict[str, Any]]],
) -> None:
    lookups_service.apply_relations_to_game(
        conn,
        processed_game_id,
        entries,
        LOOKUP_RELATIONS,
        normalize_lookup_name=_normalize_lookup_name,
        encode_lookup_id_list=_encode_lookup_id_list,
        lookup_display_text=_lookup_display_text,
        columns=get_processed_games_columns(conn),
    )


def _remove_lookup_id_from_entries(
    entries: MutableMapping[str, list[dict[str, Any]]],
    relation: Mapping[str, Any],
    lookup_id: int,
) -> None:
    lookups_service.remove_lookup_id_from_entries(entries, relation, lookup_id)


def _list_lookup_entries(
    conn: sqlite3.Connection,
    table_name: str,
    *,
    limit: int,
    offset: int,
) -> tuple[list[dict[str, Any]], int]:
    return lookups_service.list_lookup_entries(
        conn,
        table_name,
        normalize_lookup_name=_normalize_lookup_name,
        limit=limit,
        offset=offset,
    )


def _create_lookup_entry(
    conn: sqlite3.Connection, table_name: str, name: str
) -> tuple[str, dict[str, Any] | None]:
    return lookups_service.create_lookup_entry(
        conn, table_name, name, normalize_lookup_name=_normalize_lookup_name
    )


def _update_lookup_entry(
    conn: sqlite3.Connection, table_name: str, lookup_id: int, name: str
) -> tuple[str, dict[str, Any] | None]:
    return lookups_service.update_lookup_entry(
        conn,
        table_name,
        lookup_id,
        name,
        normalize_lookup_name=_normalize_lookup_name,
    )


def _delete_lookup_entry(
    conn: sqlite3.Connection, table_name: str, lookup_id: int
) -> bool:
    return lookups_service.delete_lookup_entry(conn, table_name, lookup_id)


def _related_processed_game_ids(
    conn: sqlite3.Connection, relation: Mapping[str, Any], lookup_id: int
) -> list[int]:
    return lookups_service.get_related_processed_game_ids(conn, relation, lookup_id)


def _backfill_lookup_relations(conn: sqlite3.Connection) -> None:
    lookups_service.backfill_relations(
        conn,
        LOOKUP_RELATIONS,
        normalize_lookup_name=_normalize_lookup_name,
        parse_iterable=_parse_iterable,
        get_or_create_lookup_id=_get_or_create_lookup_id,
        decode_lookup_id_list=_decode_lookup_id_list,
        row_value=_row_value,
    )


def _migrate_id_column(conn: sqlite3.Connection) -> None:
    try:
        cur = conn.execute('PRAGMA table_info(processed_games)')
    except sqlite3.OperationalError:
        return

    cols = cur.fetchall()
    if not cols:
        return

    id_col = next((c for c in cols if c[1] == 'ID'), None)
    has_source_index = any(c[1] == 'Source Index' for c in cols)
    id_type = str(id_col[2]).upper() if id_col and id_col[2] is not None else ''
    id_pk = id_col[5] if id_col else 0
    if id_col and id_type == 'INTEGER' and id_pk == 1:
        return

    try:
        cur = conn.execute('SELECT * FROM processed_games ORDER BY rowid')
    except sqlite3.OperationalError:
        rows: list[sqlite3.Row] = []
        column_names: list[str] = []
    else:
        rows = cur.fetchall()
        column_names = [desc[0] for desc in cur.description] if cur.description else []

    records: list[dict[str, Any]] = []
    for row in rows:
        record: dict[str, Any] = {}
        for idx, column in enumerate(column_names):
            if idx < len(row):
                record[column] = row[idx]
        records.append(record)

    desired_columns = [
        'ID',
        'Source Index',
        'Name',
        'Summary',
        'First Launch Date',
        'Developers',
        'developers_ids',
        'Publishers',
        'publishers_ids',
        'Genres',
        'genres_ids',
        'Game Modes',
        'game_modes_ids',
        'Category',
        'Platforms',
        'platforms_ids',
        'igdb_id',
        'Cover Path',
        'Width',
        'Height',
        'last_edited_at',
    ]

    def _coerce_old_id(value: Any) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool):
            candidate = int(value)
        elif isinstance(value, numbers.Integral):
            candidate = int(value)
        elif isinstance(value, numbers.Real):
            float_value = float(value)
            if not float_value.is_integer():
                return None
            candidate = int(float_value)
        else:
            text = str(value).strip()
            if not text:
                return None
            try:
                candidate = int(text)
            except ValueError:
                return None
        return candidate

    used_ids: set[int] = set()
    new_rows: list[tuple[Any, ...]] = []
    next_id = 1
    for record in records:
        candidate = _coerce_old_id(record.get('ID'))
        if candidate is not None and candidate > 0 and candidate not in used_ids:
            new_id = candidate
        else:
            while next_id in used_ids:
                next_id += 1
            new_id = next_id
            next_id += 1
        used_ids.add(new_id)

        row_values: list[Any] = []
        for column in desired_columns:
            if column == 'ID':
                row_values.append(new_id)
            else:
                row_values.append(record.get(column))
        new_rows.append(tuple(row_values))

    columns_sql = ', '.join(_quote_identifier(column) for column in desired_columns)
    placeholders = ', '.join('?' for _ in desired_columns)

    conn.execute('PRAGMA foreign_keys = OFF')
    try:
        conn.execute('ALTER TABLE processed_games RENAME TO processed_games_old')
        conn.execute(
            '''
            CREATE TABLE processed_games (
                "ID" INTEGER PRIMARY KEY,
                "Source Index" TEXT UNIQUE,
                "Name" TEXT,
                "Summary" TEXT,
                "First Launch Date" TEXT,
                "Developers" TEXT,
                "developers_ids" TEXT,
                "Publishers" TEXT,
                "publishers_ids" TEXT,
                "Genres" TEXT,
                "genres_ids" TEXT,
                "Game Modes" TEXT,
                "game_modes_ids" TEXT,
                "Category" TEXT,
                "Platforms" TEXT,
                "platforms_ids" TEXT,
                "igdb_id" TEXT,
                "Cover Path" TEXT,
                "Width" INTEGER,
                "Height" INTEGER,
                cache_rank INTEGER,
                last_edited_at TEXT
            )
            '''
        )
        if new_rows:
            conn.executemany(
                f'INSERT INTO processed_games ({columns_sql}) VALUES ({placeholders})',
                new_rows,
            )
        if has_source_index:
            try:
                conn.execute(
                    '''
                    UPDATE igdb_updates
                    SET processed_game_id = (
                        SELECT pg."ID"
                        FROM processed_games AS pg
                        JOIN processed_games_old AS old
                            ON old."Source Index" = pg."Source Index"
                        WHERE old."ID" = igdb_updates.processed_game_id
                        LIMIT 1
                    )
                    WHERE EXISTS (
                        SELECT 1
                        FROM processed_games_old AS old
                        WHERE old."ID" = igdb_updates.processed_game_id
                    )
                    '''
                )
            except sqlite3.OperationalError:
                pass
        conn.execute('DROP TABLE processed_games_old')
    finally:
        conn.execute('PRAGMA foreign_keys = ON')


def _init_db(*, run_migrations: bool = RUN_DB_MIGRATIONS) -> None:
    conn = _db_connection_factory()
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
                    "developers_ids" TEXT,
                    "Publishers" TEXT,
                    "publishers_ids" TEXT,
                    "Genres" TEXT,
                    "genres_ids" TEXT,
                    "Game Modes" TEXT,
                    "game_modes_ids" TEXT,
                    "Category" TEXT,
                    "Platforms" TEXT,
                    "platforms_ids" TEXT,
                    "igdb_id" TEXT,
                    "Cover Path" TEXT,
                    "Width" INTEGER,
                    "Height" INTEGER,
                    cache_rank INTEGER,
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
            if run_migrations:
                _migrate_id_column(conn)
            if run_migrations:
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
                    conn.execute('ALTER TABLE processed_games ADD COLUMN cache_rank INTEGER')
                except sqlite3.OperationalError:
                    pass
                db_utils.clear_processed_games_columns_cache()
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
                        has_diff INTEGER NOT NULL DEFAULT 0,
                        FOREIGN KEY(processed_game_id) REFERENCES processed_games("ID")
                    )'''
                )
            except sqlite3.OperationalError:
                pass

            try:
                conn.execute(
                    'ALTER TABLE igdb_updates ADD COLUMN has_diff INTEGER NOT NULL DEFAULT 0'
                )
            except sqlite3.OperationalError:
                pass

            try:
                conn.execute(
                    'CREATE INDEX IF NOT EXISTS igdb_updates_refreshed_at_idx '
                    'ON igdb_updates(refreshed_at)'
                )
            except sqlite3.OperationalError:
                pass

            try:
                conn.execute(
                    'CREATE INDEX IF NOT EXISTS igdb_updates_has_diff_updated_idx '
                    'ON igdb_updates(has_diff, igdb_updated_at)'
                )
            except sqlite3.OperationalError:
                pass

            try:
                conn.execute(
                    'CREATE INDEX IF NOT EXISTS igdb_updates_order_idx '
                    'ON igdb_updates(refreshed_at, local_last_edited_at, igdb_updated_at, processed_game_id)'
                )
            except sqlite3.OperationalError:
                pass

            try:
                conn.execute(
                    '''UPDATE igdb_updates
                       SET has_diff = CASE
                           WHEN diff IS NULL OR diff = '' OR diff = '{}' THEN 0
                           ELSE 1
                       END
                       WHERE has_diff NOT IN (0, 1) OR has_diff IS NULL'''
                )
            except sqlite3.OperationalError:
                pass

            try:
                conn.execute(
                    f'''CREATE TABLE IF NOT EXISTS {UPDATES_LIST_TABLE} (
                            processed_game_id INTEGER PRIMARY KEY,
                            igdb_id TEXT,
                            igdb_updated_at TEXT,
                            local_last_edited_at TEXT,
                            refreshed_at TEXT,
                            name TEXT,
                            has_diff INTEGER NOT NULL DEFAULT 0,
                            cover TEXT,
                            cover_available INTEGER NOT NULL DEFAULT 0,
                            update_type TEXT NOT NULL,
                            detail_available INTEGER NOT NULL DEFAULT 0,
                            cursor_value TEXT,
                            sort_numeric REAL,
                            entry_type TEXT NOT NULL DEFAULT 'm',
                            entry_rank INTEGER NOT NULL DEFAULT 0
                        )'''
                )
            except sqlite3.OperationalError:
                pass

            try:
                conn.execute(
                    f'''CREATE INDEX IF NOT EXISTS {UPDATES_LIST_TABLE}_sort_idx '''
                    f'''ON {UPDATES_LIST_TABLE}(sort_numeric DESC, processed_game_id DESC, entry_rank ASC)'''
                )
            except sqlite3.OperationalError:
                pass

            try:
                conn.execute(
                    f'''
                    CREATE TABLE IF NOT EXISTS {IGDB_CACHE_TABLE} (
                        igdb_id INTEGER PRIMARY KEY,
                        name TEXT,
                        summary TEXT,
                        updated_at INTEGER,
                        first_release_date INTEGER,
                        category INTEGER,
                        cover_image_id TEXT,
                        rating_count INTEGER,
                        developers TEXT,
                        publishers TEXT,
                        genres TEXT,
                        platforms TEXT,
                        game_modes TEXT,
                        cached_at TEXT
                    )
                    '''
                )
            except sqlite3.OperationalError:
                pass

            try:
                conn.execute(
                    f'''CREATE INDEX IF NOT EXISTS {IGDB_CACHE_TABLE}_updated_at_id_idx '''
                    f'''ON {IGDB_CACHE_TABLE}(updated_at, igdb_id)'''
                )
            except sqlite3.OperationalError:
                pass

            try:
                conn.execute(
                    'CREATE INDEX IF NOT EXISTS processed_games_igdb_id_idx '
                    'ON processed_games("igdb_id")'
                )
            except sqlite3.OperationalError:
                pass

            try:
                conn.execute(
                    '''
                    CREATE UNIQUE INDEX IF NOT EXISTS processed_games_igdb_cache_idx
                    ON processed_games("igdb_id", cache_rank)
                    WHERE "igdb_id" IS NOT NULL AND "igdb_id" != ''
                    '''
                )
            except sqlite3.OperationalError:
                pass

            try:
                conn.execute(
                    'CREATE INDEX IF NOT EXISTS processed_games_last_edited_idx '
                    'ON processed_games(last_edited_at, "ID")'
                )
            except sqlite3.OperationalError:
                pass

            try:
                conn.execute(
                    f'''
                    CREATE TABLE IF NOT EXISTS {IGDB_CACHE_STATE_TABLE} (
                        id INTEGER PRIMARY KEY CHECK (id = 1),
                        total_count INTEGER,
                        last_synced_at TEXT
                    )
                    '''
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

            _ensure_lookup_join_tables(conn)

            if run_migrations:
                _load_lookup_tables(conn)
                _recreate_lookup_join_tables(conn)
                _backfill_lookup_relations(conn)
                _ensure_lookup_id_columns(conn)

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

                cleanup_cursor = conn.execute(
                    '''
                    DELETE FROM igdb_updates
                    WHERE processed_game_id IS NULL
                       OR NOT EXISTS (
                            SELECT 1
                            FROM processed_games
                            WHERE processed_games."ID" = igdb_updates.processed_game_id
                        )
                    '''
                )
                logger.info(
                    "Startup cleanup removed %d orphan igdb_update rows; database may still"
                    " require manual attention.",
                    cleanup_cursor.rowcount,
                )

                cur = conn.execute(
                    'SELECT "ID", "igdb_id", last_edited_at FROM processed_games'
                )
                for raw_game_id, igdb_id_value, last_edit in cur.fetchall():
                    if not igdb_id_value:
                        continue
                    try:
                        if isinstance(raw_game_id, numbers.Integral):
                            game_id = int(raw_game_id)
                        elif isinstance(raw_game_id, numbers.Real):
                            float_value = float(raw_game_id)
                            if not float_value.is_integer():
                                continue
                            game_id = int(float_value)
                        else:
                            text = str(raw_game_id).strip()
                            if not text:
                                continue
                            game_id = int(text)
                    except (TypeError, ValueError):
                        continue
                    guard_cursor = conn.execute(
                        'SELECT 1 FROM processed_games WHERE "ID"=?', (game_id,)
                    )
                    if guard_cursor.fetchone() is None:
                        continue

                    try:
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
                    except sqlite3.IntegrityError:
                        logger.warning(
                            "Failed to reseed igdb_updates for processed_game_id=%s due to"
                            " integrity error.",
                            game_id,
                        )
                        continue
            if run_migrations:
                try:
                    conn.execute('ANALYZE')
                except sqlite3.OperationalError:
                    pass
    finally:
        conn.close()



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


try:  # Pillow >= 9.1
    _RESAMPLE_LANCZOS = Image.Resampling.LANCZOS  # type: ignore[attr-defined]
except AttributeError:  # pragma: no cover - fallback for older Pillow
    _RESAMPLE_LANCZOS = Image.LANCZOS


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


def _prepare_cover_image(img: Image.Image, *, min_size: int = 1080) -> Image.Image:
    """Ensure ``img`` is RGB and meets the minimum cover size."""

    prepared = img.convert('RGB')
    width, height = prepared.size
    shortest = min(width, height)
    if shortest <= 0:
        return prepared
    if shortest >= min_size:
        return prepared

    scale = min_size / shortest
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    if (new_width, new_height) == prepared.size:
        return prepared
    return prepared.resize((new_width, new_height), _RESAMPLE_LANCZOS)


def save_cover_image(
    img: Image.Image,
    dest_path: str,
    *,
    min_size: int = 1080,
    quality: int = 90,
) -> tuple[Image.Image, int, int]:
    """Persist ``img`` to ``dest_path`` ensuring minimum dimensions."""

    prepared = _prepare_cover_image(img, min_size=min_size)
    directory = os.path.dirname(dest_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    prepared.save(dest_path, format='JPEG', quality=quality)
    width, height = prepared.size
    return prepared, width, height


def load_games(*, prefer_cache: bool = False) -> pd.DataFrame:
    columns = [
        'Source Index',
        'Name',
        'Summary',
        'First Launch Date',
        'Category',
        'Developers',
        'Publishers',
        'Genres',
        'Game Modes',
        'Platforms',
        'Large Cover Image (URL)',
        'Rating Count',
        'IGDB ID',
        'igdb_id',
    ]

    def _finalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=columns)

        df = df.copy()

        if 'Source Index' in df.columns:
            def _coerce_source(value: Any) -> str:
                try:
                    return str(int(str(value).strip()))
                except (TypeError, ValueError):
                    text = str(value).strip() if value is not None else ''
                    return text

            df['Source Index'] = df['Source Index'].apply(_coerce_source)
        else:
            df.insert(0, 'Source Index', [str(i) for i in range(len(df))])

        df = df.reset_index(drop=True)

        if 'igdb_id' in df.columns:
            df['igdb_id'] = df['igdb_id'].fillna('').astype(str)
        else:
            df['igdb_id'] = ''

        df['IGDB ID'] = df['igdb_id']

        if 'Rating Count' in df.columns:
            df['Rating Count'] = (
                pd.to_numeric(df['Rating Count'], errors='coerce').fillna(0)
            )
        else:
            df['Rating Count'] = 0

        for column in columns:
            if column not in df.columns:
                df[column] = '' if column != 'Rating Count' else 0

        ordered_columns = [col for col in columns if col in df.columns]
        df = df[ordered_columns]
        return df

    def _load_from_db() -> tuple[pd.DataFrame, bool]:
        with db_lock:
            conn = get_db()
            try:
                cur = conn.execute('SELECT COUNT(*) FROM processed_games')
            except sqlite3.OperationalError:
                return pd.DataFrame(columns=columns), False
            row = cur.fetchone()
            total = row[0] if row else 0
            if not total:
                return pd.DataFrame(columns=columns), False

            processed_columns = get_processed_games_columns(conn)
            select_parts: list[str] = []
            for column in columns:
                if column == 'IGDB ID':
                    continue
                identifier = _quote_identifier(column)
                if column in processed_columns:
                    select_parts.append(identifier)
                else:
                    select_parts.append(f'NULL AS {identifier}')

            query = (
                'SELECT '
                + ', '.join(select_parts)
                + ' FROM processed_games ORDER BY CAST("Source Index" AS INTEGER)'
            )
            df = pd.read_sql_query(query, conn)

        return _finalize_dataframe(df), True

    def _load_from_cache() -> tuple[pd.DataFrame, bool]:
        with db_lock:
            conn = get_db()
            try:
                cur = conn.execute(
                    f'''SELECT igdb_id, name, summary, first_release_date,
                               category, developers, publishers, genres,
                               platforms, game_modes, cover_image_id, rating_count
                        FROM {IGDB_CACHE_TABLE}
                        ORDER BY igdb_id'''
                )
                rows = cur.fetchall()
            except sqlite3.OperationalError:
                return pd.DataFrame(columns=columns), False

        if not rows:
            return pd.DataFrame(columns=columns), False

        records: list[dict[str, Any]] = []
        for row in rows:
            igdb_id_value = row['igdb_id']
            normalized_id = coerce_igdb_id(igdb_id_value)
            cover_value = (
                {'image_id': row['cover_image_id']} if row['cover_image_id'] else None
            )
            try:
                rating_value = int(row['rating_count']) if row['rating_count'] is not None else 0
            except (TypeError, ValueError):
                rating_value = 0
            developers = _deserialize_cache_list(row['developers'])
            publishers = _deserialize_cache_list(row['publishers'])
            genres = _deserialize_cache_list(row['genres'])
            game_modes = _deserialize_cache_list(row['game_modes'])
            platforms = _deserialize_cache_list(row['platforms'])
            records.append(
                {
                    'Name': row['name'] or '',
                    'Summary': row['summary'] or '',
                    'First Launch Date': _format_first_release_date(
                        row['first_release_date']
                    ),
                    'Category': _igdb_category_display(row['category']),
                    'Developers': ', '.join(developers),
                    'Publishers': ', '.join(publishers),
                    'Genres': ', '.join(genres),
                    'Game Modes': ', '.join(game_modes),
                    'Platforms': ', '.join(platforms),
                    'Large Cover Image (URL)': cover_url_from_cover(
                        cover_value, size='t_original'
                    ),
                    'Rating Count': rating_value,
                    'IGDB ID': normalized_id or '',
                    'igdb_id': normalized_id or '',
                }
            )

        df = pd.DataFrame(records)
        if df.empty:
            return pd.DataFrame(columns=columns), False

        return _finalize_dataframe(df), True

    def _load_from_igdb() -> pd.DataFrame:
        try:
            access_token, client_id = exchange_twitch_credentials()
        except Exception as exc:
            logger.warning(
                "Unable to obtain Twitch credentials for IGDB fetch: %s", exc
            )
            return pd.DataFrame(columns=columns)

        page_size = resolve_igdb_page_size(IGDB_BATCH_SIZE)
        offset = 0
        raw_items: list[Mapping[str, Any]] = []

        while True:
            query = (
                'fields '
                'id,name,summary,first_release_date,total_rating_count,rating_count,'
                'genres.name,platforms.name,game_modes.name,category,'
                'involved_companies.company.name,involved_companies.developer,'
                'involved_companies.publisher,cover.image_id; '
                f'limit {page_size}; '
                f'offset {offset}; '
                'sort total_rating_count desc;'
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

            response_bytes = b''
            try:
                with urlopen(request) as response:
                    response_bytes = response.read()
            except HTTPError as exc:
                logger.warning(
                    "IGDB request failed while loading games: %s",
                    _read_http_error(exc),
                )
                return pd.DataFrame(columns=columns)
            except Exception as exc:
                logger.warning("Failed to query IGDB while loading games: %s", exc)
                return pd.DataFrame(columns=columns)

            try:
                payload = json.loads(response_bytes.decode('utf-8'))
            except Exception as exc:
                logger.warning("Failed to decode IGDB payload: %s", exc)
                return pd.DataFrame(columns=columns)

            if not payload:
                break
            if not isinstance(payload, list):
                logger.warning(
                    "Unexpected IGDB payload type: %s", type(payload).__name__
                )
                break
            for item in payload:
                if isinstance(item, Mapping):
                    raw_items.append(item)
            offset += len(payload)

        records: list[dict[str, Any]] = []
        for item in raw_items:
            if not isinstance(item, Mapping):
                continue
            name_value = item.get('name')
            name = str(name_value).strip() if name_value is not None else ''
            if not name:
                continue
            summary_value = item.get('summary')
            summary = (
                str(summary_value).strip() if summary_value is not None else ''
            )
            release_date = _format_first_release_date(
                item.get('first_release_date')
            )
            category = _igdb_category_display(item.get('category'))
            companies = item.get('involved_companies')
            developers = ', '.join(_collect_company_names(companies, 'developer'))
            publishers = ', '.join(_collect_company_names(companies, 'publisher'))
            genres = _format_name_list(item.get('genres'))
            game_modes = _format_name_list(item.get('game_modes'))
            platforms = _format_name_list(item.get('platforms'))
            cover_url = cover_url_from_cover(item.get('cover'))
            rating_count = _coerce_rating_count(
                item.get('total_rating_count'), item.get('rating_count')
            )
            igdb_id = coerce_igdb_id(item.get('id')) if 'id' in item else ''

            records.append(
                {
                    'Name': name,
                    'Summary': summary,
                    'First Launch Date': release_date,
                    'Category': category,
                    'Developers': developers,
                    'Publishers': publishers,
                    'Genres': genres,
                    'Game Modes': game_modes,
                    'Platforms': platforms,
                    'Large Cover Image (URL)': cover_url,
                    'Rating Count': rating_count if rating_count is not None else 0,
                    'IGDB ID': igdb_id,
                    'igdb_id': igdb_id,
                }
            )

        df = pd.DataFrame(records)
        if df.empty:
            return pd.DataFrame(columns=columns)

        df.insert(0, 'Source Index', [str(i) for i in range(len(df))])
        if 'Rating Count' in df.columns:
            df['Rating Count'] = (
                pd.to_numeric(df['Rating Count'], errors='coerce').fillna(0)
            )
            df = df.sort_values(
                by='Rating Count', ascending=False, kind='mergesort'
            ).reset_index(drop=True)
            df['Source Index'] = [str(i) for i in range(len(df))]

        return _finalize_dataframe(df)

    loaders: list[Callable[[], tuple[pd.DataFrame, bool]]] = []
    if prefer_cache:
        loaders = [_load_from_cache, _load_from_db]
    else:
        loaders = [_load_from_db, _load_from_cache]

    for loader in loaders:
        frame, has_rows = loader()
        if has_rows:
            return frame

    return _load_from_igdb()




def _encode_cover_image(img: Image.Image) -> str:
    prepared = _prepare_cover_image(img)
    buf = io.BytesIO()
    prepared.save(buf, format='JPEG')
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()


def cover_data_from_path(cover_path: str | None) -> str | None:
    if not cover_path:
        return None
    if not os.path.exists(cover_path):
        app.logger.warning("Cover path %s missing", cover_path)
        return None
    try:
        img = open_image_auto_rotate(cover_path)
    except Exception:
        app.logger.warning("Failed to open cover path %s", cover_path)
        return None
    return _encode_cover_image(img)


def cover_data_from_url(url: str | None) -> str | None:
    if not url:
        return None
    parsed_path = urlparse(str(url)).path
    base = os.path.splitext(os.path.basename(parsed_path))[0]
    for ext in ('.jpg', '.jpeg', '.png'):
        path = os.path.join(COVERS_DIR, base + ext)
        if os.path.exists(path):
            img = open_image_auto_rotate(path)
            return _encode_cover_image(img)
    try:
        with urlopen(url) as resp:
            img = open_image_auto_rotate(resp)
            return _encode_cover_image(img)
    except Exception:
        app.logger.warning("No cover found for URL %s", url)
    return None


def load_cover_data(cover_path: str | None = None, fallback_url: str | None = None) -> str | None:
    cover = cover_data_from_path(cover_path)
    if cover:
        return cover
    return cover_data_from_url(fallback_url)


def resolve_cover(
    *,
    cover_data: str | None = None,
    cover_path: str | None = None,
    cover_url: str | None = None,
    placeholder: str = '/static/no-image.jpg',
) -> str:
    """Resolve a best-effort cover source URL for UI consumption."""

    if cover_data:
        return cover_data

    normalized_path = str(cover_path or '').strip()
    if normalized_path:
        candidates = [normalized_path]
        if not os.path.isabs(normalized_path):
            joined = os.path.join(PROCESSED_DIR, normalized_path)
            if joined not in candidates:
                candidates.append(joined)
        for candidate in candidates:
            if os.path.exists(candidate):
                cover = cover_data_from_path(candidate)
                if cover:
                    return cover

    normalized_url = str(cover_url or '').strip()
    if normalized_url:
        return normalized_url

    fallback = str(placeholder or '').strip() or '/static/no-image.jpg'
    if fallback.startswith(('http://', 'https://', 'data:')):
        return fallback

    normalized_fallback = fallback
    if normalized_fallback.startswith('/static/'):
        normalized_fallback = normalized_fallback[len('/static/'):]
    elif normalized_fallback.startswith('static/'):
        normalized_fallback = normalized_fallback[len('static/'):]
    elif normalized_fallback.startswith('/'):
        # Non-static absolute paths should be returned unchanged.
        return fallback

    normalized_fallback = normalized_fallback or 'no-image.jpg'

    try:
        return url_for('static', filename=normalized_fallback)
    except RuntimeError:
        # ``url_for`` requires an application context. When unavailable fall back
        # to the conventional static path.
        return f"/static/{normalized_fallback}"


def find_cover(row: pd.Series) -> str | None:
    igdb_id = extract_igdb_id(row, allow_generic_id=True)
    prefill = get_igdb_prefill_for_id(igdb_id)
    if prefill:
        url = prefill.get('Large Cover Image (URL)')
        if url:
            data = cover_data_from_url(url)
            if data:
                return data
    url = str(row.get('Large Cover Image (URL)', '') or '')
    if not url:
        return None
    return cover_data_from_url(url)


def ensure_dirs() -> None:
    for d in (UPLOAD_DIR, PROCESSED_DIR, COVERS_DIR):
        os.makedirs(d, exist_ok=True)


def normalize_processed_games() -> None:
    """Resequence IDs based on the order of ``Source Index``."""
    with db_lock:
        conn = get_db()
        with conn:
            cur = conn.execute(
                'SELECT "ID", "Source Index" FROM processed_games '
                'ORDER BY CAST("Source Index" AS INTEGER)'
            )
            rows = cur.fetchall()
            for new_id, row in enumerate(rows, start=1):
                old_id = row["ID"]
                src_index = row["Source Index"]
                conn.execute(
                    'UPDATE processed_games SET "ID"=? WHERE "Source Index"=?',
                    (-new_id, src_index),
                )
                if old_id is not None:
                    conn.execute(
                        'UPDATE igdb_updates SET processed_game_id=? WHERE processed_game_id=?',
                        (-new_id, old_id),
                    )
            conn.execute('UPDATE processed_games SET "ID" = -"ID"')
            conn.execute('UPDATE igdb_updates SET processed_game_id = -processed_game_id')


def seed_processed_games_from_source() -> None:
    """Ensure ``processed_games`` has a seeded row for each IGDB source entry."""

    games_df = catalog_state.games_df
    if games_df is None or games_df.empty:
        return

    def _coerce_source_index(value: Any) -> str | None:
        return processed_navigator.GameNavigator.canonical_source_index(value)

    with db_lock:
        conn = get_db()
        with conn:
            processed_columns = get_processed_games_columns(conn)
            has_cache_rank = 'cache_rank' in processed_columns

            if has_cache_rank:
                try:
                    cache_row = conn.execute(
                        'SELECT MAX(cache_rank) AS max_rank FROM processed_games'
                    ).fetchone()
                except sqlite3.OperationalError:
                    cache_row = None
                max_cache_rank = cache_row['max_rank'] if cache_row else None
                try:
                    current_max = int(max_cache_rank) if max_cache_rank is not None else 0
                except (TypeError, ValueError):
                    current_max = 0
                next_cache_rank = current_max + 1
            else:
                next_cache_rank = 0

            try:
                _load_lookup_tables(conn)
            except Exception:
                pass

            cur = conn.execute(
                'SELECT "Source Index", "Name", "Summary", "Cover Path" FROM processed_games'
            )
            existing: dict[str, tuple[str, str, bool]] = {}
            for row in cur.fetchall():
                stored_source = row['Source Index']
                if stored_source is None:
                    continue
                stored_text = str(stored_source)
                canonical = _coerce_source_index(stored_source)
                if canonical is None:
                    canonical = stored_text if stored_text else None
                if canonical is None:
                    continue
                try:
                    summary_value = row['Summary']
                except (KeyError, IndexError, TypeError):
                    summary_value = None
                try:
                    cover_value = row['Cover Path']
                except (KeyError, IndexError, TypeError):
                    cover_value = None
                existing[canonical] = (
                    stored_text,
                    _normalize_seed_value(row['Name']),
                    is_processed_game_done(summary_value, cover_value),
                )

            igdb_ids: set[str] = set()
            if 'igdb_id' in games_df.columns:
                for value in games_df['igdb_id'].tolist():
                    normalized = coerce_igdb_id(value)
                    if normalized:
                        igdb_ids.add(normalized)
            metadata_map = (
                fetch_igdb_metadata(igdb_ids, conn=conn) if igdb_ids else {}
            )

            source_values = (
                games_df['Source Index'].tolist()
                if 'Source Index' in games_df.columns
                else None
            )
            name_values = (
                games_df['Name'].tolist()
                if 'Name' in games_df.columns
                else None
            )
            row_count = len(games_df.index)
            for position in range(row_count):
                raw_source = (
                    source_values[position]
                    if source_values is not None and position < len(source_values)
                    else position
                )
                src_index = _coerce_source_index(raw_source)
                if src_index is None:
                    continue

                if name_values is not None and position < len(name_values):
                    name_value = name_values[position]
                else:
                    name_value = (
                        games_df.iloc[position].get('Name')
                        if position < row_count
                        else None
                    )
                igdb_name = _normalize_seed_value(name_value)
                stored_name: str | None = igdb_name if igdb_name else None

                row = games_df.iloc[position]
                igdb_id = extract_igdb_id(row, allow_generic_id=True)
                metadata = metadata_map.get(igdb_id) if igdb_id else None

                cache_rank: int | None = None
                if has_cache_rank:
                    cache_rank, next_cache_rank = _determine_cache_rank(
                        metadata, row, next_cache_rank
                    )

                if src_index not in existing:
                    processed_row = _igdb_to_processed_row(
                        row=row,
                        src_index=src_index,
                        name=stored_name,
                        igdb_id=igdb_id,
                        metadata=metadata,
                        processed_columns=processed_columns,
                        conn=conn,
                        cache_rank=cache_rank,
                    )
                    columns = list(processed_row.keys())
                    column_sql = ', '.join(
                        db_utils._quote_identifier(column) for column in columns
                    )
                    placeholders = ', '.join('?' for _ in columns)
                    conn.execute(
                        f'INSERT OR IGNORE INTO processed_games ({column_sql}) VALUES ({placeholders})',
                        [processed_row[column] for column in columns],
                    )
                    existing[src_index] = (src_index, igdb_name, False)
                    continue

                stored_source, existing_name, is_done = existing[src_index]
                if has_cache_rank and cache_rank is not None:
                    conn.execute(
                        'UPDATE processed_games SET cache_rank=? WHERE "Source Index"=?',
                        (cache_rank, src_index),
                    )
                if is_done:
                    continue
                if stored_source != src_index:
                    conn.execute(
                        'UPDATE processed_games SET "Source Index"=? WHERE "Source Index"=?',
                        (src_index, stored_source),
                    )
                    stored_source = src_index
                if igdb_name and existing_name != igdb_name:
                    conn.execute(
                        'UPDATE processed_games SET "Name"=? WHERE "Source Index"=?',
                        (igdb_name, src_index),
                    )
                    existing_name = igdb_name
                existing[src_index] = (stored_source, existing_name, is_done)


def backfill_igdb_ids() -> None:
    games_df = catalog_state.games_df
    if games_df is None or games_df.empty:
        return
    with db_lock:
        conn = get_db()
        with conn:
            cur = conn.execute(
                'SELECT "Source Index", "igdb_id", "Summary", "Cover Path" '
                'FROM processed_games'
            )
            rows = cur.fetchall()
            for row in rows:
                igdb_id_value = row['igdb_id']
                if igdb_id_value:
                    continue
                try:
                    summary_value = row['Summary']
                except (KeyError, IndexError, TypeError):
                    summary_value = None
                try:
                    cover_value = row['Cover Path']
                except (KeyError, IndexError, TypeError):
                    cover_value = None
                if is_processed_game_done(summary_value, cover_value):
                    continue
                src_index = row['Source Index']
                position = get_position_for_source_index(src_index)
                if position is None or position < 0 or position >= len(games_df):
                    continue
                candidate = extract_igdb_id(
                    games_df.iloc[position], allow_generic_id=True
                )
                if candidate:
                    conn.execute(
                        'UPDATE processed_games SET "igdb_id"=? WHERE "Source Index"=?',
                        (candidate, src_index),
                    )


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_igdb_payload(item: Mapping[str, Any]) -> dict[str, Any] | None:
    return igdb_api_client.normalize_game(item)


def _serialize_cache_list(values: Iterable[Any]) -> str:
    items: list[str] = []
    for value in values or []:
        text = str(value).strip()
        if text:
            items.append(text)
    return json.dumps(items, ensure_ascii=False)


def _deserialize_cache_list(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return [str(item).strip() for item in raw if str(item).strip()]
    try:
        data = json.loads(raw)
    except Exception:
        text = str(raw).strip()
        return [text] if text else []
    if not isinstance(data, list):
        return []
    return [str(item).strip() for item in data if str(item).strip()]


def fetch_igdb_metadata(
    igdb_ids: Iterable[str], conn: sqlite3.Connection | None = None
) -> dict[str, dict[str, Any]]:
    numeric_ids: list[int] = []
    id_map: dict[int, str] = {}
    for value in igdb_ids:
        normalized = coerce_igdb_id(value)
        if not normalized:
            continue
        try:
            numeric = int(normalized)
        except (TypeError, ValueError):
            logger.warning('Skipping invalid IGDB id %s', value)
            continue
        if numeric in id_map:
            continue
        id_map[numeric] = normalized
        numeric_ids.append(numeric)

    if not numeric_ids:
        return {}

    if conn is None:
        conn = get_db()

    results: dict[str, dict[str, Any]] = {}
    batch_size = 500
    for start in range(0, len(numeric_ids), batch_size):
        chunk = numeric_ids[start : start + batch_size]
        if not chunk:
            continue
        placeholders = ','.join('?' for _ in chunk)
        query = (
            f'SELECT igdb_id, name, summary, updated_at, first_release_date, '
            f'category, cover_image_id, rating_count, developers, publishers, '
            f'genres, platforms, game_modes '
            f'FROM {IGDB_CACHE_TABLE} '
            f'WHERE igdb_id IN ({placeholders})'
        )
        try:
            cur = conn.execute(query, tuple(chunk))
        except sqlite3.OperationalError as exc:
            logger.warning('Failed to query IGDB cache: %s', exc)
            return {}
        for row in cur.fetchall():
            igdb_id = row['igdb_id']
            normalized = id_map.get(igdb_id)
            key = normalized if normalized is not None else str(igdb_id)
            cover_id = row['cover_image_id']
            results[str(key)] = {
                'id': igdb_id,
                'name': row['name'],
                'summary': row['summary'],
                'updated_at': row['updated_at'],
                'first_release_date': row['first_release_date'],
                'category': row['category'],
                'cover': {'image_id': cover_id} if cover_id else None,
                'rating_count': row['rating_count'],
                'developers': _deserialize_cache_list(row['developers']),
                'publishers': _deserialize_cache_list(row['publishers']),
                'genres': _deserialize_cache_list(row['genres']),
                'platforms': _deserialize_cache_list(row['platforms']),
                'game_modes': _deserialize_cache_list(row['game_modes']),
            }
    return results


def exchange_twitch_credentials(*, force_refresh: bool = False) -> tuple[str, str]:
    """Compatibility wrapper that proxies to :mod:`igdb.client`."""

    return igdb_api_client.exchange_twitch_credentials(
        request_factory=Request,
        opener=urlopen,
        force_refresh=force_refresh,
    )


def download_igdb_metadata(
    access_token: str, client_id: str, igdb_ids: Iterable[str]
) -> dict[str, dict[str, Any]]:
    """Fetch IGDB metadata using the shared client helpers."""

    batch_size = (
        IGDB_BATCH_SIZE
        if isinstance(IGDB_BATCH_SIZE, int) and IGDB_BATCH_SIZE > 0
        else 500
    )
    return igdb_api_client.fetch_metadata_by_ids(
        access_token,
        client_id,
        igdb_ids,
        batch_size=batch_size,
        user_agent=IGDB_USER_AGENT,
        request_factory=Request,
        opener=urlopen,
    )


def download_igdb_game_count(access_token: str, client_id: str) -> int:
    """Return the IGDB game count via the shared client module."""

    return igdb_api_client.fetch_game_count(
        access_token,
        client_id,
        user_agent=IGDB_USER_AGENT,
        request_factory=Request,
        opener=urlopen,
    )


def download_igdb_games(
    access_token: str, client_id: str, offset: int, limit: int
) -> list[dict[str, Any]]:
    """Download and normalize a page of IGDB games."""

    return igdb_api_client.fetch_games(
        access_token,
        client_id,
        offset,
        limit,
        user_agent=IGDB_USER_AGENT,
        request_factory=Request,
        opener=urlopen,
    )


def _build_cache_row_from_payload(
    payload: Mapping[str, Any]
) -> dict[str, Any] | None:
    normalized = payload
    if 'id' not in normalized:
        normalized = _normalize_igdb_payload(payload)
        if normalized is None:
            return None

    igdb_id = normalized.get('id')
    if igdb_id is None:
        return None
    try:
        numeric_id = int(igdb_id)
    except (TypeError, ValueError):
        return None

    def _clean_text(value: Any) -> str | None:
        if value is None:
            return None
        text = value.strip() if isinstance(value, str) else str(value).strip()
        return text or None

    cover = normalized.get('cover')
    cover_image_id: str | None = None
    if isinstance(cover, Mapping):
        cover_value = cover.get('image_id') or cover.get('imageId')
        if cover_value is not None:
            cover_image_id = str(cover_value).strip() or None
    elif isinstance(cover, str):
        cover_image_id = cover.strip() or None

    rating_value = normalized.get('rating_count')
    if isinstance(rating_value, numbers.Number):
        rating_count = int(rating_value)
    else:
        try:
            rating_count = int(str(rating_value).strip())
        except (TypeError, ValueError):
            rating_count = None

    developers_json = _serialize_cache_list(normalized.get('developers') or [])
    publishers_json = _serialize_cache_list(normalized.get('publishers') or [])
    genres_json = _serialize_cache_list(normalized.get('genres') or [])
    platforms_json = _serialize_cache_list(normalized.get('platforms') or [])
    game_modes_json = _serialize_cache_list(normalized.get('game_modes') or [])

    return {
        'igdb_id': numeric_id,
        'name': _clean_text(normalized.get('name')),
        'summary': _clean_text(normalized.get('summary')),
        'updated_at': normalized.get('updated_at'),
        'first_release_date': normalized.get('first_release_date'),
        'category': normalized.get('category'),
        'cover_image_id': cover_image_id,
        'rating_count': rating_count,
        'developers_json': developers_json,
        'publishers_json': publishers_json,
        'genres_json': genres_json,
        'platforms_json': platforms_json,
        'game_modes_json': game_modes_json,
    }


def _upsert_igdb_cache_entries(
    conn: sqlite3.Connection, payloads: Iterable[Mapping[str, Any]]
) -> tuple[int, int, int]:
    return igdb_cache.upsert_igdb_games(conn, payloads)


def _get_cached_igdb_total(conn: sqlite3.Connection) -> int | None:
    return igdb_cache.get_cached_total(conn)


def _set_cached_igdb_total(
    conn: sqlite3.Connection, total: int | None, synced_at: str | None = None
) -> None:
    igdb_cache.set_cached_total(conn, total, synced_at=synced_at)




_igdb_prefill_cache: dict[str, dict[str, Any]] = {}
_igdb_prefill_lock = Lock()


def _dedupe_normalized_names(values: Iterable[str]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = _normalize_lookup_name(value)
        if not normalized:
            continue
        fingerprint = normalized.casefold()
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        names.append(normalized)
    return names


def _resolve_lookup_ids(
    conn: sqlite3.Connection, table_name: str, names: Iterable[str]
) -> list[int]:
    ids: list[int] = []
    seen: set[int] = set()
    for name in _dedupe_normalized_names(names):
        lookup_id = _get_or_create_lookup_id(conn, table_name, name)
        if lookup_id is None or lookup_id in seen:
            continue
        seen.add(lookup_id)
        ids.append(lookup_id)
    return ids


def _determine_cache_rank(
    metadata: Mapping[str, Any] | None,
    row: Mapping[str, Any],
    next_cache_rank: int,
) -> tuple[int | None, int]:
    candidates: list[Any] = []
    if metadata is not None:
        candidates.append(metadata.get('rating_count'))
    for key in ('Rating Count', 'rating_count'):
        if isinstance(row, Mapping) and key in row:
            candidates.append(row.get(key))

    rating_count: int | None = None
    for candidate in candidates:
        if candidate in (None, ''):
            continue
        try:
            if isinstance(candidate, numbers.Real) and not isinstance(candidate, bool):
                rating_count = int(candidate)
            else:
                rating_count = int(float(str(candidate).strip()))
        except (TypeError, ValueError):
            continue
        else:
            break

    if rating_count is not None:
        if rating_count < 0:
            rating_count = abs(rating_count)
        return -rating_count, next_cache_rank

    cache_rank = next_cache_rank
    return cache_rank, next_cache_rank + 1


def _igdb_to_processed_row(
    *,
    row: Mapping[str, Any],
    src_index: str,
    name: str | None,
    igdb_id: str | None,
    metadata: Mapping[str, Any] | None,
    processed_columns: Collection[str],
    conn: sqlite3.Connection,
    cache_rank: int | None,
) -> dict[str, Any]:
    processed: dict[str, Any] = {'Source Index': src_index}

    overlay: dict[str, Any] = {}
    normalized_igdb_id = coerce_igdb_id(igdb_id) if igdb_id else ''
    if metadata is not None:
        overlay = _igdb_metadata_to_source_values(metadata, normalized_igdb_id or '')
        if not normalized_igdb_id:
            normalized_igdb_id = coerce_igdb_id(metadata.get('id'))

    processed_name = name or _normalize_seed_value(row.get('Name'))
    processed['Name'] = processed_name or None

    summary_value = overlay.get('Summary') or row.get('Summary')
    if 'Summary' in processed_columns:
        summary = _normalize_seed_value(summary_value)
        processed['Summary'] = summary or None

    if 'First Launch Date' in processed_columns:
        launch_value = overlay.get('First Launch Date') or row.get('First Launch Date')
        launch_text = _normalize_seed_value(launch_value)
        processed['First Launch Date'] = launch_text or None

    if 'Category' in processed_columns:
        category_value = overlay.get('Category') or row.get('Category')
        category_text = _normalize_seed_value(category_value)
        processed['Category'] = category_text or None

    developer_names: list[str] = []
    publisher_names: list[str] = []
    platform_names: list[str] = []
    genre_names: list[str] = []
    mode_names: list[str] = []

    if metadata is not None:
        developer_names = _dedupe_normalized_names(
            _parse_company_names(metadata.get('developers'))
        )
        publisher_names = _dedupe_normalized_names(
            _parse_company_names(metadata.get('publishers'))
        )
        platform_names = _dedupe_normalized_names(
            _parse_iterable(metadata.get('platforms'))
        )
        genre_names = _dedupe_normalized_names(
            map_igdb_genres(_parse_iterable(metadata.get('genres')))
        )
        mode_names = _dedupe_normalized_names(
            map_igdb_modes(_parse_iterable(metadata.get('game_modes')))
        )

    if not developer_names:
        developer_names = _dedupe_normalized_names(
            _parse_iterable(overlay.get('Developers') or row.get('Developers'))
        )
    if not publisher_names:
        publisher_names = _dedupe_normalized_names(
            _parse_iterable(overlay.get('Publishers') or row.get('Publishers'))
        )
    if not platform_names:
        platform_names = _dedupe_normalized_names(
            _parse_iterable(overlay.get('Platforms') or row.get('Platforms'))
        )
    if not genre_names:
        genre_names = _dedupe_normalized_names(
            map_igdb_genres(
                _parse_iterable(overlay.get('Genres') or row.get('Genres'))
            )
        )
    if not mode_names:
        mode_names = _dedupe_normalized_names(
            map_igdb_modes(
                _parse_iterable(overlay.get('Game Modes') or row.get('Game Modes'))
            )
        )

    if 'Developers' in processed_columns:
        processed['Developers'] = ', '.join(developer_names) if developer_names else None
    if 'developers_ids' in processed_columns:
        developer_ids = _resolve_lookup_ids(conn, 'developers', developer_names)
        processed['developers_ids'] = (
            _encode_lookup_id_list(developer_ids) if developer_ids else ''
        )

    if 'Publishers' in processed_columns:
        processed['Publishers'] = ', '.join(publisher_names) if publisher_names else None
    if 'publishers_ids' in processed_columns:
        publisher_ids = _resolve_lookup_ids(conn, 'publishers', publisher_names)
        processed['publishers_ids'] = (
            _encode_lookup_id_list(publisher_ids) if publisher_ids else ''
        )

    if 'Platforms' in processed_columns:
        processed['Platforms'] = ', '.join(platform_names) if platform_names else None
    if 'platforms_ids' in processed_columns:
        platform_ids = _resolve_lookup_ids(conn, 'platforms', platform_names)
        processed['platforms_ids'] = (
            _encode_lookup_id_list(platform_ids) if platform_ids else ''
        )

    if 'Genres' in processed_columns:
        processed['Genres'] = ', '.join(genre_names) if genre_names else None
    if 'genres_ids' in processed_columns:
        genre_ids = _resolve_lookup_ids(conn, 'genres', genre_names)
        processed['genres_ids'] = (
            _encode_lookup_id_list(genre_ids) if genre_ids else ''
        )

    if 'Game Modes' in processed_columns:
        processed['Game Modes'] = ', '.join(mode_names) if mode_names else None
    if 'game_modes_ids' in processed_columns:
        mode_ids = _resolve_lookup_ids(conn, 'game_modes', mode_names)
        processed['game_modes_ids'] = (
            _encode_lookup_id_list(mode_ids) if mode_ids else ''
        )

    if 'igdb_id' in processed_columns:
        processed['igdb_id'] = normalized_igdb_id or ''

    if 'cache_rank' in processed_columns and cache_rank is not None:
        processed['cache_rank'] = cache_rank

    return processed


def _should_replace_with_prefill(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    if isinstance(value, str):
        text = value.strip()
        return not text or text.lower() == 'nan'
    text = str(value).strip()
    return not text or text.lower() == 'nan'


def _igdb_metadata_to_source_values(
    metadata: Mapping[str, Any], igdb_id: str
) -> dict[str, Any]:
    overlay: dict[str, Any] = {}

    name_value = metadata.get('name')
    if isinstance(name_value, str) and name_value.strip():
        overlay['Name'] = name_value.strip()

    summary_value = metadata.get('summary')
    if isinstance(summary_value, str) and summary_value.strip():
        overlay['Summary'] = summary_value.strip()

    release_date = _format_first_release_date(metadata.get('first_release_date'))
    if release_date:
        overlay['First Launch Date'] = release_date

    category = _igdb_category_display(metadata.get('category'))
    if category:
        overlay['Category'] = category

    developers = _dedupe_normalized_names(_parse_company_names(metadata.get('developers')))
    if developers:
        overlay['Developers'] = ', '.join(developers)

    publishers = _dedupe_normalized_names(_parse_company_names(metadata.get('publishers')))
    if publishers:
        overlay['Publishers'] = ', '.join(publishers)

    genre_names = _dedupe_normalized_names(
        map_igdb_genres(_parse_iterable(metadata.get('genres')))
    )
    if genre_names:
        overlay['Genres'] = ', '.join(genre_names)

    mode_names = _dedupe_normalized_names(
        map_igdb_modes(_parse_iterable(metadata.get('game_modes')))
    )
    if mode_names:
        overlay['Game Modes'] = ', '.join(mode_names)

    platform_names = _dedupe_normalized_names(_parse_iterable(metadata.get('platforms')))
    if platform_names:
        overlay['Platforms'] = ', '.join(platform_names)

    cover_url = cover_url_from_cover(metadata.get('cover'), size='t_original')
    if cover_url:
        overlay['Large Cover Image (URL)'] = cover_url

    if igdb_id:
        overlay['IGDB ID'] = igdb_id
        overlay['igdb_id'] = igdb_id

    return overlay


def get_igdb_prefill_for_id(igdb_id: str | None) -> dict[str, Any] | None:
    if not igdb_id:
        return None
    normalized = coerce_igdb_id(igdb_id)
    if not normalized:
        return None

    with _igdb_prefill_lock:
        cached = _igdb_prefill_cache.get(normalized)
    if cached is not None:
        return dict(cached)

    metadata_map = fetch_igdb_metadata([normalized])

    if not metadata_map:
        return None

    item = metadata_map.get(normalized)
    if item is None:
        item = metadata_map.get(str(int(normalized))) if normalized.isdigit() else None
    if not item:
        return None

    overlay = _igdb_metadata_to_source_values(item, normalized)

    with _igdb_prefill_lock:
        _igdb_prefill_cache[normalized] = dict(overlay)

    return dict(overlay)


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
    'first_release_date': ('First Launch Date', 'timestamp'),
    'genres': ('Genres', 'list'),
    'platforms': ('Platforms', 'list'),
    'game_modes': ('Game Modes', 'list'),
    'developers': ('Developers', 'company_list'),
    'publishers': ('Publishers', 'company_list'),
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
            remote_items = _parse_iterable(remote_value)
            if local_field == 'Genres':
                remote_items = map_igdb_genres(remote_items)
            elif local_field == 'Game Modes':
                remote_items = map_igdb_modes(remote_items)
            remote_set = set(remote_items)
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





def _set_games_dataframe(
    df: pd.DataFrame | None,
    *,
    rebuild_metadata: bool = True,
    rebuild_navigator: bool = True,
) -> None:
    try:
        catalog_state.set_games_dataframe(
            df,
            rebuild_metadata=rebuild_metadata,
            rebuild_navigator=rebuild_navigator,
        )
    except Exception:
        logger.exception('Failed to rebuild navigator state')


def refresh_processed_games_from_cache() -> None:
    """Reload IGDB cache into ``processed_games`` when requested."""

    igdb_frame = load_games(prefer_cache=True)

    try:
        if igdb_frame is not None and not igdb_frame.empty:
            _set_games_dataframe(
                igdb_frame,
                rebuild_metadata=False,
                rebuild_navigator=False,
            )

        seed_processed_games_from_source()
        normalize_processed_games()
        backfill_igdb_ids()
    finally:
        updated_frame = load_games()
        _set_games_dataframe(
            updated_frame,
            rebuild_metadata=True,
            rebuild_navigator=True,
        )


def extract_list(row: pd.Series, keys: list[str]) -> list[str]:
    """Return a list of comma-separated values from the first matching key."""
    for key in keys:
        if key in row.index:
            val = row.get(key, '')
            if pd.isna(val):
                return []
            return [g.strip() for g in str(val).split(',') if g.strip()]
    return []


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
    placeholder = (
        "Sinopse temporariamente indisponível. Preencha manualmente ou tente novamente"
        f" para '{game_name}'."
    )
    if not OPENAI_SUMMARY_ENABLED or client is None:
        return placeholder

    try:
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
                        "Escreva uma sinopse um pouco mais longa (3 a 5 frases) para o jogo "
                        f"'{game_name}'."
                    ),
                },
            ],
            temperature=0.7,
            max_tokens=200,
        )
    except Exception as exc:  # pragma: no cover - relies on external service
        logger.warning("OpenAI summary generation failed: %s", exc)
        return placeholder

    message = (response.choices or [None])[0]
    content = getattr(message, 'message', getattr(message, 'text', None))
    if content:
        return getattr(content, 'content', content).strip()

    logger.warning("OpenAI response missing content for game '%s'", game_name)
    return placeholder


# initial load
db = initialize_app(
    ensure_dirs=ensure_dirs,
    init_db=_init_db,
    load_games=load_games,
    set_games_dataframe=_set_games_dataframe,
    connection_factory=_db_connection_factory,
    run_migrations=RUN_DB_MIGRATIONS,
)

def build_game_payload(index: int, seq: int, progress_seq: int | None = None) -> dict:
    games_df = catalog_state.games_df
    total_games = catalog_state.total_games
    try:
        row = games_df.iloc[index]
        source_index = get_source_index_for_position(index)
    except Exception:
        raise IndexError('invalid index')
    processed_row = None
    processed_lookup_entries: dict[str, list[dict[str, Any]]] = {}
    with db_lock:
        conn = get_db()
        cur = conn.execute(
            'SELECT * FROM processed_games WHERE "Source Index"=?', (source_index,)
        )
        processed_row = cur.fetchone()
        if processed_row is not None:
            processed_lookup_entries = _fetch_lookup_entries_for_game(
                conn, processed_row['ID']
            )

    processed_mapping: Mapping[str, Any] = (
        dict(processed_row) if processed_row is not None else {}
    )

    source_row = row.copy()
    processed_cover_path = None
    fallback_cover_url = str(source_row.get('Large Cover Image (URL)', '') or '')

    if processed_mapping:
        processed_cover_path = processed_mapping.get('Cover Path') or None
        for key, value in processed_mapping.items():
            if key not in source_row.index:
                continue
            if _should_replace_with_prefill(value):
                continue
            source_row.at[key] = value
            if key == 'Large Cover Image (URL)' and value:
                fallback_cover_url = str(value)

    igdb_id = extract_igdb_id(source_row, allow_generic_id=True)
    should_prefill = True
    if processed_mapping:
        summary_value = processed_mapping.get('Summary')
        cover_value = processed_mapping.get('Cover Path')
        should_prefill = not is_processed_game_done(summary_value, cover_value)

    if should_prefill:
        prefill = get_igdb_prefill_for_id(igdb_id or extract_igdb_id(row, allow_generic_id=True))
        if prefill:
            for key, value in prefill.items():
                if key in source_row.index:
                    if _should_replace_with_prefill(source_row.get(key)):
                        source_row.at[key] = value
                else:
                    source_row[key] = value
            cover_override = prefill.get('Large Cover Image (URL)')
            if cover_override:
                fallback_cover_url = cover_override
            if not igdb_id:
                igdb_id = prefill.get('IGDB ID') or prefill.get('igdb_id') or igdb_id

    cover_data = load_cover_data(processed_cover_path, fallback_cover_url)
    igdb_id = coerce_igdb_id(igdb_id) if igdb_id else ''

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
            entries = _parse_lookup_entries_from_source(source_row, processed_column)
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

    progress_value = progress_seq if progress_seq is not None else seq

    return {
        'index': int(index),
        'total': total_games,
        'game': game_fields,
        'cover': cover_data,
        'seq': progress_value,
        'id': game_id,
        'missing': missing,
    }














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


def _compute_metadata_updates(
    canonical: sqlite3.Row, duplicates: Iterable[sqlite3.Row]
) -> dict[str, Any]:
    return processed_duplicates.compute_metadata_updates(canonical, duplicates)


def _scan_duplicate_candidates(
    rows: Iterable[sqlite3.Row],
    *,
    progress_callback: Callable[[int, int, int, int], None] | None = None,
) -> tuple[list[DuplicateGroupResolution], int, int, int]:
    return processed_duplicates.scan_duplicate_candidates(
        rows,
        progress_callback=progress_callback,
        relation_count_columns=_RELATION_COUNT_COLUMNS,
    )


def _apply_metadata_updates(
    conn: sqlite3.Connection, processed_game_id: int, updates: Mapping[str, Any]
) -> None:
    if not updates:
        return
    set_fragments: list[str] = []
    params: list[Any] = []
    for column, value in updates.items():
        set_fragments.append(f'{_quote_identifier(column)} = ?')
        params.append(value)
    if not set_fragments:
        return
    params.append(processed_game_id)
    conn.execute(
        f'UPDATE processed_games SET {", ".join(set_fragments)} WHERE "ID" = ?',
        params,
    )


def _refresh_lookup_columns_for_games(
    conn: sqlite3.Connection, processed_game_ids: Iterable[int]
) -> None:
    unique_ids = sorted(
        {
            coerced
            for game_id in processed_game_ids
            for coerced in (_coerce_int(game_id),)
            if coerced is not None
        }
    )
    for game_id in unique_ids:
        entries = _fetch_lookup_entries_for_game(conn, game_id)
        if not entries:
            continue
        _apply_lookup_entries_to_processed_game(conn, game_id, entries)


def _merge_duplicate_resolutions(
    resolutions: Iterable[DuplicateGroupResolution],
) -> set[int]:
    return processed_duplicates.merge_duplicate_resolutions(
        resolutions,
        db_lock=db_lock,
        get_db=get_db,
        lookup_relations=LOOKUP_RELATIONS,
        apply_metadata_updates=_apply_metadata_updates,
        fetch_lookup_entries_for_game=_fetch_lookup_entries_for_game,
        apply_lookup_entries_to_processed_game=_apply_lookup_entries_to_processed_game,
    )


def _remove_processed_games(ids_to_delete: Iterable[int]) -> tuple[int, int]:
    return processed_duplicates.remove_processed_games(
        ids_to_delete,
        catalog_state=catalog_state,
        db_lock=db_lock,
        get_db=get_db,
        navigator_canonical=processed_navigator.GameNavigator.canonical_source_index,
        get_position_for_source_index=get_position_for_source_index,
        normalize_processed_games=normalize_processed_games,
    )


def _coerce_cursor_timestamp(value: Any) -> float:
    if value is None:
        return float('-inf')
    if isinstance(value, (int, float)):
        try:
            return float(value)
        except (TypeError, ValueError):
            return float('-inf')
    text = str(value).strip()
    if not text:
        return float('-inf')
    try:
        return float(text)
    except ValueError:
        pass
    normalized = text.replace('Z', '+00:00')
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return float('-inf')
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).timestamp()


def _entry_type_rank(entry_type: str) -> int:
    return 0 if entry_type == 'm' else 1


def _normalize_sort_numeric(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float('-inf')
    if math.isnan(numeric):
        return float('-inf')
    return numeric


def _collect_updates_list_entries(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    processed_columns = get_processed_games_columns(conn)
    cover_url_select = (
        'p."Large Cover Image (URL)" AS cover_url'
        if 'Large Cover Image (URL)' in processed_columns
        else 'NULL AS cover_url'
    )
    relation_count_sql = ', '.join(
        f'(SELECT COUNT(*) FROM {relation["join_table"]} WHERE processed_game_id = p."ID") AS {relation["join_table"]}_count'
        for relation in LOOKUP_RELATIONS
    )

    order_expr = "COALESCE(u.refreshed_at, u.local_last_edited_at, u.igdb_updated_at, '')"
    mismatch_query = (
        f'''SELECT
               u.processed_game_id,
               u.igdb_id,
               u.igdb_updated_at,
               u.local_last_edited_at,
               u.refreshed_at,
               u.has_diff,
               {order_expr} AS order_value,
               p."Name" AS game_name,
               p."Cover Path" AS cover_path,
               {cover_url_select}
           FROM igdb_updates u
           LEFT JOIN processed_games p ON p."ID" = u.processed_game_id
           ORDER BY {order_expr} DESC, u.processed_game_id DESC
        '''
    )
    mismatch_rows = conn.execute(mismatch_query).fetchall()

    duplicate_order_expr = "COALESCE(p.last_edited_at, CAST(cache.updated_at AS TEXT), '')"
    duplicate_query = (
        f'''SELECT
               p."ID",
               p."Source Index",
               p."Name",
               p."igdb_id",
               p."Summary",
               p."Cover Path",
               p."First Launch Date",
               p."Category",
               p."Width",
               p."Height",
               p.last_edited_at,
               {relation_count_sql},
               {cover_url_select},
               cache.updated_at AS cache_updated_at,
               {duplicate_order_expr} AS order_value
           FROM processed_games p
           LEFT JOIN igdb_updates u ON u.processed_game_id = p."ID"
           LEFT JOIN {IGDB_CACHE_TABLE} cache ON cache.igdb_id = p."igdb_id"
           WHERE u.processed_game_id IS NULL
             AND p."igdb_id" IS NOT NULL AND TRIM(p."igdb_id") != ''
             AND EXISTS (
                 SELECT 1 FROM processed_games other
                 WHERE other."igdb_id" = p."igdb_id" AND other."ID" != p."ID"
             )
        '''
    )
    duplicate_rows = conn.execute(duplicate_query).fetchall()

    entries: list[dict[str, Any]] = []
    existing_ids: set[int] = set()

    for row in mismatch_rows:
        processed_id = int(row['processed_game_id']) if row['processed_game_id'] is not None else None
        if processed_id is None:
            continue
        cover_available = bool(row['cover_path'] or row['cover_url'])
        entry = {
            'processed_game_id': processed_id,
            'igdb_id': row['igdb_id'],
            'igdb_updated_at': _normalize_timestamp(row['igdb_updated_at']),
            'local_last_edited_at': _normalize_timestamp(row['local_last_edited_at']),
            'refreshed_at': _normalize_timestamp(row['refreshed_at']),
            'name': row['game_name'],
            'has_diff': bool(row['has_diff']),
            'cover': None,
            'cover_available': cover_available,
            'update_type': 'mismatch',
            'detail_available': True,
            'entry_type': 'm',
            'cursor_value': row['order_value'] or '',
        }
        entry['cursor_numeric'] = _coerce_cursor_timestamp(entry['cursor_value'])
        entry['entry_rank'] = _entry_type_rank(entry['entry_type'])
        entries.append(entry)
        existing_ids.add(processed_id)

    duplicate_resolutions, _, _, _ = _scan_duplicate_candidates(duplicate_rows)
    for resolution in duplicate_resolutions:
        for duplicate_row in resolution.duplicates:
            row_map = dict(duplicate_row)
            processed_id = processed_duplicates.coerce_int(row_map.get('ID'))
            if processed_id is None or processed_id in existing_ids:
                continue
            existing_ids.add(processed_id)
            cover_available = bool(row_map.get('Cover Path') or row_map.get('cover_url'))
            order_value = row_map.get('order_value') or row_map.get('last_edited_at') or row_map.get('cache_updated_at')
            entry = {
                'processed_game_id': processed_id,
                'igdb_id': row_map.get('igdb_id'),
                'igdb_updated_at': _normalize_timestamp(row_map.get('cache_updated_at')),
                'local_last_edited_at': _normalize_timestamp(row_map.get('last_edited_at')),
                'refreshed_at': None,
                'name': row_map.get('Name'),
                'has_diff': False,
                'cover': None,
                'cover_available': cover_available,
                'update_type': 'duplicate',
                'detail_available': False,
                'entry_type': 'd',
                'cursor_value': order_value or '',
            }
            entry['cursor_numeric'] = _coerce_cursor_timestamp(entry['cursor_value'])
            entry['entry_rank'] = _entry_type_rank(entry['entry_type'])
            entries.append(entry)

    entries.sort(
        key=lambda item: (
            -float(item.get('cursor_numeric', float('-inf'))),
            -int(item.get('processed_game_id') or 0),
            item.get('entry_rank', 0),
        )
    )

    return entries


def _populate_updates_list_locked(conn: sqlite3.Connection) -> int:
    entries = _collect_updates_list_entries(conn)
    with conn:
        conn.execute(f'DELETE FROM {UPDATES_LIST_TABLE}')
        if entries:
            conn.executemany(
                f'''INSERT INTO {UPDATES_LIST_TABLE} (
                        processed_game_id,
                        igdb_id,
                        igdb_updated_at,
                        local_last_edited_at,
                        refreshed_at,
                        name,
                        has_diff,
                        cover,
                        cover_available,
                        update_type,
                        detail_available,
                        cursor_value,
                        sort_numeric,
                        entry_type,
                        entry_rank
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                [
                    (
                        entry['processed_game_id'],
                        entry.get('igdb_id'),
                        entry.get('igdb_updated_at'),
                        entry.get('local_last_edited_at'),
                        entry.get('refreshed_at'),
                        entry.get('name'),
                        1 if entry.get('has_diff') else 0,
                        entry.get('cover'),
                        1 if entry.get('cover_available') else 0,
                        entry.get('update_type'),
                        1 if entry.get('detail_available') else 0,
                        entry.get('cursor_value'),
                        _normalize_sort_numeric(entry.get('cursor_numeric')),
                        entry.get('entry_type', 'm'),
                        entry.get('entry_rank', 0),
                    )
                    for entry in entries
                ],
            )
    return len(entries)


def rebuild_updates_list_cache() -> int:
    with db_lock:
        conn = get_db()
        return _populate_updates_list_locked(conn)


def _parse_updates_cursor(cursor: str | None) -> dict[str, Any] | None:
    if not cursor:
        return None
    if not isinstance(cursor, str):
        return None
    decoded_value = cursor
    try:
        padded = cursor + '=' * (-len(cursor) % 4)
        decoded_bytes = base64.urlsafe_b64decode(padded.encode('utf-8'))
        decoded_value = decoded_bytes.decode('utf-8')
    except Exception:
        decoded_value = cursor
    parts = decoded_value.split('|', 2)
    if len(parts) != 3:
        return None
    raw_value, raw_id, raw_type = parts
    try:
        processed_id = int(raw_id)
    except (TypeError, ValueError):
        return None
    entry_type = raw_type or 'm'
    if entry_type not in {'m', 'd'}:
        entry_type = 'm'
    return {
        'sort_value': raw_value,
        'sort_numeric': _coerce_cursor_timestamp(raw_value),
        'processed_game_id': processed_id,
        'entry_type': entry_type,
    }


def _encode_updates_cursor(entry: Mapping[str, Any]) -> str:
    sort_value = str(entry.get('cursor_value') or '')
    processed_id = int(entry.get('processed_game_id') or 0)
    entry_type = str(entry.get('entry_type') or 'm')
    raw = f"{sort_value}|{processed_id}|{entry_type}"
    encoded = base64.urlsafe_b64encode(raw.encode('utf-8')).decode('ascii')
    return encoded.rstrip('=')


def fetch_cached_updates(
    *, cursor: str | None = None, limit: int = 100
) -> tuple[list[dict[str, Any]], int, str | None, bool]:
    normalized_limit = limit if isinstance(limit, int) else 100
    if normalized_limit <= 0:
        normalized_limit = 100
    normalized_limit = min(max(int(normalized_limit), 1), 500)
    fetch_limit = normalized_limit + 1
    marker = _parse_updates_cursor(cursor)

    with db_lock:
        conn = get_db()
        total_row = conn.execute(
            f'SELECT COUNT(*) FROM {UPDATES_LIST_TABLE}'
        ).fetchone()
        total_count_value = int(total_row[0] or 0)
        if total_count_value == 0:
            total_count_value = _populate_updates_list_locked(conn)

        params: list[Any] = []
        where_clause = ''
        if marker is not None:
            marker_numeric = float(marker.get('sort_numeric', float('-inf')))
            marker_id = int(marker.get('processed_game_id') or 0)
            marker_rank = _entry_type_rank(str(marker.get('entry_type', 'm')))
            where_clause = (
                'WHERE l.sort_numeric < ? '
                'OR (l.sort_numeric = ? AND l.processed_game_id < ?) '
                'OR (l.sort_numeric = ? AND l.processed_game_id = ? AND l.entry_rank > ?)'
            )
            params.extend(
                [
                    marker_numeric,
                    marker_numeric,
                    marker_id,
                    marker_numeric,
                    marker_id,
                    marker_rank,
                ]
            )

        processed_columns = get_processed_games_columns(conn)
        cover_url_select = (
            'p."Large Cover Image (URL)" AS cover_source_url'
            if 'Large Cover Image (URL)' in processed_columns
            else 'NULL AS cover_source_url'
        )

        query = (
            f'''SELECT l.processed_game_id, l.igdb_id, l.igdb_updated_at, l.local_last_edited_at,
                       l.refreshed_at, l.name, l.has_diff, l.cover, l.cover_available,
                       l.update_type, l.detail_available, l.cursor_value, l.entry_type,
                       p."Cover Path" AS cover_path,
                       {cover_url_select}
                  FROM {UPDATES_LIST_TABLE} AS l
                  LEFT JOIN processed_games AS p ON p."ID" = l.processed_game_id
                  {where_clause}
                 ORDER BY l.sort_numeric DESC, l.processed_game_id DESC, l.entry_rank ASC
                 LIMIT ?'''
        )
        rows = conn.execute(query, (*params, fetch_limit)).fetchall()

    entries: list[dict[str, Any]] = []
    for row in rows:
        row_map = dict(row)
        resolved_cover_url = resolve_cover(
            cover_path=row_map.get('cover_path'),
            cover_url=row_map.get('cover_source_url'),
        )
        entry = {
            'processed_game_id': row_map.get('processed_game_id'),
            'igdb_id': row_map.get('igdb_id'),
            'igdb_updated_at': row_map.get('igdb_updated_at'),
            'local_last_edited_at': row_map.get('local_last_edited_at'),
            'refreshed_at': row_map.get('refreshed_at'),
            'name': row_map.get('name'),
            'has_diff': bool(row_map.get('has_diff')),
            'cover': row_map.get('cover'),
            'cover_available': bool(row_map.get('cover_available')),
            'cover_path': row_map.get('cover_path'),
            'cover_source_url': row_map.get('cover_source_url'),
            'cover_url': resolved_cover_url,
            'update_type': row_map.get('update_type') or 'mismatch',
            'detail_available': bool(row_map.get('detail_available')),
            'cursor_value': row_map.get('cursor_value') or '',
            'entry_type': row_map.get('entry_type') or 'm',
        }
        entries.append(entry)

    has_more = False
    if len(entries) > normalized_limit:
        has_more = True
        entries = entries[:normalized_limit]

    next_cursor = _encode_updates_cursor(entries[-1]) if has_more and entries else None
    return entries, total_count_value, next_cursor, has_more


def _run_refresh_cache_phase(update_progress: Callable[..., None]) -> Optional[dict[str, Any]]:
    update_progress(message='Preparing IGDB cache…', data={'phase': 'cache'})

    try:
        access_token, client_id = exchange_twitch_credentials()
    except RuntimeError as exc:
        raise RuntimeError(str(exc)) from exc

    with db_lock:
        conn = get_db()
        total = _get_cached_igdb_total(conn)

    if total is None:
        total = download_igdb_game_count(access_token, client_id)
        with db_lock:
            conn = get_db()
            with conn:
                _set_cached_igdb_total(conn, total)

    if total is None or total <= 0:
        with db_lock:
            conn = get_db()
            with conn:
                _set_cached_igdb_total(conn, total)
        update_progress(
            current=0,
            total=total or 0,
            message='IGDB cache is empty.',
            data={'phase': 'cache', 'inserted': 0, 'updated': 0, 'unchanged': 0},
        )
        return {
            'total': total or 0,
            'processed': 0,
            'inserted': 0,
            'updated': 0,
            'unchanged': 0,
        }

    limit = IGDB_BATCH_SIZE if isinstance(IGDB_BATCH_SIZE, int) and IGDB_BATCH_SIZE > 0 else 500
    limit = min(max(int(limit), 1), 500)

    offset = 0
    processed = 0
    inserted_total = 0
    updated_total = 0
    unchanged_total = 0

    while offset < total:
        try:
            access_token, client_id = exchange_twitch_credentials()
        except RuntimeError as exc:
            raise RuntimeError(str(exc)) from exc
        payloads = download_igdb_games(access_token, client_id, offset, limit)
        batch_count = len(payloads)
        with db_lock:
            conn = get_db()
            with conn:
                inserted, updated, unchanged = _upsert_igdb_cache_entries(conn, payloads)
        inserted_total += inserted
        updated_total += updated
        unchanged_total += unchanged
        offset += batch_count
        if batch_count == 0:
            processed = total
            break
        processed = min(offset, total)
        update_progress(
            current=processed,
            total=total,
            message='Refreshing IGDB cache…',
            data={
                'phase': 'cache',
                'inserted': inserted_total,
                'updated': updated_total,
                'unchanged': unchanged_total,
            },
        )

    update_progress(
        current=processed,
        total=total,
        message='Finished refreshing IGDB cache.',
        data={
            'phase': 'cache',
            'inserted': inserted_total,
            'updated': updated_total,
            'unchanged': unchanged_total,
        },
    )

    return {
        'total': total,
        'processed': processed,
        'inserted': inserted_total,
        'updated': updated_total,
        'unchanged': unchanged_total,
    }


def _run_refresh_diff_phase(
    update_progress: Callable[..., None],
    *,
    allowed_cache_ids: set[str] | None = None,
) -> dict[str, Any]:
    processed_rows = _collect_processed_games_with_igdb()

    if allowed_cache_ids is not None:
        if not allowed_cache_ids:
            processed_rows = []
        else:
            filtered_rows: list[dict[str, Any]] = []
            for row in processed_rows:
                normalized = coerce_igdb_id(row.get('igdb_id'))
                if normalized and normalized in allowed_cache_ids:
                    filtered_rows.append(row)
            processed_rows = filtered_rows

    total = len(processed_rows)
    update_progress(
        total=total,
        current=0,
        message='Refreshing update diffs…',
        data={'phase': 'diffs', 'updated': 0, 'missing_count': 0},
    )

    if not processed_rows:
        return {'updated': 0, 'missing': [], 'missing_count': 0, 'total': 0}

    igdb_ids = [row.get('igdb_id') for row in processed_rows]
    igdb_payloads = fetch_igdb_metadata(igdb_ids)
    if igdb_payloads is None:
        igdb_payloads = {}

    updated_count = 0
    missing: list[int] = []
    refreshed_at = now_utc_iso()
    processed = 0

    with db_lock:
        conn = get_db()
        for row in processed_rows:
            igdb_id_value = row.get('igdb_id')
            normalized_id = coerce_igdb_id(igdb_id_value)
            if not normalized_id:
                continue
            try:
                processed_game_id = int(row.get('ID'))
            except (TypeError, ValueError):
                processed_game_id = None
            start_time = time.perf_counter()
            outcome = 'updated'
            try:
                payload = igdb_payloads.get(normalized_id)
                if not payload and normalized_id.isdigit():
                    payload = igdb_payloads.get(str(int(normalized_id)))
                if not payload:
                    outcome = 'missing'
                    if processed_game_id is not None:
                        missing.append(processed_game_id)
                    continue
                igdb_updated_at = _normalize_timestamp(payload.get('updated_at'))
                diff = build_igdb_diff(row, payload)
                has_diff_value = 1 if diff else 0
                conn.execute(
                    '''INSERT INTO igdb_updates (
                            processed_game_id, igdb_id, igdb_updated_at,
                            igdb_payload, diff, local_last_edited_at,
                            refreshed_at, has_diff
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(processed_game_id) DO UPDATE SET
                            igdb_id=excluded.igdb_id,
                            igdb_updated_at=excluded.igdb_updated_at,
                            igdb_payload=excluded.igdb_payload,
                            diff=excluded.diff,
                            local_last_edited_at=excluded.local_last_edited_at,
                            refreshed_at=excluded.refreshed_at,
                            has_diff=excluded.has_diff''',
                    (
                        int(row['ID']),
                        str(igdb_id_value),
                        igdb_updated_at,
                        json.dumps(payload),
                        json.dumps(diff),
                        row.get('last_edited_at') or refreshed_at,
                        refreshed_at,
                        has_diff_value,
                    ),
                )
                updated_count += 1
                processed += 1
                if not diff:
                    outcome = 'unchanged'
            except Exception:
                outcome = 'error'
                raise
            finally:
                elapsed_ms = int((time.perf_counter() - start_time) * 1000)
                logger.info(
                    'updates.diff',
                    extra={
                        'event': 'updates.diff',
                        'igdb_id': normalized_id,
                        'processed_game_id': processed_game_id,
                        'ms': elapsed_ms,
                        'outcome': outcome,
                    },
                )
            if processed % 25 == 0 or processed == total:
                update_progress(
                    current=processed,
                    total=total,
                    message='Refreshing update diffs…',
                    data={
                        'phase': 'diffs',
                        'updated': updated_count,
                        'missing_count': len(missing),
                    },
                )
        conn.commit()
        _populate_updates_list_locked(conn)

    update_progress(
        current=total,
        total=total,
        message='Finished refreshing update diffs.',
        data={'phase': 'diffs', 'updated': updated_count, 'missing_count': len(missing)},
    )

    return {
        'updated': updated_count,
        'missing': missing,
        'missing_count': len(missing),
        'total': total,
        'status_message': (
            f"Fetched {updated_count} update{'s' if updated_count != 1 else ''}."
            + (
                f" {len(missing)} IGDB record{'s' if len(missing) != 1 else ''} missing."
                if missing
                else ''
            )
        ).strip(),
    }


def _execute_refresh_cache_job(
    update_progress: Callable[..., None],
    *,
    offset: int | None = None,
    limit: int | None = None,
    process_all: bool = True,
) -> dict[str, Any]:
    """Background job helper that refreshes the IGDB cache only."""

    update_progress(
        message='Preparing IGDB cache…',
        data={'phase': 'cache', 'inserted': 0, 'updated': 0, 'unchanged': 0},
    )

    try:
        access_token, client_id = exchange_twitch_credentials()
    except RuntimeError as exc:
        raise RuntimeError(str(exc)) from exc

    try:
        current_offset = int(offset or 0)
    except (TypeError, ValueError):
        current_offset = 0
    if current_offset < 0:
        current_offset = 0
    start_offset = current_offset

    if limit is None:
        limit_candidate = IGDB_BATCH_SIZE
    else:
        limit_candidate = limit
    try:
        limit_value = int(limit_candidate)
    except (TypeError, ValueError):
        limit_value = 0
    if limit_value <= 0:
        limit_value = 500
    limit_value = max(1, min(limit_value, 500))

    inserted_total = 0
    updated_total = 0
    unchanged_total = 0
    total_value = 0
    processed_value = 0
    next_offset = current_offset
    done = False

    def _coerce_count(value: Any) -> int:
        try:
            return max(int(value), 0)
        except (TypeError, ValueError):
            return 0

    last_result: dict[str, Any] | None = None

    while True:
        result = updates_service.refresh_igdb_cache(
            access_token,
            client_id,
            current_offset,
            limit_value,
            db_lock=db_lock,
            get_db=get_db,
            get_cached_total=_get_cached_igdb_total,
            set_cached_total=_set_cached_igdb_total,
            download_total=download_igdb_game_count,
            download_games=download_igdb_games,
            upsert_games=_upsert_igdb_cache_entries,
            exchange_credentials=exchange_twitch_credentials,
            igdb_prefill_cache=_igdb_prefill_cache,
            igdb_prefill_lock=_igdb_prefill_lock,
        )

        last_result = dict(result or {})

        if not isinstance(result, Mapping):
            break

        if result.get('status') == 'error':
            message = str(result.get('error') or 'Failed to refresh IGDB cache.')
            update_progress(
                message=f'Error refreshing IGDB cache: {message}',
                data={'phase': 'cache'},
            )
            raise RuntimeError(message)

        inserted_total += _coerce_count(result.get('inserted'))
        updated_total += _coerce_count(result.get('updated'))
        unchanged_total += _coerce_count(result.get('unchanged'))
        total_value = _coerce_count(result.get('total'))
        processed_value = _coerce_count(result.get('processed'))
        next_offset = _coerce_count(result.get('next_offset'))
        done = bool(result.get('done'))
        batch_count = _coerce_count(result.get('batch_count'))

        progress_total = total_value if total_value > 0 else processed_value
        progress_current = (
            min(processed_value, progress_total) if progress_total > 0 else processed_value
        )
        update_progress(
            current=progress_current,
            total=progress_total,
            message='Refreshing IGDB cache…',
            data={
                'phase': 'cache',
                'inserted': inserted_total,
                'updated': updated_total,
                'unchanged': unchanged_total,
            },
        )

        if done or batch_count == 0:
            break

        current_offset = next_offset if next_offset > current_offset else current_offset + batch_count

        if not process_all:
            break

        try:
            access_token, client_id = exchange_twitch_credentials()
        except RuntimeError as exc:
            raise RuntimeError(str(exc)) from exc

    progress_total = total_value if total_value > 0 else processed_value
    progress_current = (
        min(processed_value, progress_total) if progress_total > 0 else processed_value
    )
    update_progress(
        current=progress_current,
        total=progress_total,
        message='Finished refreshing IGDB cache.',
        data={
            'phase': 'cache',
            'inserted': inserted_total,
            'updated': updated_total,
            'unchanged': unchanged_total,
        },
    )

    changed_total = inserted_total + updated_total
    if total_value <= 0:
        status_message = 'IGDB cache is empty.'
    elif changed_total > 0:
        status_message = (
            f"Cached {changed_total} IGDB record{'s' if changed_total != 1 else ''}."
        )
    else:
        status_message = 'IGDB cache is up to date.'

    summary = {
        'status': 'ok',
        'total': total_value,
        'processed': processed_value,
        'inserted': inserted_total,
        'updated': updated_total,
        'unchanged': unchanged_total,
        'done': done or (progress_total > 0 and progress_current >= progress_total),
        'next_offset': next_offset,
        'offset': start_offset,
        'limit': limit_value,
        'message': status_message,
        'toast_type': 'success' if changed_total > 0 else 'info',
    }

    if last_result and 'error' in last_result:
        summary['error'] = last_result['error']

    return summary


def _execute_refresh_job(
    update_progress: Callable[..., None],
    *,
    offset: int | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    cache_summary = None
    cache_error: str | None = None
    try:
        if offset is None and limit is None:
            cache_summary = _run_refresh_cache_phase(update_progress)
        else:
            cache_summary = _execute_refresh_cache_job(
                update_progress,
                offset=offset,
                limit=limit,
                process_all=False,
            )
    except RuntimeError as exc:
        cache_error = str(exc)
    else:
        try:
            refresh_processed_games_from_cache()
        except Exception as exc:
            cache_error = str(exc)
            logger.exception('Failed to refresh processed games from IGDB cache')

    diff_summary = _run_refresh_diff_phase(update_progress)

    messages: list[str] = []
    toast_type = 'success'

    if cache_summary:
        total = cache_summary.get('total') or 0
        inserted = cache_summary.get('inserted') or 0
        updated = cache_summary.get('updated') or 0
        changed = inserted + updated
        if total <= 0:
            messages.append('IGDB cache is empty.')
        elif changed > 0:
            messages.append(
                f"Cached {changed} IGDB record{'s' if changed != 1 else ''}."
            )
        else:
            messages.append('IGDB cache is up to date.')
    elif cache_error:
        messages.append(f'IGDB cache refresh skipped: {cache_error}.')
        toast_type = 'warning'

    if diff_summary:
        messages.append(diff_summary.get('status_message', 'Updates reloaded.'))
        if diff_summary.get('missing_count', 0) > 0:
            toast_type = 'warning'

    if not messages:
        messages.append('Updates reloaded.')

    update_progress(message='Refresh complete.', data={'phase': 'done'})

    return {
        'status': 'ok',
        'cache_summary': cache_summary,
        'diff_summary': diff_summary,
        'message': ' '.join(messages).strip(),
        'toast_type': toast_type,
        'updated': diff_summary.get('updated', 0) if diff_summary else 0,
        'missing': diff_summary.get('missing', []) if diff_summary else [],
        'cache_error': cache_error,
    }




def _execute_compare_updates_job(
    update_progress: Callable[..., None],
    *,
    igdb_ids: Iterable[int | str] | None = None,
) -> dict[str, Any]:
    """Background job helper that recomputes IGDB update diffs only."""

    allowed_cache_ids: set[str] | None = None
    if igdb_ids is not None:
        normalized_ids: set[str] = set()
        for value in igdb_ids:
            normalized = coerce_igdb_id(value)
            if normalized:
                normalized_ids.add(normalized)
        allowed_cache_ids = normalized_ids

    diff_summary = _run_refresh_diff_phase(update_progress, allowed_cache_ids=allowed_cache_ids)

    message = diff_summary.get('status_message', '').strip() if diff_summary else ''
    if not message:
        message = 'Comparison complete.'
    toast_type = 'warning' if diff_summary and diff_summary.get('missing_count', 0) else 'success'

    update_progress(message='Comparison complete.', data={'phase': 'done'})

    return {
        'status': 'ok',
        'diff_summary': diff_summary,
        'message': message,
        'toast_type': toast_type,
        'updated': diff_summary.get('updated', 0) if diff_summary else 0,
        'missing': diff_summary.get('missing', []) if diff_summary else [],
        'missing_count': diff_summary.get('missing_count', 0) if diff_summary else 0,
    }

_blueprints_configured = False


def configure_blueprints(flask_app: Flask) -> None:
    global _blueprints_configured
    if _blueprints_configured:
        return

    routes_games.configure({
        'get_navigator': lambda: _ensure_navigator_dataframe(rebuild_state=False),
        'get_total_games': lambda: catalog_state.total_games,
        'get_games_df': lambda: catalog_state.games_df,
        'build_game_payload': build_game_payload,
        'generate_pt_summary': generate_pt_summary,
        'open_image_auto_rotate': open_image_auto_rotate,
        'save_cover_image': save_cover_image,
        'upload_dir': UPLOAD_DIR,
        'processed_dir': PROCESSED_DIR,
        'db_lock': db_lock,
        'get_db': get_db,
        'get_source_index_for_position': get_source_index_for_position,
        'get_position_for_source_index': get_position_for_source_index,
        'LOOKUP_RELATIONS': LOOKUP_RELATIONS,
        '_resolve_lookup_selection': _resolve_lookup_selection,
        '_lookup_display_text': _lookup_display_text,
        '_encode_lookup_id_list': _encode_lookup_id_list,
        '_persist_lookup_relations': _persist_lookup_relations,
        'is_processed_game_done': is_processed_game_done,
        'now_utc_iso': now_utc_iso,
        'extract_igdb_id': extract_igdb_id,
        'coerce_igdb_id': coerce_igdb_id,
        'get_cell': get_cell,
        'extract_list': extract_list,
        'load_cover_data': load_cover_data,
        'get_igdb_prefill_for_id': get_igdb_prefill_for_id,
    })

    routes_lookups.configure({
        'LOOKUP_TABLES': LOOKUP_TABLES,
        '_format_lookup_label': _format_lookup_label,
        'LOOKUP_ENDPOINT_MAP': LOOKUP_ENDPOINT_MAP,
        'LOOKUP_TABLES_BY_NAME': LOOKUP_TABLES_BY_NAME,
        'LOOKUP_RELATIONS_BY_TABLE': LOOKUP_RELATIONS_BY_TABLE,
        'db_lock': db_lock,
        'get_db': get_db,
        '_normalize_lookup_name': _normalize_lookup_name,
        '_fetch_lookup_entries_for_game': _fetch_lookup_entries_for_game,
        '_lookup_entries_to_selection': _lookup_entries_to_selection,
        '_persist_lookup_relations': _persist_lookup_relations,
        '_apply_lookup_entries_to_processed_game': _apply_lookup_entries_to_processed_game,
        '_remove_lookup_id_from_entries': _remove_lookup_id_from_entries,
        '_list_lookup_entries': _list_lookup_entries,
        '_create_lookup_entry': _create_lookup_entry,
        '_update_lookup_entry': _update_lookup_entry,
        '_delete_lookup_entry': _delete_lookup_entry,
        '_related_processed_game_ids': _related_processed_game_ids,
    })

    routes_updates.configure({
        'IGDB_BATCH_SIZE': IGDB_BATCH_SIZE,
        'validate_igdb_credentials': app_config.validate_igdb_credentials,
        'exchange_twitch_credentials': lambda: exchange_twitch_credentials,
        'db_lock': db_lock,
        'get_db': get_db,
        'now_utc_iso': now_utc_iso,
        '_get_cached_igdb_total': _get_cached_igdb_total,
        '_set_cached_igdb_total': _set_cached_igdb_total,
        'download_igdb_game_count': lambda: download_igdb_game_count,
        'download_igdb_games': lambda: download_igdb_games,
        '_upsert_igdb_cache_entries': _upsert_igdb_cache_entries,
        '_igdb_prefill_lock': _igdb_prefill_lock,
        '_igdb_prefill_cache': _igdb_prefill_cache,
        'refresh_cache_job': _execute_refresh_cache_job,
        '_execute_refresh_job': _execute_refresh_job,
        'compare_updates_job': _execute_compare_updates_job,
        'job_manager': job_manager,
        'fetch_cached_updates': fetch_cached_updates,
        'get_processed_games_columns': get_processed_games_columns,
        'load_cover_data': load_cover_data,
        'resolve_cover': resolve_cover,
        'get_igdb_timeout_count': get_igdb_timeout_count,
        'LOOKUP_RELATIONS': LOOKUP_RELATIONS,
        '_scan_duplicate_candidates': _scan_duplicate_candidates,
        '_coerce_int': _coerce_int,
        '_compute_metadata_updates': _compute_metadata_updates,
        '_merge_duplicate_resolutions': _merge_duplicate_resolutions,
        '_remove_processed_games': _remove_processed_games,
        '_normalize_text': _normalize_text,
        'DuplicateGroupResolution': DuplicateGroupResolution,
    })

    routes_web.configure({
        'app_password': APP_PASSWORD,
        'get_total_games': lambda: catalog_state.total_games,
        'get_categories': catalog_state.get_categories,
        'get_platforms': catalog_state.get_platforms,
    })

    if 'games' not in flask_app.blueprints:
        flask_app.register_blueprint(routes_games.games_blueprint)
    if 'lookups' not in flask_app.blueprints:
        flask_app.register_blueprint(routes_lookups.lookups_blueprint)
    if 'updates' not in flask_app.blueprints:
        flask_app.register_blueprint(routes_updates.updates_blueprint)
    if 'web' not in flask_app.blueprints:
        flask_app.register_blueprint(routes_web.web_blueprint)

    _blueprints_configured = True


from web.app_factory import create_app

app = create_app(app, configure_blueprints=configure_blueprints)


if __name__ == '__main__':
    app.run(debug=True)
