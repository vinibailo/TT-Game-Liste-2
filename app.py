import os
import json
import uuid
import base64
import io
import sqlite3
import numbers
import re
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional
from threading import Lock, Thread
import logging
from bisect import bisect_left

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
from db import utils as db_utils
from igdb import cache as igdb_cache
from igdb import client as igdb_client
from igdb import diff as igdb_diff
from ingestion import data_loader as ingestion_data_loader
from jobs import manager as jobs_manager
from lookups import config as lookups_config
from lookups import service as lookups_service
from media import covers as media_covers
from processed import duplicates as processed_duplicates
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
    igdb_client,
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

INPUT_XLSX = 'igdb_all_games.xlsx'
PROCESSED_DB = 'processed_games.db'
UPLOAD_DIR = 'uploaded_sources'
PROCESSED_DIR = 'processed_covers'
COVERS_DIR = 'covers_out'


def _coerce_positive_float(value: str | None, default: float) -> float:
    try:
        numeric = float(value) if value is not None else default
    except (TypeError, ValueError):
        return default
    return numeric if numeric > 0 else default


SQLITE_TIMEOUT_SECONDS = _coerce_positive_float(
    os.environ.get('SQLITE_TIMEOUT'), 120.0
)


def _coerce_truthy_env(value: str | None) -> bool:
    if value is None:
        return False
    text = value.strip().lower()
    return text in {'1', 'true', 'yes', 'on'}


RUN_DB_MIGRATIONS = _coerce_truthy_env(os.environ.get('RUN_DB_MIGRATIONS'))


def _db_connection_factory() -> sqlite3.Connection:
    return db_utils._create_sqlite_connection(
        PROCESSED_DB, timeout=SQLITE_TIMEOUT_SECONDS
    )

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

LOOKUP_RELATIONS_BY_TABLE = {
    relation['lookup_table']: relation for relation in LOOKUP_RELATIONS
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

FIX_NAMES_BATCH_LIMIT = 50
MAX_BACKGROUND_JOBS = 50


# Global catalog state populated during startup and refresh jobs.
games_df: pd.DataFrame = pd.DataFrame()
_category_values: set[str] = set()
categories_list: list[str] = []
platforms_list: list[str] = []
total_games: int = 0
navigator = None


def _format_lookup_label(value: str) -> str:
    text = str(value or '').replace('_', ' ').strip()
    if not text:
        return ''
    spaced = re.sub(r'(?<!^)(?=[A-Z])', ' ', text)
    return spaced.strip().title()


def has_summary_text(value: Any) -> bool:
    """Return ``True`` when ``value`` contains non-empty summary text."""

    if value is None:
        return False
    if isinstance(value, str):
        text = value.strip()
    else:
        try:
            if pd.isna(value):
                return False
        except Exception:
            pass
        text = str(value).strip()
    if not text:
        return False
    if text.lower() == 'nan':
        return False
    return True


def has_cover_path_value(value: Any) -> bool:
    """Return ``True`` when ``value`` contains a usable cover path."""

    if value is None:
        return False
    if isinstance(value, str):
        text = value.strip()
    else:
        try:
            if pd.isna(value):
                return False
        except Exception:
            pass
        text = str(value).strip()
    if not text:
        return False
    if text.lower() == 'nan':
        return False
    return True


def is_processed_game_done(summary_value: Any, cover_path_value: Any) -> bool:
    """Return ``True`` when a processed row has the required summary and cover."""

    return has_summary_text(summary_value) and has_cover_path_value(cover_path_value)

app = Flask(__name__)
app.secret_key = os.environ.get('APP_SECRET_KEY', 'dev-secret')
APP_PASSWORD = os.environ.get('APP_PASSWORD', 'password')
DEFAULT_IGDB_USER_AGENT = 'TT-Game-Liste/1.0 (support@example.com)'
IGDB_USER_AGENT = os.environ.get('IGDB_USER_AGENT') or DEFAULT_IGDB_USER_AGENT
IGDB_BATCH_SIZE = 500

IGDB_CATEGORY_LABELS = {
    0: 'Main Game',
    1: 'DLC / Add-on',
    2: 'Expansion',
    3: 'Bundle',
    4: 'Standalone Expansion',
    5: 'Mod',
    6: 'Episode',
    7: 'Season',
    8: 'Remake',
    9: 'Remaster',
    10: 'Expanded Game',
    11: 'Port',
    12: 'Fork',
    13: 'Pack',
    14: 'Update',
}


IGDB_CACHE_TABLE = 'igdb_games'
IGDB_CACHE_STATE_TABLE = 'igdb_cache_state'


def _igdb_page_size() -> int:
    try:
        size = int(IGDB_BATCH_SIZE)
    except (TypeError, ValueError):
        return 500
    if size <= 0:
        return 500
    return min(size, 500)


def _dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def _format_name_list(value: Any) -> str:
    return ', '.join(_dedupe_preserve_order(_parse_iterable(value)))


def _collect_company_names(
    companies: Any, role_key: str
) -> list[str]:  # pragma: no cover - simple data formatter
    names: list[str] = []
    if isinstance(companies, list):
        for company in companies:
            if not isinstance(company, Mapping):
                continue
            if not company.get(role_key):
                continue
            company_obj = company.get('company')
            name_value: Any = None
            if isinstance(company_obj, Mapping):
                name_value = company_obj.get('name')
            elif isinstance(company_obj, str):
                name_value = company_obj
            if not name_value:
                continue
            text = str(name_value).strip()
            if text:
                names.append(text)
    return _dedupe_preserve_order(names)


def _format_first_release_date(value: Any) -> str:
    if value in (None, '', 0):
        return ''
    try:
        timestamp = float(value)
    except (TypeError, ValueError):
        try:
            timestamp = float(str(value).strip())
        except (TypeError, ValueError):
            return ''
    if timestamp <= 0:
        return ''
    try:
        dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return ''
    return dt.date().isoformat()


def _cover_url_from_cover(value: Any, size: str = 't_cover_big') -> str:
    image_id: str | None = None
    if isinstance(value, Mapping):
        raw_id = value.get('image_id')
        if isinstance(raw_id, str):
            image_id = raw_id.strip()
        elif raw_id is not None:
            image_id = str(raw_id).strip()
    elif isinstance(value, str):
        image_id = value.strip()
    elif value is not None:
        image_id = str(value).strip()
    if not image_id:
        return ''
    size_key = str(size).strip() if size else 't_cover_big'
    if not size_key:
        size_key = 't_cover_big'
    return (
        'https://images.igdb.com/igdb/image/upload/'
        f'{size_key}/{image_id}.jpg'
    )


def _igdb_category_display(value: Any) -> str:
    if value in (None, ''):
        return ''
    try:
        key = int(value)
    except (TypeError, ValueError):
        return str(value).strip()
    return IGDB_CATEGORY_LABELS.get(key, 'Other')


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


def _normalize_translation_key(value: str) -> str:
    key = str(value).strip().casefold()
    for old, new in (
        ('&', ' and '),
        ('/', ' '),
        ('-', ' '),
        ('_', ' '),
        ("'", ''),
        (',', ' '),
        ('.', ' '),
        ('+', ' '),
    ):
        key = key.replace(old, new)
    for char in '()[]{}':
        key = key.replace(char, ' ')
    key = ''.join(ch for ch in key if ch.isalnum() or ch.isspace())
    return ' '.join(key.split())


IGDB_GENRE_TRANSLATIONS: dict[str, tuple[str, ...]] = {
    _normalize_translation_key('Action'): ('Ação e Aventura',),
    _normalize_translation_key('Action Adventure'): ('Ação e Aventura',),
    _normalize_translation_key('Adventure'): ('Ação e Aventura',),
    _normalize_translation_key('Point-and-click'): ('Ação e Aventura',),
    _normalize_translation_key('Stealth'): ('Ação e Aventura',),
    _normalize_translation_key('Survival'): ('Ação e Aventura',),
    _normalize_translation_key('Platform'): ('Plataformas',),
    _normalize_translation_key('Platformer'): ('Plataformas',),
    _normalize_translation_key('Shooter'): ('Tiro',),
    _normalize_translation_key("Shoot 'em up"): ('Tiro',),
    _normalize_translation_key('Fighting'): ('Luta',),
    _normalize_translation_key("Hack and slash/Beat 'em up"): ('Luta',),
    _normalize_translation_key('Brawler'): ('Luta',),
    _normalize_translation_key('Racing'): ('Corrida e Voo',),
    _normalize_translation_key('Driving Racing'): ('Corrida e Voo',),
    _normalize_translation_key('Flight'): ('Corrida e Voo',),
    _normalize_translation_key('Simulator'): ('Simulação',),
    _normalize_translation_key('Simulation'): ('Simulação',),
    _normalize_translation_key('Strategy'): ('Estratégia',),
    _normalize_translation_key('Real Time Strategy (RTS)'): ('Estratégia',),
    _normalize_translation_key('Turn-based strategy (TBS)'): ('Estratégia',),
    _normalize_translation_key('Tactical'): ('Estratégia',),
    _normalize_translation_key('MOBA'): ('Multijogador',),
    _normalize_translation_key('Massively Multiplayer Online (MMO)'): ('Multijogador',),
    _normalize_translation_key('Battle Royale'): ('Multijogador',),
    _normalize_translation_key('MMORPG'): ('RPG', 'Multijogador'),
    _normalize_translation_key('Role-playing (RPG)'): ('RPG',),
    _normalize_translation_key('Role playing'): ('RPG',),
    _normalize_translation_key('Roguelike'): ('RPG',),
    _normalize_translation_key('Roguelite'): ('RPG',),
    _normalize_translation_key('Puzzle'): ('Quebra-cabeça e Trivia',),
    _normalize_translation_key('Quiz/Trivia'): ('Quebra-cabeça e Trivia',),
    _normalize_translation_key('Trivia'): ('Quebra-cabeça e Trivia',),
    _normalize_translation_key('Card & Board Game'): ('Cartas e Tabuleiro',),
    _normalize_translation_key('Board game'): ('Cartas e Tabuleiro',),
    _normalize_translation_key('Tabletop'): ('Cartas e Tabuleiro',),
    _normalize_translation_key('Family'): ('Família e Crianças',),
    _normalize_translation_key('Kids'): ('Família e Crianças',),
    _normalize_translation_key('Educational'): ('Família e Crianças',),
    _normalize_translation_key('Party'): ('Família e Crianças',),
    _normalize_translation_key('Music'): ('Família e Crianças',),
    _normalize_translation_key('Indie'): ('Indie',),
    _normalize_translation_key('Arcade'): ('Clássicos',),
    _normalize_translation_key('Pinball'): ('Clássicos',),
    _normalize_translation_key('Classic'): ('Clássicos',),
    _normalize_translation_key('Visual Novel'): ('Visual Novel',),
    _normalize_translation_key('ação e aventura'): ('Ação e Aventura',),
    _normalize_translation_key('plataformas'): ('Plataformas',),
    _normalize_translation_key('tiro'): ('Tiro',),
    _normalize_translation_key('luta'): ('Luta',),
    _normalize_translation_key('corrida e voo'): ('Corrida e Voo',),
    _normalize_translation_key('simulação'): ('Simulação',),
    _normalize_translation_key('estratégia'): ('Estratégia',),
    _normalize_translation_key('multijogador'): ('Multijogador',),
    _normalize_translation_key('rpg'): ('RPG',),
    _normalize_translation_key('quebra-cabeça e trivia'): ('Quebra-cabeça e Trivia',),
    _normalize_translation_key('cartas e tabuleiro'): ('Cartas e Tabuleiro',),
    _normalize_translation_key('família e crianças'): ('Família e Crianças',),
    _normalize_translation_key('indie'): ('Indie',),
    _normalize_translation_key('clássicos'): ('Clássicos',),
    _normalize_translation_key('visual novel'): ('Visual Novel',),
}


IGDB_MODE_TRANSLATIONS: dict[str, tuple[str, ...]] = {
    _normalize_translation_key('Single player'): ('Single-player',),
    _normalize_translation_key('Single-player'): ('Single-player',),
    _normalize_translation_key('Singleplayer'): ('Single-player',),
    _normalize_translation_key('Solo'): ('Single-player',),
    _normalize_translation_key('Campaign'): ('Single-player',),
    _normalize_translation_key('Co-operative'): ('Cooperativo (Co-op)',),
    _normalize_translation_key('Cooperative'): ('Cooperativo (Co-op)',),
    _normalize_translation_key('Co-op'): ('Cooperativo (Co-op)',),
    _normalize_translation_key('Co op'): ('Cooperativo (Co-op)',),
    _normalize_translation_key('Local co-op'): ('Cooperativo (Co-op)', 'Multiplayer local'),
    _normalize_translation_key('Offline co-op'): ('Cooperativo (Co-op)', 'Multiplayer local'),
    _normalize_translation_key('Online co-op'): ('Cooperativo (Co-op)', 'Multiplayer online'),
    _normalize_translation_key('Co-op campaign'): ('Cooperativo (Co-op)',),
    _normalize_translation_key('Multiplayer'): ('Multiplayer online',),
    _normalize_translation_key('Online multiplayer'): ('Multiplayer online',),
    _normalize_translation_key('Multiplayer online'): ('Multiplayer online',),
    _normalize_translation_key('Offline multiplayer'): ('Multiplayer local',),
    _normalize_translation_key('Local multiplayer'): ('Multiplayer local',),
    _normalize_translation_key('Split screen'): ('Multiplayer local',),
    _normalize_translation_key('Shared/Split screen'): ('Multiplayer local',),
    _normalize_translation_key('PvP'): ('Competitivo (PvP)',),
    _normalize_translation_key('Player vs Player'): ('Competitivo (PvP)',),
    _normalize_translation_key('Versus'): ('Competitivo (PvP)',),
    _normalize_translation_key('Competitive'): ('Competitivo (PvP)',),
    _normalize_translation_key('Battle Royale'): ('Competitivo (PvP)', 'Multiplayer online'),
    _normalize_translation_key('Massively Multiplayer Online (MMO)'): ('Multiplayer online',),
    _normalize_translation_key('MMO'): ('Multiplayer online',),
    _normalize_translation_key('MMORPG'): ('Cooperativo (Co-op)', 'Multiplayer online'),
    _normalize_translation_key('single-player'): ('Single-player',),
    _normalize_translation_key('cooperativo (co-op)'): ('Cooperativo (Co-op)',),
    _normalize_translation_key('multiplayer local'): ('Multiplayer local',),
    _normalize_translation_key('multiplayer online'): ('Multiplayer online',),
    _normalize_translation_key('competitivo (pvp)'): ('Competitivo (PvP)',),
}
logging.basicConfig(level=logging.DEBUG)
app.logger.setLevel(logging.DEBUG)
logger.setLevel(logging.DEBUG)
# Configure OpenAI using API key from environment
client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY', ''))

# SQLite setup for processed games
db_lock = db_utils.db_lock

_source_index_cache_lock = Lock()
_source_index_by_position: dict[int, str] | None = None
_position_by_source_index: dict[str, int] | None = None
_source_index_cache_df_id: int | None = None


def _job_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


JOB_STATUS_PENDING = 'pending'
JOB_STATUS_RUNNING = 'running'
JOB_STATUS_SUCCESS = 'success'
JOB_STATUS_ERROR = 'error'
JOB_ACTIVE_STATUSES = {JOB_STATUS_PENDING, JOB_STATUS_RUNNING}
JOB_TERMINAL_STATUSES = {JOB_STATUS_SUCCESS, JOB_STATUS_ERROR}


@dataclass
class BackgroundJob:
    id: str
    job_type: str
    status: str = JOB_STATUS_PENDING
    message: str = ''
    progress_current: int = 0
    progress_total: int = 0
    data: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    created_at: str = ''
    updated_at: str = ''
    started_at: Optional[str] = None
    finished_at: Optional[str] = None


class BackgroundJobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, BackgroundJob] = {}
        self._lock = Lock()

    def _serialize_job(self, job: BackgroundJob) -> dict[str, Any]:
        return {
            'id': job.id,
            'job_type': job.job_type,
            'status': job.status,
            'message': job.message,
            'progress_current': job.progress_current,
            'progress_total': job.progress_total,
            'data': dict(job.data),
            'result': dict(job.result),
            'error': job.error,
            'created_at': job.created_at,
            'updated_at': job.updated_at,
            'started_at': job.started_at,
            'finished_at': job.finished_at,
        }

    def _find_active_job_locked(self, job_type: str) -> Optional[BackgroundJob]:
        for job in self._jobs.values():
            if job.job_type == job_type and job.status in JOB_ACTIVE_STATUSES:
                return job
        return None

    def _prune_jobs_locked(self) -> None:
        if len(self._jobs) <= MAX_BACKGROUND_JOBS:
            return
        removable: list[BackgroundJob] = [
            job
            for job in self._jobs.values()
            if job.status in JOB_TERMINAL_STATUSES
        ]
        removable.sort(key=lambda j: j.finished_at or j.updated_at)
        while len(self._jobs) > MAX_BACKGROUND_JOBS and removable:
            victim = removable.pop(0)
            self._jobs.pop(victim.id, None)

    def list_jobs(self, job_type: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            jobs = list(self._jobs.values())
            if job_type:
                jobs = [job for job in jobs if job.job_type == job_type]
            jobs.sort(key=lambda job: job.created_at)
            return [self._serialize_job(job) for job in jobs]

    def get_job(self, job_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            return self._serialize_job(job)

    def get_active_job(self, job_type: str) -> Optional[dict[str, Any]]:
        with self._lock:
            job = self._find_active_job_locked(job_type)
            if job is None:
                return None
            return self._serialize_job(job)

    def start_job(
        self,
        job_type: str,
        runner: Callable[[Callable[..., None]], Optional[dict[str, Any]]],
        *,
        description: str | None = None,
    ) -> tuple[dict[str, Any], bool]:
        with self._lock:
            existing = self._find_active_job_locked(job_type)
            if existing is not None:
                return self._serialize_job(existing), False
            job_id = uuid.uuid4().hex
            timestamp = _job_timestamp()
            job = BackgroundJob(
                id=job_id,
                job_type=job_type,
                message=description or '',
                created_at=timestamp,
                updated_at=timestamp,
            )
            self._jobs[job_id] = job
            self._prune_jobs_locked()

        thread = Thread(
            target=self._run_job,
            args=(job_id, runner),
            name=f'job-{job_type}-{job_id}',
            daemon=True,
        )
        thread.start()
        return self.get_job(job_id), True

    def _set_job_running(self, job_id: str) -> None:
        timestamp = _job_timestamp()
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = JOB_STATUS_RUNNING
            job.started_at = timestamp
            job.updated_at = timestamp
            if not job.message:
                job.message = 'Running…'

    def _update_job(
        self,
        job_id: str,
        *,
        progress_current: int | None = None,
        progress_total: int | None = None,
        message: str | None = None,
        data: Optional[Mapping[str, Any]] = None,
    ) -> None:
        timestamp = _job_timestamp()
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            if progress_current is not None:
                job.progress_current = max(int(progress_current), 0)
            if progress_total is not None:
                job.progress_total = max(int(progress_total), 0)
            if message is not None:
                job.message = str(message)
            if data:
                for key, value in data.items():
                    job.data[key] = value
            job.updated_at = timestamp

    def _finalize_job(
        self,
        job_id: str,
        status: str,
        result: Optional[dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        timestamp = _job_timestamp()
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = status
            job.error = error
            job.result = dict(result or {})
            job.finished_at = timestamp
            job.updated_at = timestamp

    def _run_job(
        self,
        job_id: str,
        runner: Callable[[Callable[..., None]], Optional[dict[str, Any]]],
    ) -> None:
        def progress_callback(
            current: int | None = None,
            total: int | None = None,
            message: str | None = None,
            *,
            data: Optional[Mapping[str, Any]] = None,
            **extra: Any,
        ) -> None:
            merged: dict[str, Any] = {}
            if data:
                merged.update(dict(data))
            if extra:
                merged.update({k: v for k, v in extra.items() if v is not None})
            self._update_job(
                job_id,
                progress_current=current,
                progress_total=total,
                message=message,
                data=merged or None,
            )

        self._set_job_running(job_id)
        try:
            result = runner(progress_callback)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception('Background job %s failed', job_id)
            self._finalize_job(job_id, JOB_STATUS_ERROR, error=str(exc))
            return
        self._finalize_job(job_id, JOB_STATUS_SUCCESS, result=result)


job_manager = BackgroundJobManager()


def _canonical_source_index(value: Any) -> str | None:
    """Normalize ``Source Index`` values to a consistent string representation."""

    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return str(int(text))
    except (TypeError, ValueError):
        return text


def reset_source_index_cache() -> None:
    """Clear cached mappings between navigator positions and ``Source Index`` values."""

    global _source_index_by_position, _position_by_source_index, _source_index_cache_df_id
    with _source_index_cache_lock:
        _source_index_by_position = None
        _position_by_source_index = None
        _source_index_cache_df_id = None


def _ensure_source_index_cache() -> tuple[dict[int, str], dict[str, int]]:
    """Build and return cached lookup tables for ``Source Index`` values."""

    global _source_index_by_position, _position_by_source_index, _source_index_cache_df_id

    df = globals().get('games_df')
    if df is None:
        raise RuntimeError('games_df is not loaded')

    with _source_index_cache_lock:
        df_id = id(df)
        if (
            _source_index_by_position is None
            or _position_by_source_index is None
            or _source_index_cache_df_id != df_id
            or len(_source_index_by_position) != len(df)
        ):
            mapping: dict[int, str] = {}
            reverse: dict[str, int] = {}
            if len(df) > 0:
                source_values: list[Any] | None = None
                if 'Source Index' in df.columns:
                    source_values = df['Source Index'].tolist()
                for position in range(len(df)):
                    raw_value: Any
                    if source_values is not None and position < len(source_values):
                        raw_value = source_values[position]
                    else:
                        raw_value = position
                    canonical = _canonical_source_index(raw_value)
                    if canonical is None:
                        canonical = str(position)
                    mapping[position] = canonical
                    reverse.setdefault(canonical, position)
            _source_index_by_position = mapping
            _position_by_source_index = reverse
            _source_index_cache_df_id = df_id

        return _source_index_by_position, _position_by_source_index


def get_source_index_for_position(position: int) -> str:
    """Return the normalized ``Source Index`` string for a DataFrame position."""

    if position < 0:
        raise IndexError('invalid index')
    mapping, _ = _ensure_source_index_cache()
    try:
        return mapping[position]
    except KeyError as exc:
        raise IndexError('invalid index') from exc


def get_position_for_source_index(value: Any) -> int | None:
    """Resolve a ``Source Index`` value back to its DataFrame position."""

    canonical = _canonical_source_index(value)
    if canonical is None:
        return None
    mapping, reverse = _ensure_source_index_cache()
    position = reverse.get(canonical)
    if position is not None:
        return position
    try:
        fallback = int(canonical)
    except (TypeError, ValueError):
        return None
    if fallback < 0:
        return None
    return fallback


def get_db() -> sqlite3.Connection:
    return db_utils.get_db(_db_connection_factory)


def get_processed_games_columns(conn: sqlite3.Connection | None = None) -> set[str]:
    return db_utils.get_processed_games_columns(
        conn, connection_factory=_db_connection_factory
    )


def _quote_identifier(identifier: str) -> str:
    return db_utils._quote_identifier(identifier)


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


def _map_igdb_values(
    names: Iterable[str], translations: Mapping[str, tuple[str, ...]]
) -> list[str]:
    results: list[str] = []
    seen: set[str] = set()
    for raw_name in names:
        normalized = _normalize_lookup_name(raw_name)
        if not normalized:
            continue
        key = _normalize_translation_key(normalized)
        mapped = translations.get(key)
        if not mapped:
            mapped = (normalized,)
        for candidate in mapped:
            final = _normalize_lookup_name(candidate)
            if not final:
                continue
            dedupe_key = final.casefold()
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            results.append(final)
    return results


def _map_igdb_genres(names: Iterable[str]) -> list[str]:
    return _map_igdb_values(names, IGDB_GENRE_TRANSLATIONS)


def _map_igdb_modes(names: Iterable[str]) -> list[str]:
    return _map_igdb_values(names, IGDB_MODE_TRANSLATIONS)


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
        id_column = relation.get('id_column')
        stored_ids: list[int] = []
        if id_column and processed_mapping:
            stored_ids = _decode_lookup_id_list(processed_mapping.get(id_column))
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

        join_id_to_name: dict[int, str] = {}
        join_order: list[int] = []
        for join_row in join_rows:
            raw_id = _row_value(join_row, 'lookup_id', 0)
            try:
                lookup_id = int(raw_id)
            except (TypeError, ValueError):
                continue
            if lookup_id in join_id_to_name:
                continue
            lookup_name = _normalize_lookup_name(_row_value(join_row, 'name', 1))
            if not lookup_name:
                lookup_name = _lookup_name_for_id(conn, lookup_table, lookup_id)
            join_id_to_name[lookup_id] = _normalize_lookup_name(lookup_name)
            join_order.append(lookup_id)

        id_sequence = stored_ids if stored_ids else join_order
        seen_ids: set[int] = set()
        for lookup_id in id_sequence:
            if lookup_id in seen_ids:
                continue
            seen_ids.add(lookup_id)
            lookup_name = join_id_to_name.get(lookup_id)
            if not lookup_name:
                lookup_name = _lookup_name_for_id(conn, lookup_table, lookup_id)
            entries.append({'id': lookup_id, 'name': _normalize_lookup_name(lookup_name)})
        for lookup_id in join_order:
            if lookup_id in seen_ids:
                continue
            seen_ids.add(lookup_id)
            lookup_name = join_id_to_name.get(lookup_id)
            if not lookup_name:
                lookup_name = _lookup_name_for_id(conn, lookup_table, lookup_id)
            entries.append({'id': lookup_id, 'name': _normalize_lookup_name(lookup_name)})

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


def _backfill_lookup_id_columns(conn: sqlite3.Connection) -> None:
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

        updates: dict[str, str] = {}
        for relation in LOOKUP_RELATIONS:
            id_column = relation.get('id_column')
            if not id_column:
                continue
            join_table = relation['join_table']
            join_column = relation['join_column']
            stored_value = row_dict.get(id_column)
            existing_serialized = stored_value if isinstance(stored_value, str) else ''

            join_ids: list[int] = []
            try:
                cur_join = conn.execute(
                    f'SELECT {join_column} FROM {join_table} '
                    'WHERE processed_game_id = ? ORDER BY rowid',
                    (game_id,),
                )
                join_rows = cur_join.fetchall()
            except sqlite3.OperationalError:
                join_rows = []

            seen: set[int] = set()
            for join_row in join_rows:
                raw_value = _row_value(join_row, join_column, 0)
                try:
                    coerced = int(raw_value)
                except (TypeError, ValueError):
                    continue
                if coerced in seen:
                    continue
                seen.add(coerced)
                join_ids.append(coerced)

            serialized = _encode_lookup_id_list(join_ids)
            if serialized != (existing_serialized or ''):
                updates[id_column] = serialized

        if updates:
            assignments = ', '.join(
                f'{_quote_identifier(column)}=?' for column in updates
            )
            params = list(updates.values()) + [game_id]
            conn.execute(
                f'UPDATE processed_games SET {assignments} WHERE "ID"=?',
                params,
            )


def _ensure_lookup_id_columns(conn: sqlite3.Connection) -> None:
    try:
        cur = conn.execute('PRAGMA table_info(processed_games)')
    except sqlite3.OperationalError:
        return

    rows = cur.fetchall()
    existing_columns = {row[1] for row in rows}
    added = False
    for relation in LOOKUP_RELATIONS:
        id_column = relation.get('id_column')
        if not id_column or id_column in existing_columns:
            continue
        try:
            conn.execute(
                f'ALTER TABLE processed_games ADD COLUMN {_quote_identifier(id_column)} TEXT'
            )
            added = True
        except sqlite3.OperationalError:
            continue
    if added:
        try:
            cur = conn.execute('PRAGMA table_info(processed_games)')
        except sqlite3.OperationalError:
            return
        rows = cur.fetchall()
        existing_columns = {row[1] for row in rows}

    expected_columns = {
        relation['id_column']
        for relation in LOOKUP_RELATIONS
        if relation.get('id_column')
    }
    if expected_columns & existing_columns:
        _backfill_lookup_id_columns(conn)


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


def _lookup_entries_to_selection(
    entries: Mapping[str, list[dict[str, Any]]]
) -> dict[str, dict[str, list[int]]]:
    payload: dict[str, dict[str, list[int]]] = {}
    for relation in LOOKUP_RELATIONS:
        response_key = relation['response_key']
        relation_entries = entries.get(response_key, [])
        ids: list[int] = []
        for entry in relation_entries:
            if not isinstance(entry, Mapping):
                continue
            entry_id = entry.get('id')
            if entry_id is None:
                continue
            try:
                ids.append(int(entry_id))
            except (TypeError, ValueError):
                continue
        payload[response_key] = {'ids': ids}
    return payload


def _apply_lookup_entries_to_processed_game(
    conn: sqlite3.Connection,
    processed_game_id: int,
    entries: Mapping[str, list[dict[str, Any]]],
) -> None:
    columns = get_processed_games_columns(conn)
    set_fragments: list[str] = []
    params: list[Any] = []
    for relation in LOOKUP_RELATIONS:
        response_key = relation['response_key']
        processed_column = relation['processed_column']
        id_column = relation.get('id_column')
        relation_entries = entries.get(response_key, [])
        if processed_column in columns:
            names: list[str] = []
            for entry in relation_entries:
                if not isinstance(entry, Mapping):
                    continue
                normalized = _normalize_lookup_name(entry.get('name'))
                if normalized:
                    names.append(normalized)
            set_fragments.append(f'{_quote_identifier(processed_column)} = ?')
            params.append(_lookup_display_text(names))
        if id_column and id_column in columns:
            ids: list[int] = []
            for entry in relation_entries:
                if not isinstance(entry, Mapping):
                    continue
                entry_id = entry.get('id')
                if entry_id is None:
                    continue
                try:
                    ids.append(int(entry_id))
                except (TypeError, ValueError):
                    continue
            set_fragments.append(f'{_quote_identifier(id_column)} = ?')
            params.append(_encode_lookup_id_list(ids))
    if not set_fragments:
        return
    params.append(processed_game_id)
    conn.execute(
        f'UPDATE processed_games SET {", ".join(set_fragments)} WHERE "ID" = ?',
        params,
    )


def _remove_lookup_id_from_entries(
    entries: Mapping[str, list[dict[str, Any]]],
    relation: Mapping[str, Any],
    lookup_id: int,
) -> None:
    response_key = relation['response_key']
    relation_entries = list(entries.get(response_key, []) or [])
    filtered: list[dict[str, Any]] = []
    for entry in relation_entries:
        if not isinstance(entry, Mapping):
            continue
        entry_id = entry.get('id')
        try:
            coerced = int(entry_id) if entry_id is not None else None
        except (TypeError, ValueError):
            coerced = None
        if coerced == lookup_id:
            continue
        filtered.append(entry)
    entries[response_key] = filtered


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
    finally:
        conn.close()


_init_db(run_migrations=RUN_DB_MIGRATIONS)

# Expose a module-level connection for tests and utilities
db = _db_connection_factory()
db_utils.set_fallback_connection(db)

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
                    'Large Cover Image (URL)': _cover_url_from_cover(
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

        page_size = _igdb_page_size()
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
            cover_url = _cover_url_from_cover(item.get('cover'))
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
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
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

    if 'games_df' not in globals():
        return
    if games_df.empty:
        return

    def _normalize_name(value: Any) -> str:
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
        if text.lower() == 'nan':
            return ''
        return text

    def _coerce_source_index(value: Any) -> str | None:
        return _canonical_source_index(value)

    with db_lock:
        conn = get_db()
        with conn:
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
                    _normalize_name(row['Name']),
                    is_processed_game_done(summary_value, cover_value),
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
                    name_value = games_df.iloc[position].get('Name') if position < row_count else None
                igdb_name = _normalize_name(name_value)
                stored_name: str | None = igdb_name if igdb_name else None

                if src_index not in existing:
                    conn.execute(
                        'INSERT OR IGNORE INTO processed_games ("Source Index", "Name") VALUES (?, ?)',
                        (src_index, stored_name),
                    )
                    existing[src_index] = (src_index, igdb_name, False)
                    continue

                stored_source, existing_name, is_done = existing[src_index]
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
    if 'games_df' not in globals():
        return
    if games_df.empty:
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
    if not isinstance(item, Mapping):
        return None

    raw_id = item.get('id')
    try:
        igdb_id = int(str(raw_id).strip())
    except (TypeError, ValueError):
        logger.warning('Skipping IGDB entry with invalid id %s', raw_id)
        return None

    name_value = item.get('name')
    name = name_value.strip() if isinstance(name_value, str) else ''

    summary_value = item.get('summary')
    summary = summary_value.strip() if isinstance(summary_value, str) else ''

    cover_obj = item.get('cover')
    cover: dict[str, Any] | None = None
    if isinstance(cover_obj, Mapping):
        image_id_value = cover_obj.get('image_id') or cover_obj.get('imageId')
        if image_id_value:
            cover = {'image_id': str(image_id_value)}
    elif isinstance(cover_obj, str):
        image_id = cover_obj.strip()
        if image_id:
            cover = {'image_id': image_id}

    rating_count = _coerce_rating_count(
        item.get('total_rating_count'), item.get('rating_count')
    )

    involved_companies = item.get('involved_companies')
    developer_names: list[str] = []
    publisher_names: list[str] = []
    if isinstance(involved_companies, list):
        seen_dev: set[str] = set()
        seen_pub: set[str] = set()
        for company in involved_companies:
            if not isinstance(company, Mapping):
                continue
            company_obj = company.get('company')
            company_name: str | None = None
            if isinstance(company_obj, Mapping):
                name_candidate = company_obj.get('name')
                if isinstance(name_candidate, str):
                    company_name = name_candidate.strip()
            elif isinstance(company_obj, str):
                company_name = company_obj.strip()
            if not company_name:
                name_candidate = company.get('name')
                if isinstance(name_candidate, str):
                    company_name = name_candidate.strip()
            if not company_name:
                continue
            fingerprint = company_name.casefold()
            if company.get('developer') and fingerprint not in seen_dev:
                seen_dev.add(fingerprint)
                developer_names.append(company_name)
            if company.get('publisher') and fingerprint not in seen_pub:
                seen_pub.add(fingerprint)
                publisher_names.append(company_name)

    if not developer_names:
        developer_names = _parse_company_names(item.get('developers'))
    if not publisher_names:
        publisher_names = _parse_company_names(item.get('publishers'))

    genres = _parse_iterable(item.get('genres'))
    platforms = _parse_iterable(item.get('platforms'))
    game_modes = _parse_iterable(item.get('game_modes'))

    return {
        'id': igdb_id,
        'name': name,
        'summary': summary,
        'updated_at': item.get('updated_at'),
        'first_release_date': item.get('first_release_date'),
        'category': item.get('category'),
        'cover': cover,
        'rating_count': rating_count,
        'developers': developer_names,
        'publishers': publisher_names,
        'genres': genres,
        'platforms': platforms,
        'game_modes': game_modes,
    }


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


def exchange_twitch_credentials() -> tuple[str, str]:
    """Compatibility wrapper that proxies to :mod:`igdb.client`."""

    return igdb_client.exchange_twitch_credentials()


def download_igdb_metadata(
    access_token: str, client_id: str, igdb_ids: Iterable[str]
) -> dict[str, dict[str, Any]]:
    """Fetch IGDB metadata using the shared client helpers."""

    batch_size = (
        IGDB_BATCH_SIZE
        if isinstance(IGDB_BATCH_SIZE, int) and IGDB_BATCH_SIZE > 0
        else 500
    )
    return igdb_client.download_igdb_metadata(
        access_token,
        client_id,
        igdb_ids,
        batch_size=batch_size,
        user_agent=IGDB_USER_AGENT,
        normalize=_normalize_igdb_payload,
        coerce_id=coerce_igdb_id,
        request_factory=Request,
        opener=urlopen,
    )


def download_igdb_game_count(access_token: str, client_id: str) -> int:
    """Return the IGDB game count via the shared client module."""

    return igdb_client.download_igdb_game_count(
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

    return igdb_client.download_igdb_games(
        access_token,
        client_id,
        offset,
        limit,
        user_agent=IGDB_USER_AGENT,
        normalize=_normalize_igdb_payload,
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
    inserted = updated = unchanged = 0
    for payload in payloads:
        row = _build_cache_row_from_payload(payload)
        if row is None:
            continue
        igdb_id = row['igdb_id']
        existing = conn.execute(
            f'''SELECT name, summary, updated_at, first_release_date, category,
                       cover_image_id, rating_count, developers, publishers, genres,
                       platforms, game_modes
                FROM {IGDB_CACHE_TABLE}
                WHERE igdb_id = ?''',
            (igdb_id,),
        ).fetchone()
        cached_at = now_utc_iso()
        params = (
            row['name'],
            row['summary'],
            row['updated_at'],
            row['first_release_date'],
            row['category'],
            row['cover_image_id'],
            row['rating_count'],
            row['developers_json'],
            row['publishers_json'],
            row['genres_json'],
            row['platforms_json'],
            row['game_modes_json'],
        )
        if existing is None:
            conn.execute(
                f'''INSERT INTO {IGDB_CACHE_TABLE} (
                        igdb_id, name, summary, updated_at, first_release_date,
                        category, cover_image_id, rating_count, developers,
                        publishers, genres, platforms, game_modes, cached_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (
                    igdb_id,
                    *params,
                    cached_at,
                ),
            )
            inserted += 1
            continue

        existing_values = (
            existing['name'],
            existing['summary'],
            existing['updated_at'],
            existing['first_release_date'],
            existing['category'],
            existing['cover_image_id'],
            existing['rating_count'],
            existing['developers'],
            existing['publishers'],
            existing['genres'],
            existing['platforms'],
            existing['game_modes'],
        )
        if existing_values == params:
            unchanged += 1
            continue

        conn.execute(
            f'''UPDATE {IGDB_CACHE_TABLE}
                   SET name=?, summary=?, updated_at=?, first_release_date=?,
                       category=?, cover_image_id=?, rating_count=?,
                       developers=?, publishers=?, genres=?, platforms=?,
                       game_modes=?, cached_at=?
                 WHERE igdb_id=?''',
            (*params, cached_at, igdb_id),
        )
        updated += 1
    return inserted, updated, unchanged


def _get_cached_igdb_total(conn: sqlite3.Connection) -> int | None:
    row = conn.execute(
        f'SELECT total_count FROM {IGDB_CACHE_STATE_TABLE} WHERE id = 1'
    ).fetchone()
    if row is None:
        return None
    try:
        return int(row['total_count']) if row['total_count'] is not None else None
    except (TypeError, ValueError):
        return None


def _set_cached_igdb_total(
    conn: sqlite3.Connection, total: int | None, synced_at: str | None = None
) -> None:
    timestamp = synced_at or now_utc_iso()
    conn.execute(
        f'''INSERT INTO {IGDB_CACHE_STATE_TABLE} (id, total_count, last_synced_at)
            VALUES (1, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                total_count=excluded.total_count,
                last_synced_at=excluded.last_synced_at''',
        (total, timestamp),
    )




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
        _map_igdb_genres(_parse_iterable(metadata.get('genres')))
    )
    if genre_names:
        overlay['Genres'] = ', '.join(genre_names)

    mode_names = _dedupe_normalized_names(
        _map_igdb_modes(_parse_iterable(metadata.get('game_modes')))
    )
    if mode_names:
        overlay['Game Modes'] = ', '.join(mode_names)

    platform_names = _dedupe_normalized_names(_parse_iterable(metadata.get('platforms')))
    if platform_names:
        overlay['Platforms'] = ', '.join(platform_names)

    cover_url = _cover_url_from_cover(metadata.get('cover'), size='t_original')
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
                remote_items = _map_igdb_genres(remote_items)
            elif local_field == 'Game Modes':
                remote_items = _map_igdb_modes(remote_items)
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


class GameNavigator:
    """Thread-safe helper to navigate game list and track progress."""

    def __init__(self, total_rows: int):
        self.lock = Lock()
        self.total = total_rows
        self.current_index = 0
        self.seq_index = 1
        self.processed_total = 0
        self.skip_queue: list[dict[str, int]] = []
        self._load_initial()

    def _load_initial(self) -> None:
        with db_lock:
            conn = get_db()
            cur = conn.execute('SELECT current_index, seq_index, skip_queue FROM navigator_state WHERE id=1')
            state_row = cur.fetchone()
            cur = conn.execute(
                'SELECT "Source Index", "ID", "Summary", "Cover Path" FROM processed_games'
            )
            rows = cur.fetchall()
        processed: set[int] = set()
        max_seq = 0
        for row in rows:
            position = get_position_for_source_index(row['Source Index'])
            if position is not None and 0 <= position < self.total:
                try:
                    summary_value = row['Summary']
                except (KeyError, IndexError, TypeError):
                    summary_value = None
                try:
                    cover_value = row['Cover Path']
                except (KeyError, IndexError, TypeError):
                    cover_value = None
                if is_processed_game_done(summary_value, cover_value):
                    processed.add(position)
            try:
                row_id = int(row['ID'])
            except (TypeError, ValueError):
                row_id = None
            if row_id is not None and row_id > max_seq:
                max_seq = row_id
        next_index = next((i for i in range(self.total) if i not in processed), self.total)
        expected_seq = max_seq + 1 if max_seq > 0 else 1
        self.processed_total = len(processed)
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
            cur = conn.execute(
                'SELECT "Summary", "Cover Path" FROM processed_games'
            )
            rows = cur.fetchall()
        processed_total = 0
        for row in rows:
            try:
                summary_value = row['Summary']
            except (KeyError, IndexError, TypeError):
                summary_value = None
            try:
                cover_value = row['Cover Path']
            except (KeyError, IndexError, TypeError):
                cover_value = None
            if is_processed_game_done(summary_value, cover_value):
                processed_total += 1
        self.processed_total = max(processed_total, 0)
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


def _set_games_dataframe(
    df: pd.DataFrame | None,
    *,
    rebuild_metadata: bool = True,
    rebuild_navigator: bool = True,
) -> None:
    global games_df, total_games, categories_list, platforms_list, _category_values, navigator

    if df is None:
        df = pd.DataFrame()

    games_df = df
    total_games = len(df)

    if rebuild_metadata:
        category_values: set[str] = set()
        if not df.empty and 'Category' in df.columns:
            for raw_category in df['Category'].dropna():
                text = str(raw_category).strip()
                if text:
                    category_values.add(text)
        category_values.update(
            label for label in IGDB_CATEGORY_LABELS.values() if label
        )
        _category_values = category_values
        categories_list = sorted(category_values, key=str.casefold)

        platform_values: set[str] = set()
        if not df.empty and 'Platforms' in df.columns:
            for raw_platforms in df['Platforms'].dropna():
                for entry in str(raw_platforms).split(','):
                    text = entry.strip()
                    if text:
                        platform_values.add(text)
        platforms_list = sorted(platform_values, key=str.casefold)

    if rebuild_navigator:
        previous_navigator = navigator
        try:
            navigator = GameNavigator(total_games)
        except Exception:
            logger.exception('Failed to rebuild navigator state')
            navigator = previous_navigator

    reset_source_index_cache()


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
_set_games_dataframe(load_games(), rebuild_metadata=True, rebuild_navigator=True)

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








def build_game_payload(index: int, seq: int, progress_seq: int | None = None) -> dict:
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


@dataclass
class DuplicateGroupResolution:
    canonical: sqlite3.Row
    duplicates: list[sqlite3.Row] = field(default_factory=list)
    metadata_updates: dict[str, Any] = field(default_factory=dict)


def _coerce_int(value: Any) -> int | None:
    try:
        if isinstance(value, numbers.Integral):
            return int(value)
        if isinstance(value, numbers.Real):
            float_value = float(value)
            if math.isnan(float_value):
                return None
            if float_value.is_integer():
                return int(float_value)
            return None
        text = str(value).strip()
        if not text:
            return None
        return int(text)
    except (TypeError, ValueError):
        return None


def _row_get(row: sqlite3.Row, column: str) -> Any:
    try:
        return row[column]
    except (KeyError, IndexError):
        return None


def _metadata_value_score(column: str, value: Any) -> float:
    if column in {'Summary', 'First Launch Date', 'Category'}:
        normalized = _normalize_text(value)
        return float(len(normalized)) if normalized else 0.0
    if column == 'Cover Path':
        text = str(value).strip() if value is not None else ''
        return 1.0 if text else 0.0
    if column in {'Width', 'Height'}:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return 0.0
        return numeric if numeric > 0 else 0.0
    if column == 'last_edited_at':
        if not value:
            return 0.0
        try:
            parsed = datetime.fromisoformat(str(value))
        except ValueError:
            return 0.0
        return parsed.timestamp()
    return 0.0


def _compute_metadata_updates(
    canonical: sqlite3.Row, duplicates: Iterable[sqlite3.Row]
) -> dict[str, Any]:
    duplicate_list = list(duplicates)
    if not duplicate_list:
        return {}
    updates: dict[str, Any] = {}
    metadata_columns = ['Summary', 'Cover Path', 'First Launch Date', 'Category', 'Width', 'Height', 'last_edited_at']
    for column in metadata_columns:
        canonical_value = _row_get(canonical, column)
        canonical_score = _metadata_value_score(column, canonical_value)
        best_value = canonical_value
        best_score = canonical_score
        for entry in duplicate_list:
            candidate_value = _row_get(entry, column)
            candidate_score = _metadata_value_score(column, candidate_value)
            if candidate_score > best_score:
                best_score = candidate_score
                best_value = candidate_value
        if best_score > canonical_score:
            updates[column] = best_value
    return updates


def _relation_count_from_row(row: sqlite3.Row) -> int:
    total = 0
    for relation in LOOKUP_RELATIONS:
        column_name = f"{relation['join_table']}_count"
        try:
            value = row[column_name]
        except (KeyError, IndexError):
            value = None
        count = _coerce_int(value)
        if count:
            total += count
    return total


def _choose_canonical_duplicate(group: list[sqlite3.Row]) -> sqlite3.Row | None:
    best_row: sqlite3.Row | None = None
    best_score: tuple[Any, ...] | None = None
    for entry in group:
        entry_id = _coerce_int(entry['ID'])
        if entry_id is None:
            continue
        relation_score = _relation_count_from_row(entry)
        metadata_presence = sum(
            1
            for column in ('Summary', 'Cover Path', 'First Launch Date', 'Category')
            if _metadata_value_score(column, _row_get(entry, column)) > 0
        )
        cover_score = _metadata_value_score('Cover Path', _row_get(entry, 'Cover Path'))
        summary_score = _metadata_value_score('Summary', _row_get(entry, 'Summary'))
        edited_score = _metadata_value_score('last_edited_at', _row_get(entry, 'last_edited_at'))
        score = (
            relation_score,
            metadata_presence,
            cover_score,
            summary_score,
            edited_score,
            -entry_id,
        )
        if best_score is None or score > best_score:
            best_score = score
            best_row = entry
    return best_row


def _scan_duplicate_candidates(
    rows: Iterable[sqlite3.Row],
    *,
    progress_callback: Callable[[int, int, int, int], None] | None = None,
) -> tuple[list[DuplicateGroupResolution], int, int, int]:
    groups: dict[tuple[str, str], list[sqlite3.Row]] = {}
    for row in rows:
        name_value = _normalize_text(row['Name']).casefold()
        igdb_value = coerce_igdb_id(row['igdb_id'])
        if not name_value or not igdb_value:
            continue
        groups.setdefault((name_value, igdb_value), []).append(row)

    group_values = list(groups.values())
    total_groups = len(group_values)

    duplicate_groups = 0
    skipped_groups = 0
    resolutions: list[DuplicateGroupResolution] = []
    for index, group in enumerate(group_values, start=1):
        if len(group) <= 1:
            continue
        duplicate_groups += 1
        canonical_row = _choose_canonical_duplicate(group)
        if canonical_row is None:
            skipped_groups += 1
            continue
        canonical_id = _coerce_int(canonical_row['ID'])
        if canonical_id is None:
            skipped_groups += 1
            continue
        duplicate_rows: list[sqlite3.Row] = []
        for entry in group:
            if entry is canonical_row:
                continue
            entry_id = _coerce_int(entry['ID'])
            if entry_id is None:
                continue
            duplicate_rows.append(entry)
        if not duplicate_rows:
            skipped_groups += 1
            continue
        metadata_updates = _compute_metadata_updates(canonical_row, duplicate_rows)
        resolutions.append(
            DuplicateGroupResolution(
                canonical=canonical_row,
                duplicates=duplicate_rows,
                metadata_updates=metadata_updates,
            )
        )
        if progress_callback is not None:
            progress_callback(index, total_groups or len(group_values) or 1, duplicate_groups, skipped_groups)

    return resolutions, duplicate_groups, skipped_groups, total_groups


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
    resolution_list = [resolution for resolution in resolutions if resolution.duplicates]
    if not resolution_list:
        return set()

    ids_to_delete: set[int] = set()
    canonical_ids: set[int] = set()

    with db_lock:
        conn = get_db()
        with conn:
            for resolution in resolution_list:
                canonical_id = _coerce_int(resolution.canonical['ID'])
                if canonical_id is None:
                    continue
                duplicate_ids: list[int] = []
                for duplicate_row in resolution.duplicates:
                    duplicate_id = _coerce_int(duplicate_row['ID'])
                    if duplicate_id is None:
                        continue
                    duplicate_ids.append(duplicate_id)
                    for relation in LOOKUP_RELATIONS:
                        join_table = relation['join_table']
                        join_column = relation['join_column']
                        conn.execute(
                            f'''
                                INSERT OR IGNORE INTO {join_table} (processed_game_id, {join_column})
                                SELECT ?, {join_column}
                                FROM {join_table}
                                WHERE processed_game_id = ?
                            ''',
                            (canonical_id, duplicate_id),
                        )
                        conn.execute(
                            f'DELETE FROM {join_table} WHERE processed_game_id = ?',
                            (duplicate_id,),
                        )
                if not duplicate_ids:
                    continue
                ids_to_delete.update(duplicate_ids)
                canonical_ids.add(canonical_id)
                if resolution.metadata_updates:
                    _apply_metadata_updates(conn, canonical_id, resolution.metadata_updates)
            if canonical_ids:
                _refresh_lookup_columns_for_games(conn, canonical_ids)

    return ids_to_delete


def _remove_processed_games(ids_to_delete: Iterable[int]) -> tuple[int, int]:
    global games_df, total_games, navigator

    unique_ids = sorted({int(game_id) for game_id in ids_to_delete if str(game_id).strip()})
    if not unique_ids:
        remaining_total = len(games_df) if games_df is not None else total_games
        return 0, remaining_total

    placeholders = ','.join('?' for _ in unique_ids)
    with db_lock:
        conn = get_db()
        cur = conn.execute(
            f'SELECT "ID", "Source Index" FROM processed_games WHERE "ID" IN ({placeholders})',
            tuple(unique_ids),
        )
        rows = cur.fetchall()

    if not rows:
        remaining_total = len(games_df) if games_df is not None else total_games
        return 0, remaining_total

    delete_params = [(row['ID'],) for row in rows]
    raw_source_indices = [row['Source Index'] for row in rows if row['Source Index'] is not None]

    with db_lock:
        conn = get_db()
        with conn:
            conn.executemany(
                'DELETE FROM igdb_updates WHERE processed_game_id=?',
                delete_params,
            )
            conn.executemany(
                'DELETE FROM processed_games WHERE "ID"=?',
                delete_params,
            )

    canonical_indices: set[str] = set()
    for value in raw_source_indices:
        canonical = _canonical_source_index(value)
        if canonical is not None:
            canonical_indices.add(canonical)

    removed_numeric = sorted(
        {int(candidate) for candidate in canonical_indices if candidate.isdigit()}
    )

    positions_to_remove: set[int] = set()
    for value in raw_source_indices:
        position = get_position_for_source_index(value)
        if position is None:
            canonical = _canonical_source_index(value)
            if canonical is not None and canonical != value:
                position = get_position_for_source_index(canonical)
        if position is not None:
            positions_to_remove.add(position)

    if games_df is not None:
        if not games_df.empty and positions_to_remove:
            drop_indices = sorted(positions_to_remove)
            games_df = games_df.drop(games_df.index[drop_indices]).reset_index(drop=True)
        if not games_df.empty:
            if 'Source Index' in games_df.columns:
                current_values = games_df['Source Index'].tolist()
            else:
                current_values = [str(idx) for idx in range(len(games_df))]
            new_values: list[str] = []
            for idx, value in enumerate(current_values):
                canonical = _canonical_source_index(value)
                if canonical is None:
                    new_values.append(str(idx))
                    continue
                if canonical.isdigit():
                    numeric_value = int(canonical)
                    shift = bisect_left(removed_numeric, numeric_value)
                    new_numeric = numeric_value - shift
                    stripped = str(value).strip()
                    if stripped.isdigit():
                        formatted = str(new_numeric).zfill(len(stripped))
                    else:
                        formatted = str(new_numeric)
                    new_values.append(formatted)
                else:
                    new_values.append(canonical)
            games_df = games_df.copy()
            games_df['Source Index'] = new_values
        total_games = len(games_df)
        try:
            reset_source_index_cache()
        except Exception:
            pass
    else:
        total_games = 0

    if removed_numeric:
        with db_lock:
            conn = get_db()
            with conn:
                cur = conn.execute(
                    'SELECT "ID", "Source Index" FROM processed_games'
                )
                stored_rows = cur.fetchall()
                for entry in stored_rows:
                    canonical = _canonical_source_index(entry['Source Index'])
                    if canonical is None or not canonical.isdigit():
                        continue
                    numeric_value = int(canonical)
                    shift = bisect_left(removed_numeric, numeric_value)
                    if shift <= 0:
                        continue
                    new_numeric = numeric_value - shift
                    stored_text = str(entry['Source Index'])
                    stripped = stored_text.strip()
                    if stripped.isdigit():
                        new_value = str(new_numeric).zfill(len(stripped))
                    else:
                        new_value = str(new_numeric)
                    conn.execute(
                        'UPDATE processed_games SET "Source Index"=? WHERE "ID"=?',
                        (new_value, entry['ID']),
                    )

    normalize_processed_games()

    try:
        navigator = GameNavigator(total_games)
    except Exception:
        pass

    remaining_total = total_games
    removed_count = len(delete_params)
    return removed_count, remaining_total


def fetch_cached_updates() -> list[dict[str, Any]]:
    with db_lock:
        conn = get_db()
        processed_columns = get_processed_games_columns(conn)
        cover_url_select = (
            'p."Large Cover Image (URL)" AS cover_url'
            if 'Large Cover Image (URL)' in processed_columns
            else 'NULL AS cover_url'
        )
        cur = conn.execute(
            f'''SELECT
                   u.processed_game_id,
                   u.igdb_id,
                   u.igdb_updated_at,
                   u.local_last_edited_at,
                   u.refreshed_at,
                   u.diff,
                   p."Name" AS game_name,
                   p."Cover Path" AS cover_path,
                   {cover_url_select}
               FROM igdb_updates u
               LEFT JOIN processed_games p ON p."ID" = u.processed_game_id
               ORDER BY p."Name" COLLATE NOCASE
            '''
        )
        update_rows = cur.fetchall()

        relation_count_sql = ', '.join(
            f'(SELECT COUNT(*) FROM {relation["join_table"]} WHERE processed_game_id = p."ID") AS {relation["join_table"]}_count'
            for relation in LOOKUP_RELATIONS
        )
        duplicate_cur = conn.execute(
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
                   cache.updated_at AS cache_updated_at
               FROM processed_games p
               LEFT JOIN {IGDB_CACHE_TABLE} cache ON cache.igdb_id = p."igdb_id"
            '''
        )
        duplicate_rows = duplicate_cur.fetchall()

    updates: list[dict[str, Any]] = []
    existing_ids: set[int] = set()
    for row in update_rows:
        has_diff = bool(row['diff'])
        try:
            processed_id = int(row['processed_game_id'])
        except (TypeError, ValueError):
            processed_id = None
        if processed_id is not None:
            existing_ids.add(processed_id)
        updates.append(
            {
                'processed_game_id': row['processed_game_id'],
                'igdb_id': row['igdb_id'],
                'igdb_updated_at': row['igdb_updated_at'],
                'local_last_edited_at': row['local_last_edited_at'],
                'refreshed_at': row['refreshed_at'],
                'name': row['game_name'],
                'has_diff': has_diff,
                'cover': load_cover_data(row['cover_path'], row['cover_url']),
                'update_type': 'mismatch' if has_diff else None,
                'detail_available': True,
            }
        )

    duplicate_resolutions, _, _, _ = _scan_duplicate_candidates(duplicate_rows)
    for resolution in duplicate_resolutions:
        for row in resolution.duplicates:
            try:
                processed_id = int(row['ID'])
            except (TypeError, ValueError):
                continue
            if processed_id in existing_ids:
                continue
            existing_ids.add(processed_id)
            igdb_updated = _normalize_timestamp(row['cache_updated_at']) if row['cache_updated_at'] else ''
            updates.append(
                {
                    'processed_game_id': processed_id,
                    'igdb_id': row['igdb_id'],
                    'igdb_updated_at': igdb_updated,
                    'local_last_edited_at': row['last_edited_at'],
                    'refreshed_at': None,
                    'name': row['Name'],
                    'has_diff': False,
                    'cover': load_cover_data(row['Cover Path'], row['cover_url']),
                    'update_type': 'duplicate',
                    'detail_available': False,
                }
            )

    return updates


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


def _run_refresh_diff_phase(update_progress: Callable[..., None]) -> dict[str, Any]:
    processed_rows = _collect_processed_games_with_igdb()
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
            payload = igdb_payloads.get(normalized_id)
            if not payload and normalized_id.isdigit():
                payload = igdb_payloads.get(str(int(normalized_id)))
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
            updated_count += 1
            processed += 1
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


def _execute_refresh_job(update_progress: Callable[..., None]) -> dict[str, Any]:
    cache_summary = None
    cache_error: str | None = None
    try:
        cache_summary = _run_refresh_cache_phase(update_progress)
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




def _execute_fix_names_job(
    update_progress: Callable[..., None],
    *,
    offset: int = 0,
    limit: int | None = None,
    process_all: bool = True,
) -> dict[str, Any]:
    try:
        start_offset = int(offset)
    except (TypeError, ValueError):
        start_offset = 0
    if start_offset < 0:
        start_offset = 0
    limit_value: int | None = None
    if limit is not None:
        try:
            limit_value = int(limit)
        except (TypeError, ValueError):
            limit_value = None
    if limit_value is None or limit_value <= 0:
        limit_value = FIX_NAMES_BATCH_LIMIT
    limit_value = max(1, min(int(limit_value), 200))

    with db_lock:
        conn = get_db()
        cur = conn.execute(
            'SELECT COUNT(*) AS total FROM processed_games '
            'WHERE TRIM(COALESCE("igdb_id", "")) != ""'
        )
        total_row = cur.fetchone()

    total = 0
    if total_row is not None:
        try:
            total = int(total_row['total'])
        except (KeyError, TypeError, ValueError):
            try:
                total = int(total_row[0])
            except (IndexError, TypeError, ValueError):
                total = 0
    if total < 0:
        total = 0

    update_progress(total=total, current=0, message='Scanning processed games…')

    if total == 0:
        return {
            'status': 'ok',
            'total': 0,
            'processed': 0,
            'updated': 0,
            'unchanged': 0,
            'missing': [],
            'missing_remote': [],
            'missing_name': [],
            'invalid': 0,
            'toast_type': 'warning',
            'message': 'No games with an IGDB ID were found.',
            'done': True,
            'next_offset': 0,
        }

    current_offset = start_offset
    processed = start_offset
    updated_total = 0
    unchanged_total = 0
    invalid_total = 0
    missing_remote: set[str] = set()
    missing_name: set[str] = set()

    timestamp = now_utc_iso()

    while current_offset < total:
        with db_lock:
            conn = get_db()
            cur = conn.execute(
                '''SELECT "ID", "igdb_id", "Name" FROM processed_games
                   WHERE TRIM(COALESCE("igdb_id", "")) != ""
                   ORDER BY "ID"
                   LIMIT ? OFFSET ?''',
                (limit_value, current_offset),
            )
            rows = cur.fetchall()

        batch_count = len(rows)
        if batch_count == 0:
            break

        entries: list[dict[str, Any]] = []
        unique_ids: list[str] = []
        seen_ids: set[str] = set()

        for row in rows:
            db_id = row['ID']
            raw_igdb_id = row['igdb_id']
            igdb_id = coerce_igdb_id(raw_igdb_id)
            current_name = _normalize_text(row['Name'])
            entries.append(
                {
                    'id': db_id,
                    'igdb_id': igdb_id,
                    'current_name': current_name,
                }
            )
            if igdb_id:
                if igdb_id not in seen_ids:
                    seen_ids.add(igdb_id)
                    unique_ids.append(igdb_id)
            else:
                invalid_total += 1

        metadata: dict[str, Mapping[str, Any]] = {}
        if unique_ids:
            metadata = fetch_igdb_metadata(unique_ids) or {}

        updates: list[tuple[str, str, Any]] = []

        for entry in entries:
            igdb_id = entry['igdb_id']
            if not igdb_id:
                continue
            payload = metadata.get(igdb_id)
            if not isinstance(payload, Mapping):
                missing_remote.add(igdb_id)
                continue
            remote_name = _normalize_text(payload.get('name'))
            if not remote_name:
                missing_name.add(igdb_id)
                continue
            current_name = entry['current_name']
            if remote_name == current_name:
                unchanged_total += 1
                continue
            db_id = entry['id']
            try:
                numeric_id = int(db_id)
            except (TypeError, ValueError):
                missing_remote.add(igdb_id)
                continue
            updates.append((remote_name, timestamp, numeric_id))
            updated_total += 1

        if updates:
            with db_lock:
                conn = get_db()
            conn.executemany(
                'UPDATE processed_games SET "Name"=?, last_edited_at=? WHERE "ID"=?',
                updates,
            )
            conn.commit()

        current_offset += batch_count
        processed = min(current_offset, total)
        update_progress(
            current=processed,
            total=total,
            message='Fixing IGDB names…',
            data={
                'updated': updated_total,
                'unchanged': unchanged_total,
                'invalid': invalid_total,
                'missing_remote': len(missing_remote),
                'missing_name': len(missing_name),
            },
        )

        if not process_all:
            break

    missing_remote_list = sorted(missing_remote)
    missing_name_list = sorted(missing_name)
    missing_combined = sorted({*missing_remote, *missing_name})

    toast_type = 'success'
    if updated_total > 0:
        message = f"Updated {updated_total} game name{'s' if updated_total != 1 else ''} from IGDB."
    else:
        message = 'No game names required updating.'
    if processed == 0:
        message = 'No games with an IGDB ID were found.'
        toast_type = 'warning'
    if missing_combined:
        plural = 's' if len(missing_combined) != 1 else ''
        message += f" {len(missing_combined)} IGDB record{plural} missing."
        toast_type = 'warning'

    update_progress(
        current=processed,
        total=total,
        message='Finished fixing IGDB names.',
        data={
            'updated': updated_total,
            'unchanged': unchanged_total,
            'invalid': invalid_total,
            'missing_remote': len(missing_remote),
            'missing_name': len(missing_name),
        },
    )

    processed_value = min(processed, total)

    return {
        'status': 'ok',
        'total': total,
        'processed': processed_value,
        'updated': updated_total,
        'unchanged': unchanged_total,
        'missing': missing_combined,
        'missing_remote': missing_remote_list,
        'missing_name': missing_name_list,
        'invalid': invalid_total,
        'toast_type': toast_type,
        'message': message.strip(),
        'done': processed_value >= total,
        'next_offset': processed_value,
    }




def _execute_remove_duplicates_job(update_progress: Callable[..., None]) -> dict[str, Any]:
    global games_df, total_games, navigator

    update_progress(message='Scanning for duplicates…', data={'phase': 'dedupe'}, current=0, total=0)

    with db_lock:
        conn = get_db()
        relation_count_sql = ', '.join(
            f'(SELECT COUNT(*) FROM {relation["join_table"]} WHERE processed_game_id = pg."ID") AS {relation["join_table"]}_count'
            for relation in LOOKUP_RELATIONS
        )
        cur = conn.execute(
            f'''SELECT
                    pg."ID",
                    pg."Source Index",
                    pg."Name",
                    pg."igdb_id",
                    pg."Summary",
                    pg."Cover Path",
                    pg."First Launch Date",
                    pg."Category",
                    pg."Width",
                    pg."Height",
                    pg.last_edited_at,
                    {relation_count_sql}
               FROM processed_games AS pg'''
        )
        rows = cur.fetchall()

    def _progress_callback(index: int, total_groups: int, duplicate_groups: int, skipped: int) -> None:
        update_progress(
            current=index,
            total=total_groups or len(rows) or 1,
            message='Evaluating duplicate groups…',
            data={
                'phase': 'dedupe',
                'duplicate_groups': duplicate_groups,
                'skipped': skipped,
            },
        )

    resolutions, duplicate_groups, skipped_groups, total_groups = _scan_duplicate_candidates(
        rows, progress_callback=_progress_callback
    )

    ids_to_delete = _merge_duplicate_resolutions(resolutions)

    if not ids_to_delete:
        remaining_total = len(games_df) if games_df is not None else total_games
        message = (
            'No removable duplicates found.'
            if duplicate_groups
            else 'No duplicates detected.'
        )
        toast_type = 'info'
        if skipped_groups and not duplicate_groups:
            toast_type = 'warning'
        update_progress(
            message=message,
            current=total_groups,
            total=total_groups or len(rows) or 1,
            data={
                'phase': 'dedupe',
                'removed': 0,
                'duplicate_groups': duplicate_groups,
                'skipped': skipped_groups,
            },
        )
        return {
            'status': 'ok',
            'removed': 0,
            'duplicate_groups': duplicate_groups,
            'skipped': skipped_groups,
            'remaining': remaining_total,
            'message': message,
            'toast_type': toast_type,
        }

    removed_count, remaining_total = _remove_processed_games(ids_to_delete)

    message = (
        f"Removed {removed_count} duplicate{'s' if removed_count != 1 else ''}."
        if removed_count > 0
        else 'No removable duplicates found.'
    )
    toast_type = 'success' if removed_count > 0 else 'info'
    if skipped_groups and removed_count == 0:
        toast_type = 'warning'
        message += f" Skipped {skipped_groups} duplicate group{'s' if skipped_groups != 1 else ''}."

    update_progress(
        message='Removed duplicate entries.',
        current=total_groups or len(resolutions) or 1,
        total=total_groups or len(resolutions) or 1,
        data={
            'phase': 'dedupe',
            'removed': removed_count,
            'duplicate_groups': duplicate_groups,
            'skipped': skipped_groups,
        },
    )

    return {
        'status': 'ok',
        'removed': removed_count,
        'duplicate_groups': duplicate_groups,
        'skipped': skipped_groups,
        'remaining': remaining_total,
        'message': message.strip(),
        'toast_type': toast_type,
    }






















routes_games.configure({
    'navigator': navigator,
    'get_navigator': lambda: navigator,
    'get_total_games': lambda: total_games,
    'get_games_df': lambda: games_df,
    'build_game_payload': build_game_payload,
    'generate_pt_summary': generate_pt_summary,
    'open_image_auto_rotate': open_image_auto_rotate,
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
    '_row_value': _row_value,
    '_normalize_lookup_name': _normalize_lookup_name,
    '_get_or_create_lookup_id': _get_or_create_lookup_id,
    '_lookup_name_for_id': _lookup_name_for_id,
    '_fetch_lookup_entries_for_game': _fetch_lookup_entries_for_game,
    '_lookup_entries_to_selection': _lookup_entries_to_selection,
    '_persist_lookup_relations': _persist_lookup_relations,
    '_apply_lookup_entries_to_processed_game': _apply_lookup_entries_to_processed_game,
    '_remove_lookup_id_from_entries': _remove_lookup_id_from_entries,
})

routes_updates.configure({
    'IGDB_BATCH_SIZE': IGDB_BATCH_SIZE,
    'FIX_NAMES_BATCH_LIMIT': FIX_NAMES_BATCH_LIMIT,
    'exchange_twitch_credentials': lambda: exchange_twitch_credentials,
    'db_lock': db_lock,
    'get_db': get_db,
    '_get_cached_igdb_total': _get_cached_igdb_total,
    '_set_cached_igdb_total': _set_cached_igdb_total,
    'download_igdb_game_count': lambda: download_igdb_game_count,
    'download_igdb_games': lambda: download_igdb_games,
    '_upsert_igdb_cache_entries': _upsert_igdb_cache_entries,
    '_igdb_prefill_lock': _igdb_prefill_lock,
    '_igdb_prefill_cache': _igdb_prefill_cache,
    '_execute_refresh_job': _execute_refresh_job,
    'job_manager': job_manager,
    '_execute_fix_names_job': _execute_fix_names_job,
    '_execute_remove_duplicates_job': _execute_remove_duplicates_job,
    'fetch_cached_updates': fetch_cached_updates,
    'get_processed_games_columns': get_processed_games_columns,
    'load_cover_data': load_cover_data,
    'LOOKUP_RELATIONS': LOOKUP_RELATIONS,
    '_scan_duplicate_candidates': _scan_duplicate_candidates,
    '_coerce_int': _coerce_int,
    '_compute_metadata_updates': _compute_metadata_updates,
    '_merge_duplicate_resolutions': _merge_duplicate_resolutions,
    '_remove_processed_games': _remove_processed_games,
    '_normalize_text': _normalize_text,
    'DuplicateGroupResolution': DuplicateGroupResolution,
})

app.register_blueprint(routes_games.games_blueprint)
app.register_blueprint(routes_lookups.lookups_blueprint)
app.register_blueprint(routes_updates.updates_blueprint)

if __name__ == '__main__':
    app.run(debug=True)
