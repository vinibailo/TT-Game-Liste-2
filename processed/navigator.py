"""Navigation helpers for iterating through processed games."""
from __future__ import annotations

import json
import logging
from threading import Lock
from typing import Any, Callable

from sqlalchemy import select
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.sql import column, table

import pandas as pd

from . import source_index_cache
from db import utils as db_utils


class GameNavigator:
    """Thread-safe helper to navigate game list and track progress."""

    def __init__(
        self,
        *,
        db_lock: Lock,
        get_db: Callable[[], db_utils.DatabaseHandle],
        is_processed_game_done: Callable[[Any, Any], bool],
        logger: logging.Logger | None = None,
    ) -> None:
        self.lock = Lock()
        self._cache_lock = Lock()
        self._db_lock = db_lock
        self._get_db = get_db
        self._is_processed_game_done = is_processed_game_done
        self._logger = logger or logging.getLogger(__name__)

        self._games_df: pd.DataFrame = pd.DataFrame()
        self.total: int = 0
        self.current_index: int = 0
        self.seq_index: int = 1
        self.processed_total: int = 0
        self.skip_queue: list[dict[str, int]] = []

    @property
    def games_df(self) -> pd.DataFrame:
        """Return the currently loaded games dataframe."""

        return self._games_df

    def set_games_df(
        self,
        games_df: pd.DataFrame | None,
        *,
        rebuild_state: bool = True,
    ) -> None:
        """Assign a new games dataframe and optionally rebuild persisted state."""

        old_df = self._games_df

        if games_df is None:
            games_df = pd.DataFrame()
        self._games_df = games_df
        self.total = len(games_df)
        source_index_cache.invalidate_cache(old_df)
        self.reset_source_index_cache()
        if rebuild_state:
            self._load_initial()
        else:
            if self.current_index > self.total:
                self.current_index = self.total
            if self.seq_index < 1:
                self.seq_index = 1
            if self.processed_total > self.total:
                self.processed_total = self.total

    @staticmethod
    def canonical_source_index(value: Any) -> str | None:
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

    def reset_source_index_cache(self) -> None:
        """Clear cached mappings between navigator positions and ``Source Index`` values."""

        with self._cache_lock:
            source_index_cache.invalidate_cache(self._games_df)

    def _ensure_source_index_cache(self) -> tuple[dict[int, str], dict[str, int]]:
        """Build and return cached lookup tables for ``Source Index`` values."""

        df = self._games_df
        if df is None:
            raise RuntimeError('games_df is not loaded')

        with self._cache_lock:
            mapping, reverse = source_index_cache.build_source_index_mappings(
                df, canonicalize=self.canonical_source_index
            )

        return mapping, reverse

    def get_source_index_for_position(self, position: int) -> str:
        """Return the normalized ``Source Index`` string for a DataFrame position."""

        if position < 0:
            raise IndexError('invalid index')
        mapping, _ = self._ensure_source_index_cache()
        try:
            return mapping[position]
        except KeyError as exc:
            raise IndexError('invalid index') from exc

    def get_position_for_source_index(self, value: Any) -> int | None:
        """Resolve a ``Source Index`` value back to its DataFrame position."""

        canonical = self.canonical_source_index(value)
        if canonical is None:
            return None
        mapping, reverse = self._ensure_source_index_cache()
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

    def _load_initial(self) -> None:
        with self._db_lock:
            db_handle = self._get_db()
            with db_handle.sa_connection() as sa_conn:
                navigator_state = table(
                    "navigator_state",
                    column("id"),
                    column("current_index"),
                    column("seq_index"),
                    column("skip_queue"),
                )
                processed_games = table(
                    "processed_games",
                    column("Source Index"),
                    column("ID"),
                    column("Summary"),
                    column("Cover Path"),
                )

                state_stmt = (
                    select(
                        navigator_state.c.current_index,
                        navigator_state.c.seq_index,
                        navigator_state.c.skip_queue,
                    )
                    .where(navigator_state.c.id == 1)
                    .limit(1)
                )
                games_stmt = select(
                    processed_games.c["Source Index"],
                    processed_games.c["ID"],
                    processed_games.c["Summary"],
                    processed_games.c["Cover Path"],
                )

                state_row = sa_conn.execute(state_stmt).mappings().fetchone()
                rows = sa_conn.execute(games_stmt).mappings().all()
        processed: set[int] = set()
        max_seq = 0
        for row in rows:
            position = self.get_position_for_source_index(row['Source Index'])
            if position is not None and 0 <= position < self.total:
                try:
                    summary_value = row['Summary']
                except (KeyError, IndexError, TypeError):
                    summary_value = None
                try:
                    cover_value = row['Cover Path']
                except (KeyError, IndexError, TypeError):
                    cover_value = None
                if self._is_processed_game_done(summary_value, cover_value):
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
                    self._logger.debug(
                        "Loaded progress: current_index=%s seq_index=%s skip_queue=%s",
                        self.current_index,
                        self.seq_index,
                        self.skip_queue,
                    )
                    return
                self._logger.warning("Navigator state out of sync with database; rebuilding")
            except Exception as exc:  # pragma: no cover - defensive logging
                self._logger.warning("Failed to load navigator state: %s", exc)
        self.current_index = next_index
        self.seq_index = expected_seq
        self.skip_queue = []
        self._logger.debug(
            "Loaded progress: current_index=%s seq_index=%s skip_queue=%s",
            self.current_index,
            self.seq_index,
            self.skip_queue,
        )
        self._save()

    def _load(self) -> None:
        with self._db_lock:
            db_handle = self._get_db()
            with db_handle.sa_connection() as sa_conn:
                navigator_state = table(
                    "navigator_state",
                    column("id"),
                    column("current_index"),
                    column("seq_index"),
                    column("skip_queue"),
                )
                processed_games = table(
                    "processed_games",
                    column("Summary"),
                    column("Cover Path"),
                )

                state_stmt = (
                    select(
                        navigator_state.c.current_index,
                        navigator_state.c.seq_index,
                        navigator_state.c.skip_queue,
                    )
                    .where(navigator_state.c.id == 1)
                    .limit(1)
                )
                games_stmt = select(
                    processed_games.c["Summary"],
                    processed_games.c["Cover Path"],
                )

                state_row = sa_conn.execute(state_stmt).mappings().fetchone()
                rows = sa_conn.execute(games_stmt).mappings().all()
        processed_total = 0
        for row in rows:
            try:
                summary_value = (
                    row.get("Summary") if hasattr(row, "get") else row["Summary"]
                )
            except (KeyError, IndexError, TypeError):
                summary_value = None
            try:
                cover_value = (
                    row.get("Cover Path") if hasattr(row, "get") else row["Cover Path"]
                )
            except (KeyError, IndexError, TypeError):
                cover_value = None
            if self._is_processed_game_done(summary_value, cover_value):
                processed_total += 1
        self.processed_total = max(processed_total, 0)
        if state_row is not None:
            try:
                current_value = (
                    state_row.get("current_index")
                    if hasattr(state_row, "get")
                    else state_row["current_index"]
                )
                seq_value = (
                    state_row.get("seq_index")
                    if hasattr(state_row, "get")
                    else state_row["seq_index"]
                )
                skip_value = (
                    state_row.get("skip_queue")
                    if hasattr(state_row, "get")
                    else state_row["skip_queue"]
                )
                self.current_index = int(current_value)
                self.seq_index = int(seq_value)
                self.skip_queue = json.loads(skip_value or '[]')
                self._logger.debug(
                    "Loaded progress: current_index=%s seq_index=%s skip_queue=%s",
                    self.current_index,
                    self.seq_index,
                    self.skip_queue,
                )
                return
            except Exception as exc:  # pragma: no cover - defensive logging
                self._logger.warning("Failed to load navigator state: %s", exc)
        self._load_initial()

    def _save(self) -> None:
        try:
            with self._db_lock:
                db_handle = self._get_db()
                navigator_state = table(
                    "navigator_state",
                    column("id"),
                    column("current_index"),
                    column("seq_index"),
                    column("skip_queue"),
                )
                values = {
                    "id": 1,
                    "current_index": self.current_index,
                    "seq_index": self.seq_index,
                    "skip_queue": json.dumps(self.skip_queue),
                }
                engine = getattr(db_handle, "engine", None)
                dialect_name = getattr(getattr(engine, "dialect", None), "name", "")
                if dialect_name in {"mariadb", "mysql"}:
                    stmt = mysql_insert(navigator_state).values(**values)
                    stmt = stmt.on_duplicate_key_update(
                        current_index=stmt.inserted.current_index,
                        seq_index=stmt.inserted.seq_index,
                        skip_queue=stmt.inserted.skip_queue,
                    )
                elif dialect_name == "sqlite":
                    stmt = navigator_state.insert().values(**values).prefix_with("OR REPLACE")
                else:
                    stmt = navigator_state.insert().values(**values)

                with db_handle.sa_connection() as sa_conn:
                    with sa_conn.begin():
                        sa_conn.execute(stmt)
        except Exception as exc:  # pragma: no cover - defensive logging
            self._logger.warning("Failed to save navigator state: %s", exc)

    def _process_skip_queue(self) -> None:
        self._logger.debug(
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
                self._logger.debug(
                    "Skip queue hit: index from %s to %s",
                    old_index,
                    self.current_index,
                )
                break
        self._logger.debug(
            "After processing skip queue: index=%s queue=%s",
            self.current_index,
            self.skip_queue,
        )

    def next(self) -> int:
        with self.lock:
            self._load()
            before = self.current_index
            self._logger.debug("next() before skip queue: index=%s", before)
            self._process_skip_queue()
            self._logger.debug("next() after skip queue: index=%s", self.current_index)
            if self.current_index < self.total:
                self.current_index += 1
            self._logger.debug("next() after increment: index=%s", self.current_index)
            self._save()
            return self.current_index

    def back(self) -> int:
        with self.lock:
            self._load()
            before = self.current_index
            self._logger.debug("back() before skip queue: index=%s", before)
            self._process_skip_queue()
            self._logger.debug("back() after skip queue: index=%s", self.current_index)
            if self.current_index > 0:
                self.current_index -= 1
            self._logger.debug("back() after decrement: index=%s", self.current_index)
            self._save()
            return self.current_index

    def current(self) -> int:
        with self.lock:
            self._load()
            return self.current_index

    def completion_percentage(self) -> float:
        """Return the percentage of games that have been processed."""

        with self.lock:
            self._load()
            if self.total <= 0:
                return 0.0
            completed = min(max(self.processed_total, 0), self.total)
            return (completed / self.total) * 100.0

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
