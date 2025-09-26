"""Utilities for caching and mapping processed source index data."""

from __future__ import annotations

from threading import Lock
from typing import Any, Callable

import pandas as pd

SOURCE_INDEX_COLUMN = "Source Index"

_CACHE_LOCK = Lock()
_CACHE: dict[int, tuple[int, dict[int, str], dict[str, int]]] = {}


def _compute_mappings(
    df: pd.DataFrame, canonicalize: Callable[[Any], str | None]
) -> tuple[dict[int, str], dict[str, int]]:
    mapping: dict[int, str] = {}
    reverse: dict[str, int] = {}

    if df.empty:
        return mapping, reverse

    source_values: list[Any] | None = None
    if SOURCE_INDEX_COLUMN in df.columns:
        source_values = df[SOURCE_INDEX_COLUMN].tolist()

    for position in range(len(df)):
        if source_values is not None and position < len(source_values):
            raw_value = source_values[position]
        else:
            raw_value = position

        canonical = canonicalize(raw_value)
        if canonical is None:
            canonical = str(position)

        mapping[position] = canonical
        reverse.setdefault(canonical, position)

    return mapping, reverse


def build_source_index_mappings(
    df: pd.DataFrame, *, canonicalize: Callable[[Any], str | None]
) -> tuple[dict[int, str], dict[str, int]]:
    """Return cached mappings for ``Source Index`` lookups on ``df``."""

    if df is None:
        raise RuntimeError("games_df is not loaded")

    df_key = id(df)
    df_length = len(df)

    with _CACHE_LOCK:
        cached = _CACHE.get(df_key)
        if cached and cached[0] == df_length:
            return cached[1], cached[2]

    mapping, reverse = _compute_mappings(df, canonicalize)

    with _CACHE_LOCK:
        _CACHE[df_key] = (df_length, mapping, reverse)

    return mapping, reverse


def invalidate_cache(df: pd.DataFrame | None = None) -> None:
    """Invalidate cached mappings for ``df`` or clear all caches when ``None``."""

    with _CACHE_LOCK:
        if df is None:
            _CACHE.clear()
        else:
            _CACHE.pop(id(df), None)


__all__ = [
    "SOURCE_INDEX_COLUMN",
    "build_source_index_mappings",
    "invalidate_cache",
]
