"""Utilities for building IGDB diff reports."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
import numbers
from typing import Any

DEFAULT_FIELD_TYPES: dict[str, str] = {
    "first_release_date": "timestamp",
    "genres": "list",
    "platforms": "list",
    "game_modes": "list",
    "developers": "list",
    "publishers": "list",
}

__all__ = [
    "DEFAULT_FIELD_TYPES",
    "build_diff_report",
]


def build_diff_report(
    cached_payload: Mapping[str, Any],
    new_payload: Mapping[str, Any],
    *,
    field_types: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Return a diff describing the differences between two payloads.

    The structure of the returned diff mirrors the one used by the Flask
    application: each key maps to a dictionary containing ``added`` and
    ``removed`` values.  Lists are compared by value (after normalising
    textual representation) while scalar values are coerced to strings.

    Parameters
    ----------
    cached_payload:
        The payload previously stored in the local cache.
    new_payload:
        The fresh payload obtained from IGDB.
    field_types:
        Optional mapping describing how specific keys should be treated.
        Supported field types are ``"list"``, ``"timestamp"`` and
        ``"text"`` (default).
    """

    types = dict(DEFAULT_FIELD_TYPES)
    if field_types:
        types.update(field_types)

    diff: dict[str, Any] = {}
    keys = set(cached_payload.keys()) | set(new_payload.keys())
    for key in sorted(keys):
        field_type = types.get(key, "text")
        cached_value = cached_payload.get(key)
        new_value = new_payload.get(key)

        if field_type == "list":
            cached_items = _normalize_sequence(cached_value)
            new_items = _normalize_sequence(new_value)
            added = sorted(set(new_items) - set(cached_items))
            removed = sorted(set(cached_items) - set(new_items))
            if added or removed:
                diff[key] = {"added": added, "removed": removed}
            continue

        if field_type == "timestamp":
            cached_text = _normalize_timestamp(cached_value)
            new_text = _normalize_timestamp(new_value)
        else:
            cached_text = _normalize_text(cached_value)
            new_text = _normalize_text(new_value)

        if cached_text == new_text:
            continue

        entry: dict[str, Any] = {}
        if new_text:
            entry["added"] = new_text
        if cached_text:
            entry["removed"] = cached_text
        diff[key] = entry

    return diff


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, numbers.Number):
        if isinstance(value, numbers.Real) and float(value).is_integer():
            return str(int(value))
        return str(value)
    return str(value).strip()


def _normalize_sequence(value: Any) -> list[str]:
    items: list[str] = []
    seen: set[str] = set()

    for candidate in _iter_sequence_items(value):
        text = _normalize_text(candidate)
        if not text:
            continue
        fingerprint = text.casefold()
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        items.append(text)

    return items


def _iter_sequence_items(value: Any) -> Iterable[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        yield from (part for part in value.split(",") if part is not None)
        return
    if isinstance(value, Mapping):
        for key in ("name", "value", "label", "title"):
            if key in value:
                yield value[key]
                return
        if "id" in value:
            yield value["id"]
            return
        yield value
        return
    try:
        iterator = iter(value)
    except TypeError:
        yield value
        return
    for item in iterator:
        if isinstance(item, Mapping):
            yield from _iter_sequence_items(item)
        elif isinstance(item, str):
            yield from (part for part in item.split(",") if part is not None)
        else:
            try:
                iter(item)
            except TypeError:
                yield item
            else:
                yield from _iter_sequence_items(item)


def _normalize_timestamp(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, numbers.Number):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc).date().isoformat()
        except Exception:
            return ""
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return ""
        try:
            return datetime.fromtimestamp(float(stripped), tz=timezone.utc).date().isoformat()
        except Exception:
            return stripped
    return _normalize_text(value)
