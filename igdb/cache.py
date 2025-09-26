"""IGDB cache management utilities.

This module provides a very small persistence layer for IGDB API
responses.  The production application stores the authoritative cache in
SQLite, however having a lightweight filesystem cache is extremely
useful when running scripts or tests that operate outside of the main
Flask app.  The helpers implemented here keep the responsibilities
deliberately small:

* Persist raw IGDB payloads under an arbitrary cache directory.
* Reload cached payloads and expose them in a convenient structure.
* Compare cached entries with freshly downloaded payloads and expose the
  diffs via :func:`fetch_cached_updates`.

The functions are designed to be dependency free so that they can be
used from unit tests or ad-hoc maintenance scripts without importing the
entire application module.  Any domain specific handling (for example
mapping genres to localised labels) is delegated to
``igdb.diff.build_diff_report`` which receives the raw payloads and
produces a diff report.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, MutableMapping
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
import re
from pathlib import Path
from typing import Any, Callable

from . import diff as diff_utils

logger = logging.getLogger(__name__)


__all__ = [
    "CacheEntry",
    "persist_igdb_payloads",
    "iter_cached_entries",
    "load_cached_entry",
    "fetch_cached_updates",
]


_SAFE_FILENAME_RE = re.compile(r"[^0-9A-Za-z_.-]+")


@dataclass(slots=True)
class CacheEntry:
    """Representation of a cached IGDB payload."""

    igdb_id: str
    payload: Mapping[str, Any]
    cached_at: str | None = None
    path: Path | None = None


def persist_igdb_payloads(
    cache_dir: str | Path,
    payloads: Iterable[Mapping[str, Any]],
    *,
    id_field: str = "id",
    timestamp_factory: Callable[[], str] | None = None,
) -> dict[str, Any]:
    """Persist a collection of IGDB payloads to ``cache_dir``.

    Parameters
    ----------
    cache_dir:
        Directory in which JSON cache files should be stored.
    payloads:
        Iterable of IGDB payloads (as dictionaries) that should be
        persisted.  Each payload must expose an identifier, typically
        under the ``id`` key but this can be customised via ``id_field``.
    id_field:
        Name of the key that should be used to extract the IGDB
        identifier.  The helper will also fall back to ``igdb_id`` and
        ``id`` automatically so callers rarely need to override this.
    timestamp_factory:
        Optional callable returning an ISO formatted timestamp.  When not
        provided a ``datetime.utcnow()`` timestamp is generated for each
        payload.

    Returns
    -------
    dict
        A summary containing the number of ``inserted``, ``updated`` and
        ``unchanged`` payloads together with the ``entries`` that were
        processed.
    """

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    inserted = updated = unchanged = 0
    entries: list[CacheEntry] = []

    for payload in payloads:
        if not isinstance(payload, Mapping):
            logger.debug("Skipping non-mapping payload: %r", payload)
            continue

        identifier = _extract_identifier(payload, id_field=id_field)
        if identifier is None:
            logger.debug("Skipping payload without identifier: %r", payload)
            continue

        file_path = cache_path / f"{_sanitize_identifier(identifier)}.json"

        cached_at = _resolve_timestamp(timestamp_factory)
        record: MutableMapping[str, Any] = {
            "igdb_id": identifier,
            "cached_at": cached_at,
            "payload": dict(payload),
        }

        existing: Mapping[str, Any] | None = None
        if file_path.exists():
            existing = _load_json(file_path)

        if existing and isinstance(existing.get("payload"), Mapping):
            existing_payload = existing["payload"]
            if _payloads_equal(existing_payload, payload):
                unchanged += 1
                cached_at = str(existing.get("cached_at") or cached_at)
                entries.append(
                    CacheEntry(
                        igdb_id=identifier,
                        payload=existing_payload,
                        cached_at=cached_at,
                        path=file_path,
                    )
                )
                continue
            updated += 1
        else:
            if existing is not None:
                logger.warning("Discarding corrupt cache entry at %s", file_path)
            inserted += 1

        _dump_json(file_path, record)
        entries.append(
            CacheEntry(
                igdb_id=identifier,
                payload=record["payload"],
                cached_at=cached_at,
                path=file_path,
            )
        )

    return {
        "inserted": inserted,
        "updated": updated,
        "unchanged": unchanged,
        "entries": entries,
    }


def iter_cached_entries(cache_dir: str | Path) -> list[CacheEntry]:
    """Return all cache entries found under ``cache_dir``."""

    cache_path = Path(cache_dir)
    if not cache_path.exists():
        return []

    entries: list[CacheEntry] = []
    for file_path in sorted(cache_path.glob("*.json")):
        data = _load_json(file_path)
        if not isinstance(data, Mapping):
            continue
        payload = data.get("payload")
        if not isinstance(payload, Mapping):
            continue
        identifier = _extract_identifier(data, id_field="igdb_id")
        if identifier is None:
            identifier = _extract_identifier(payload) or file_path.stem
        cached_at = data.get("cached_at")
        entries.append(
            CacheEntry(
                igdb_id=str(identifier),
                payload=dict(payload),
                cached_at=str(cached_at) if cached_at is not None else None,
                path=file_path,
            )
        )
    return entries


def load_cached_entry(cache_dir: str | Path, igdb_id: Any) -> CacheEntry | None:
    """Load a single cached payload by ``igdb_id``."""

    identifier = _extract_identifier({"id": igdb_id})
    if identifier is None:
        return None
    file_path = Path(cache_dir) / f"{_sanitize_identifier(identifier)}.json"
    if not file_path.exists():
        return None
    data = _load_json(file_path)
    if not isinstance(data, Mapping):
        return None
    payload = data.get("payload")
    if not isinstance(payload, Mapping):
        return None
    cached_at = data.get("cached_at")
    return CacheEntry(
        igdb_id=identifier,
        payload=dict(payload),
        cached_at=str(cached_at) if cached_at is not None else None,
        path=file_path,
    )


def fetch_cached_updates(
    cache_dir: str | Path,
    new_payloads: Mapping[str, Mapping[str, Any]] | Iterable[Mapping[str, Any]] | None = None,
    *,
    diff_builder: Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Return cached entries enriched with diff information.

    Parameters
    ----------
    cache_dir:
        Directory containing cached IGDB payload JSON files.
    new_payloads:
        Optional mapping or iterable of freshly downloaded payloads.  If
        provided each cached entry will be compared against the matching
        payload using ``diff_builder`` and the resulting diff will be
        included in the returned structure.
    diff_builder:
        Callable responsible for building the diff report.  When omitted
        :func:`igdb.diff.build_diff_report` is used.
    """

    entries = iter_cached_entries(cache_dir)
    if not entries:
        return []

    payload_map: dict[str, Mapping[str, Any]] | None = None
    if new_payloads is not None:
        payload_map = {}
        if isinstance(new_payloads, Mapping):
            sources = new_payloads.items()
        else:
            sources = (
                (None, payload)
                for payload in new_payloads
                if isinstance(payload, Mapping)
            )
        for key, payload in sources:
            if not isinstance(payload, Mapping):
                continue
            identifier = _extract_identifier(payload)
            if identifier is None:
                continue
            payload_map[str(identifier)] = payload
            if identifier.isdigit():
                payload_map.setdefault(str(int(identifier)), payload)
            if key is not None:
                payload_map[str(key)] = payload

    builder = diff_builder or diff_utils.build_diff_report

    results: list[dict[str, Any]] = []
    for entry in entries:
        result: dict[str, Any] = {
            "igdb_id": entry.igdb_id,
            "cached_at": entry.cached_at,
            "payload": entry.payload,
        }
        if payload_map is not None:
            candidate = payload_map.get(entry.igdb_id)
            if candidate is None and entry.igdb_id.isdigit():
                candidate = payload_map.get(str(int(entry.igdb_id)))
            if candidate is None:
                candidate = payload_map.get(entry.igdb_id.lstrip("0"))
            if candidate is not None:
                diff = builder(entry.payload, candidate)
                result["diff"] = diff
                result["has_diff"] = bool(diff)
                result["remote_payload"] = candidate
        results.append(result)
    return sorted(results, key=lambda item: item["igdb_id"])


def _extract_identifier(
    payload: Mapping[str, Any], *, id_field: str = "id"
) -> str | None:
    for key in (id_field, "igdb_id", "id"):
        if key not in payload:
            continue
        value = payload.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            text = value.strip()
            if text:
                return text
            continue
        if isinstance(value, (int, float)):
            try:
                return str(int(value))
            except Exception:
                return str(value)
        text = str(value).strip()
        if text:
            return text
    return None


def _sanitize_identifier(identifier: str) -> str:
    sanitized = _SAFE_FILENAME_RE.sub("_", identifier.strip())
    return sanitized or "cache_entry"


def _payloads_equal(
    existing: Mapping[str, Any], new_payload: Mapping[str, Any]
) -> bool:
    try:
        return json.dumps(existing, sort_keys=True) == json.dumps(
            new_payload, sort_keys=True
        )
    except TypeError:
        try:
            return existing == new_payload
        except Exception:  # pragma: no cover - defensive path
            return False


def _resolve_timestamp(
    timestamp_factory: Callable[[], str] | None,
) -> str:
    if timestamp_factory is not None:
        value = timestamp_factory()
        if isinstance(value, str):
            return value
        if isinstance(value, datetime):
            if value.tzinfo is None:
                value = value.replace(tzinfo=timezone.utc)
            return value.astimezone(timezone.utc).isoformat()
        return str(value)
    now = datetime.now(timezone.utc)
    return now.isoformat()


def _load_json(path: Path) -> Mapping[str, Any] | None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        logger.exception("Unable to read cache file: %s", path)
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.exception("Invalid JSON in cache file: %s", path)
        return None
    if isinstance(data, Mapping):
        return data
    logger.warning("Ignoring cache file with unexpected payload: %s", path)
    return None


def _dump_json(path: Path, data: Mapping[str, Any]) -> None:
    try:
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    except OSError:
        logger.exception("Unable to write cache file: %s", path)
