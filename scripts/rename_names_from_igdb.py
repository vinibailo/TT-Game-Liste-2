#!/usr/bin/env python3
"""Rename processed game rows using the live IGDB catalogue."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from typing import Any, Callable, Iterable, Mapping

from app import (
    coerce_igdb_id,
    db_lock,
    exchange_twitch_credentials,
    fetch_igdb_metadata,
    get_db,
)


def _normalize_text(value: Any) -> str:
    """Return a stripped string representation for comparison and storage."""

    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return text


def _load_processed_rows(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Load all processed game rows with their identifiers and names."""

    with db_lock:
        cursor = conn.execute(
            'SELECT "ID", "Source Index", "igdb_id", "Name" FROM processed_games '
            'ORDER BY "ID"'
        )
        return cursor.fetchall()


def rename_processed_games_from_igdb(
    *,
    conn: sqlite3.Connection | None = None,
    exchange_credentials: Callable[[], tuple[str, str]] | None = None,
    metadata_loader: Callable[[str, str, Iterable[str]], Mapping[str, Mapping[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Update the ``Name`` column for each processed game using IGDB data."""

    if exchange_credentials is None:
        exchange_credentials = exchange_twitch_credentials
    if metadata_loader is None:
        metadata_loader = fetch_igdb_metadata

    owns_connection = False
    if conn is None:
        conn = get_db()
        owns_connection = True

    try:
        rows = _load_processed_rows(conn)
        total_rows = len(rows)
        rows_with_igdb_id: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        unique_ids: list[str] = []
        missing_id_count = 0

        for row in rows:
            raw_igdb = row["igdb_id"]
            igdb_id = coerce_igdb_id(raw_igdb)
            normalized_name = _normalize_text(row["Name"])
            source_index = _normalize_text(row["Source Index"])
            entry = {
                "db_id": row["ID"],
                "source_index": source_index,
                "igdb_id": igdb_id,
                "current_name": normalized_name,
            }
            if igdb_id:
                rows_with_igdb_id.append(entry)
                if igdb_id not in seen_ids:
                    seen_ids.add(igdb_id)
                    unique_ids.append(igdb_id)
            else:
                missing_id_count += 1

        if not rows_with_igdb_id:
            return {
                "total_rows": total_rows,
                "rows_with_igdb_id": 0,
                "missing_id": missing_id_count,
                "renamed_rows": [],
                "missing_remote": [],
                "missing_name": [],
                "unchanged": 0,
                "updated": 0,
            }

        access_token, client_id = exchange_credentials()
        metadata = metadata_loader(access_token, client_id, unique_ids) or {}

        renamed_rows: list[dict[str, Any]] = []
        missing_remote_ids: set[str] = set()
        missing_name_ids: set[str] = set()
        unchanged_count = 0

        for entry in rows_with_igdb_id:
            igdb_id = entry["igdb_id"]
            payload = metadata.get(igdb_id)
            if not isinstance(payload, Mapping):
                missing_remote_ids.add(igdb_id)
                continue
            remote_name = _normalize_text(payload.get("name"))
            if not remote_name:
                missing_name_ids.add(igdb_id)
                continue
            if remote_name == entry["current_name"]:
                unchanged_count += 1
                continue
            db_id = entry["db_id"]
            if db_id is None:
                missing_remote_ids.add(igdb_id)
                continue
            renamed_rows.append(
                {
                    "id": db_id,
                    "source_index": entry["source_index"],
                    "igdb_id": igdb_id,
                    "old_name": entry["current_name"],
                    "new_name": remote_name,
                }
            )

        if renamed_rows:
            updates = [(row["new_name"], row["id"]) for row in renamed_rows]
            with db_lock:
                with conn:
                    conn.executemany(
                        'UPDATE processed_games SET "Name"=? WHERE "ID"=?', updates
                    )

        return {
            "total_rows": total_rows,
            "rows_with_igdb_id": len(rows_with_igdb_id),
            "missing_id": missing_id_count,
            "renamed_rows": renamed_rows,
            "missing_remote": sorted(missing_remote_ids),
            "missing_name": sorted(missing_name_ids),
            "unchanged": unchanged_count,
            "updated": len(renamed_rows),
        }
    finally:
        if owns_connection and conn is not None:
            conn.close()


def _format_row_label(row: Mapping[str, Any]) -> str:
    source_index = _normalize_text(row.get("source_index"))
    if source_index:
        return f"Source {source_index}"
    return f"ID {row.get('id')}"


def _format_name(value: str) -> str:
    return value if value else "(empty)"


def main() -> None:
    try:
        summary = rename_processed_games_from_igdb()
    except Exception as exc:  # pragma: no cover - surface unexpected failures
        print(f"Failed to rename processed games: {exc}")
        raise SystemExit(1)

    renamed_rows = summary["renamed_rows"]
    if renamed_rows:
        print("Updated names:")
        for row in renamed_rows:
            label = _format_row_label(row)
            print(
                f"  {label} (IGDB {row['igdb_id']}): "
                f"{_format_name(row['old_name'])} -> {_format_name(row['new_name'])}"
            )
    else:
        print("No names required updating.")

    if summary["missing_remote"]:
        print(
            "Missing IGDB records for IDs: "
            + ", ".join(summary["missing_remote"])
        )
    if summary["missing_name"]:
        print(
            "IGDB responses without names for IDs: "
            + ", ".join(summary["missing_name"])
        )

    print(
        "Processed {total} rows (with IGDB ID: {with_id}, updated: {updated}, "
        "unchanged: {unchanged}, without IGDB ID: {missing_id}).".format(
            total=summary["total_rows"],
            with_id=summary["rows_with_igdb_id"],
            updated=summary["updated"],
            unchanged=summary["unchanged"],
            missing_id=summary["missing_id"],
        )
    )


if __name__ == "__main__":  # pragma: no cover - manual execution path
    main()
