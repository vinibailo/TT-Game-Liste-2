#!/usr/bin/env python3
"""Rename processed game rows using the live IGDB catalogue."""

from __future__ import annotations

import sys
from pathlib import Path
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from typing import Any, Callable

from sqlalchemy import text
from sqlalchemy.engine import Connection

from config import (
    DB_CONNECT_TIMEOUT_SECONDS,
    DB_DSN,
    IGDB_BATCH_SIZE,
    IGDB_USER_AGENT,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from db import utils as db_utils
try:
    from app import (
        coerce_igdb_id,
        db_lock,
        exchange_twitch_credentials,
        fetch_igdb_metadata,
        get_db_sa_connection,
    )
except ModuleNotFoundError:
    import json
    import logging
    import numbers
    import os
    from math import isnan
    from threading import Lock
    from urllib.error import HTTPError
    from urllib.parse import urlencode
    from urllib.request import Request, urlopen

    logger = logging.getLogger(__name__)

    db_lock = Lock()

    try:
        engine = db_utils.build_engine_from_dsn(
            DB_DSN,
            timeout=DB_CONNECT_TIMEOUT_SECONDS,
            pool_size=1,
            pool_recycle=1_800,
            pool_pre_ping=True,
        )
    except ValueError as exc:  # pragma: no cover - defensive for unsupported DSNs
        raise RuntimeError(f"Unsupported database DSN: {DB_DSN}") from exc

    db_handle = db_utils.DatabaseHandle(engine)
    db_utils.set_fallback_connection(db_handle)

    def get_db() -> db_utils.DatabaseHandle:
        return db_handle

    def get_db_sa_connection():
        return db_handle.sa_connection()

    def _is_nan(value: Any) -> bool:
        try:
            return isnan(float(value))
        except (TypeError, ValueError):
            return False

    def coerce_igdb_id(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            text = value.strip()
            if not text or text.lower() == "nan":
                return ""
            if text.endswith(".0") and text[:-2].isdigit():
                return text[:-2]
            return text
        if isinstance(value, numbers.Integral):
            return str(int(value))
        if isinstance(value, numbers.Real):
            if _is_nan(value):
                return ""
            numeric = float(value)
            if numeric.is_integer():
                return str(int(numeric))
            return str(value)
        text = str(value).strip()
        if not text or text.lower() == "nan":
            return ""
        return text

    def exchange_twitch_credentials() -> tuple[str, str]:
        client_id = os.environ.get("TWITCH_CLIENT_ID")
        client_secret = os.environ.get("TWITCH_CLIENT_SECRET")
        if not client_id or not client_secret:
            raise RuntimeError("missing twitch client credentials")

        payload = urlencode(
            {
                "client_id": client_id,
                "client_secret": client_secret,
                "grant_type": "client_credentials",
            }
        ).encode("utf-8")

        request = Request(
            "https://id.twitch.tv/oauth2/token",
            data=payload,
            method="POST",
        )
        request.add_header("Content-Type", "application/x-www-form-urlencoded")

        try:
            with urlopen(request) as response:
                data = json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # pragma: no cover - network failures surfaced
            raise RuntimeError(f"failed to obtain twitch token: {exc}") from exc

        token = data.get("access_token")
        if not token:
            raise RuntimeError("missing access token in twitch response")
        return token, client_id

    def _parse_iterable(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [v.strip() for v in value.split(",") if v.strip()]
        if isinstance(value, numbers.Number):
            if _is_nan(value):
                return []
            return [str(value)]
        items: list[str] = []
        try:
            iterator = iter(value)
        except TypeError:
            text = str(value).strip()
            return [text] if text else []
        for element in iterator:
            if isinstance(element, Mapping):
                name_value = element.get("name")
                if isinstance(name_value, str):
                    stripped = name_value.strip()
                    if stripped:
                        items.append(stripped)
                        continue
                items.append(str(element).strip())
            else:
                candidate = str(element).strip()
                if candidate:
                    items.append(candidate)
        return items

    def fetch_igdb_metadata(
        access_token: str, client_id: str, igdb_ids: Iterable[str]
    ) -> dict[str, dict[str, Any]]:
        if not igdb_ids:
            return {}

        numeric_ids: list[int] = []
        for value in igdb_ids:
            try:
                numeric_ids.append(int(str(value).strip()))
            except (TypeError, ValueError):
                logger.warning("Skipping invalid IGDB id %s", value)
        if not numeric_ids:
            return {}

        results: dict[str, dict[str, Any]] = {}
        batch_size = IGDB_BATCH_SIZE if isinstance(IGDB_BATCH_SIZE, int) and IGDB_BATCH_SIZE > 0 else 500
        for start in range(0, len(numeric_ids), batch_size):
            chunk = numeric_ids[start : start + batch_size]
            if not chunk:
                continue
            query = (
                "fields "
                "id,name,summary,updated_at,first_release_date,"
                "genres.name,platforms.name,game_modes.name,category,"
                "involved_companies.company.name,"
                "involved_companies.developer,"
                "involved_companies.publisher; "
                f"where id = ({', '.join(str(v) for v in chunk)}); "
                f"limit {len(chunk)};"
            )
            request = Request(
                "https://api.igdb.com/v4/games",
                data=query.encode("utf-8"),
                method="POST",
            )
            request.add_header("Client-ID", client_id)
            request.add_header("Authorization", f"Bearer {access_token}")
            request.add_header("Accept", "application/json")
            request.add_header("User-Agent", IGDB_USER_AGENT)

            try:
                with urlopen(request) as response:
                    payload = json.loads(response.read().decode("utf-8"))
            except HTTPError as exc:
                error_message = ""
                try:
                    error_body = exc.read()
                except Exception:  # pragma: no cover - best effort to capture error body
                    error_body = b""
                if error_body:
                    try:
                        error_message = error_body.decode("utf-8", errors="replace").strip()
                    except Exception:  # pragma: no cover - unexpected decoding failures
                        error_message = ""
                if not error_message and getattr(exc, "reason", None):
                    error_message = str(exc.reason)
                message = f"IGDB request failed: {exc.code}"
                if error_message:
                    message = f"{message} {error_message}"
                raise RuntimeError(message) from exc
            except Exception as exc:  # pragma: no cover - network failures surfaced
                logger.warning("Failed to query IGDB: %s", exc)
                return {}

            for item in payload or []:
                if not isinstance(item, dict):
                    continue
                igdb_id = item.get("id")
                if igdb_id is None:
                    continue
                parsed_item = dict(item)
                involved_companies = item.get("involved_companies")
                developer_names: list[str] = []
                publisher_names: list[str] = []
                if isinstance(involved_companies, list):
                    for company in involved_companies:
                        if not isinstance(company, Mapping):
                            continue
                        company_obj = company.get("company")
                        company_name: str | None = None
                        if isinstance(company_obj, Mapping):
                            name_value = company_obj.get("name")
                            if isinstance(name_value, str):
                                stripped = name_value.strip()
                                if stripped:
                                    company_name = stripped
                        elif isinstance(company_obj, str):
                            stripped = company_obj.strip()
                            if stripped:
                                company_name = stripped
                        if not company_name:
                            continue
                        if company.get("developer"):
                            developer_names.append(company_name)
                        if company.get("publisher"):
                            publisher_names.append(company_name)
                parsed_item["developers"] = developer_names
                parsed_item["publishers"] = publisher_names
                parsed_item["genres"] = _parse_iterable(item.get("genres"))
                parsed_item["platforms"] = _parse_iterable(item.get("platforms"))
                parsed_item["game_modes"] = _parse_iterable(item.get("game_modes"))
                results[str(igdb_id)] = parsed_item
        return results


def _normalize_text(value: Any) -> str:
    """Return a stripped string representation for comparison and storage."""

    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return text


def _load_processed_rows(conn: Connection) -> list[dict[str, Any]]:
    """Load all processed game rows with their identifiers and names."""

    processed_games_table = db_utils._quote_identifier("processed_games")
    id_column = db_utils._quote_identifier("ID")
    source_index_column = db_utils._quote_identifier("Source Index")
    igdb_id_column = db_utils._quote_identifier("igdb_id")
    name_column = db_utils._quote_identifier("Name")

    select_sql = text(
        f"SELECT {id_column} AS db_id, "
        f"{source_index_column} AS source_index_value, "
        f"{igdb_id_column} AS igdb_id_value, "
        f"{name_column} AS name_value "
        f"FROM {processed_games_table} "
        f"ORDER BY {id_column}"
    )

    with db_lock:
        rows = conn.execute(select_sql).mappings().all()

    return [
        {
            "ID": row.get("db_id"),
            "Source Index": row.get("source_index_value"),
            "igdb_id": row.get("igdb_id_value"),
            "Name": row.get("name_value"),
        }
        for row in rows
    ]


@contextmanager
def _yield_sa_connection(
    conn: Connection | db_utils.DatabaseHandle | None,
):
    """Yield a SQLAlchemy connection from ``conn`` or the configured factory."""

    if conn is None:
        with get_db_sa_connection() as sa_conn:
            yield sa_conn
        return

    if isinstance(conn, Connection):
        yield conn
        return

    if isinstance(conn, db_utils.DatabaseHandle) or hasattr(conn, "sa_connection"):
        context = conn.sa_connection()  # type: ignore[assignment]
        with context as sa_conn:  # type: ignore[assignment]
            yield sa_conn
        return

    raise TypeError(f"Unsupported connection type: {type(conn)!r}")


def rename_processed_games_from_igdb(
    *,
    conn: Connection | db_utils.DatabaseHandle | None = None,
    exchange_credentials: Callable[[], tuple[str, str]] | None = None,
    metadata_loader: Callable[[str, str, Iterable[str]], Mapping[str, Mapping[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Update the ``Name`` column for each processed game using IGDB data."""

    if exchange_credentials is None:
        exchange_credentials = exchange_twitch_credentials
    if metadata_loader is None:
        metadata_loader = fetch_igdb_metadata

    with _yield_sa_connection(conn) as sa_conn:
        rows = _load_processed_rows(sa_conn)
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
            if sa_conn.in_transaction():
                sa_conn.rollback()
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
            updates = [
                {"new_name": row["new_name"], "db_id": row["id"]}
                for row in renamed_rows
            ]
            if updates:
                if sa_conn.in_transaction():
                    sa_conn.rollback()
                processed_games_table = db_utils._quote_identifier("processed_games")
                name_column = db_utils._quote_identifier("Name")
                id_column = db_utils._quote_identifier("ID")
                update_sql = text(
                    f"UPDATE {processed_games_table} "
                    f"SET {name_column} = :new_name "
                    f"WHERE {id_column} = :db_id"
                )
                with db_lock:
                    with sa_conn.begin():
                        sa_conn.execute(update_sql, updates)

        elif sa_conn.in_transaction():
            sa_conn.rollback()

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
