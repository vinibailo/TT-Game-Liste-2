"""IGDB client and external API integration helpers."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Callable, Iterable, Mapping

from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


logger = logging.getLogger(__name__)


def exchange_twitch_credentials(
    client_id: str | None = None, client_secret: str | None = None
) -> tuple[str, str]:
    """Return a Twitch access token paired with the resolved client id."""

    resolved_client_id = client_id or os.environ.get('TWITCH_CLIENT_ID')
    resolved_client_secret = client_secret or os.environ.get('TWITCH_CLIENT_SECRET')
    if not resolved_client_id or not resolved_client_secret:
        raise RuntimeError('missing twitch client credentials')

    payload = urlencode(
        {
            'client_id': resolved_client_id,
            'client_secret': resolved_client_secret,
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

    token = data.get('access_token') if isinstance(data, Mapping) else None
    if not token:
        raise RuntimeError('missing access token in twitch response')
    return str(token), resolved_client_id


def download_igdb_metadata(
    access_token: str,
    client_id: str,
    igdb_ids: Iterable[str],
    *,
    batch_size: int,
    user_agent: str,
    normalize: Callable[[Mapping[str, Any]], dict[str, Any] | None],
    coerce_id: Callable[[Any], str | None],
    request_factory: Callable[..., Any] | None = None,
    opener: Callable[[Any], Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Fetch detailed IGDB metadata for a collection of ids."""

    numeric_ids: list[int] = []
    seen: set[int] = set()
    for value in igdb_ids:
        normalized = coerce_id(value)
        if not normalized:
            continue
        try:
            numeric = int(normalized)
        except (TypeError, ValueError):
            logger.warning('Skipping invalid IGDB id %s', value)
            continue
        if numeric in seen:
            continue
        seen.add(numeric)
        numeric_ids.append(numeric)

    if not numeric_ids:
        return {}

    results: dict[str, dict[str, Any]] = {}
    chunk_size = batch_size if batch_size > 0 else 500
    for start in range(0, len(numeric_ids), chunk_size):
        chunk = numeric_ids[start : start + chunk_size]
        if not chunk:
            continue
        query = (
            'fields '
            'id,name,summary,updated_at,first_release_date,'
            'genres.name,platforms.name,game_modes.name,category,'
            'involved_companies.company.name,'
            'involved_companies.developer,'
            'involved_companies.publisher,'
            'cover.image_id,total_rating_count,rating_count; '
            f"where id = ({', '.join(str(v) for v in chunk)}); "
            f'limit {len(chunk)};'
        )
        build_request = request_factory or Request
        open_request = opener or urlopen

        request = build_request(
            'https://api.igdb.com/v4/games',
            data=query.encode('utf-8'),
            method='POST',
        )
        request.add_header('Client-ID', client_id)
        request.add_header('Authorization', f'Bearer {access_token}')
        request.add_header('Accept', 'application/json')
        request.add_header('User-Agent', user_agent)

        try:
            with open_request(request) as response:
                payload = json.loads(response.read().decode('utf-8'))
        except HTTPError as exc:
            message = _format_http_error('IGDB request failed', exc)
            raise RuntimeError(message) from exc
        except Exception as exc:  # pragma: no cover - network failures surfaced
            logger.warning('Failed to query IGDB: %s', exc)
            return {}

        for item in payload or []:
            normalized_item = normalize(item) if isinstance(item, Mapping) else None
            if not normalized_item:
                continue
            results[str(normalized_item['id'])] = normalized_item
    return results


def download_igdb_game_count(
    access_token: str,
    client_id: str,
    *,
    user_agent: str,
    request_factory: Callable[..., Any] | None = None,
    opener: Callable[[Any], Any] | None = None,
) -> int:
    """Return the total number of IGDB game records."""

    build_request = request_factory or Request
    open_request = opener or urlopen

    request = build_request(
        'https://api.igdb.com/v4/games/count',
        data='where id != null;'.encode('utf-8'),
        method='POST',
    )
    request.add_header('Client-ID', client_id)
    request.add_header('Authorization', f'Bearer {access_token}')
    request.add_header('Accept', 'application/json')
    request.add_header('User-Agent', user_agent)

    try:
        with open_request(request) as response:
            payload = json.loads(response.read().decode('utf-8'))
    except HTTPError as exc:
        message = _format_http_error('IGDB count request failed', exc)
        raise RuntimeError(message) from exc
    except Exception as exc:  # pragma: no cover - network failures surfaced
        raise RuntimeError(f'failed to query IGDB count: {exc}') from exc

    if isinstance(payload, Mapping):
        count_value = payload.get('count')
    elif isinstance(payload, list) and payload:
        first = payload[0]
        count_value = first.get('count') if isinstance(first, Mapping) else None
    else:
        count_value = None

    try:
        return int(count_value)
    except (TypeError, ValueError):
        raise RuntimeError('invalid count payload from IGDB')


def download_igdb_games(
    access_token: str,
    client_id: str,
    offset: int,
    limit: int,
    *,
    user_agent: str,
    normalize: Callable[[Mapping[str, Any]], dict[str, Any] | None],
    request_factory: Callable[..., Any] | None = None,
    opener: Callable[[Any], Any] | None = None,
) -> list[dict[str, Any]]:
    """Return a normalized page of IGDB games."""

    if limit <= 0:
        return []

    query = (
        'fields '
        'id,name,summary,updated_at,first_release_date,'
        'genres.name,platforms.name,game_modes.name,category,'
        'involved_companies.company.name,'
        'involved_companies.developer,'
        'involved_companies.publisher,'
        'cover.image_id,total_rating_count,rating_count; '
        f'limit {limit}; '
        f'offset {offset}; '
        'sort id asc;'
    )
    build_request = request_factory or Request
    open_request = opener or urlopen

    request = build_request(
        'https://api.igdb.com/v4/games',
        data=query.encode('utf-8'),
        method='POST',
    )
    request.add_header('Client-ID', client_id)
    request.add_header('Authorization', f'Bearer {access_token}')
    request.add_header('Accept', 'application/json')
    request.add_header('User-Agent', user_agent)

    try:
        with open_request(request) as response:
            payload = json.loads(response.read().decode('utf-8'))
    except HTTPError as exc:
        message = _format_http_error('IGDB request failed', exc)
        raise RuntimeError(message) from exc
    except Exception as exc:  # pragma: no cover - network failures surfaced
        raise RuntimeError(f'failed to query IGDB games: {exc}') from exc

    results: list[dict[str, Any]] = []
    for item in payload or []:
        normalized_item = normalize(item) if isinstance(item, Mapping) else None
        if normalized_item is not None:
            results.append(normalized_item)
    return results


def _format_http_error(prefix: str, error: HTTPError) -> str:
    message = f'{prefix}: {error.code}'
    error_message = ''
    try:
        error_body = error.read()
    except Exception:  # pragma: no cover - best effort to capture error body
        error_body = b''
    if error_body:
        try:
            error_message = error_body.decode('utf-8', errors='replace').strip()
        except Exception:  # pragma: no cover - unexpected decoding failures
            error_message = ''
    if not error_message and error.reason:
        error_message = str(error.reason)
    if error_message:
        message = f"{message} {error_message}"
    return message

