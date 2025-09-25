"""IGDB client and external API integration helpers."""

from __future__ import annotations

import json
import logging
import numbers
import os
from typing import Any, Callable, Iterable, Mapping

from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from helpers import _normalize_lookup_name


logger = logging.getLogger(__name__)


__all__ = [
    "cover_url_from_cover",
    "coerce_igdb_id",
    "extract_igdb_id",
    "exchange_twitch_credentials",
    "download_igdb_metadata",
    "download_igdb_game_count",
    "download_igdb_games",
    "map_igdb_genres",
    "map_igdb_modes",
    "resolve_igdb_page_size",
]


def resolve_igdb_page_size(batch_size: Any, *, max_page_size: int = 500) -> int:
    """Return a sanitized IGDB page size respecting API constraints."""

    try:
        size = int(batch_size)
    except (TypeError, ValueError):
        return max_page_size
    if size <= 0:
        return max_page_size
    return min(size, max_page_size)


def cover_url_from_cover(value: Any, size: str = "t_cover_big") -> str:
    """Return the IGDB image URL for a cover payload or identifier."""

    image_id: str | None = None
    if isinstance(value, Mapping):
        raw_id = value.get("image_id")
        if isinstance(raw_id, str):
            image_id = raw_id.strip()
        elif raw_id is not None:
            image_id = str(raw_id).strip()
    elif isinstance(value, str):
        image_id = value.strip()
    elif value is not None:
        image_id = str(value).strip()
    if not image_id:
        return ""
    size_key = str(size).strip() if size else "t_cover_big"
    if not size_key:
        size_key = "t_cover_big"
    return "https://images.igdb.com/igdb/image/upload/" f"{size_key}/{image_id}.jpg"


def coerce_igdb_id(value: Any) -> str:
    """Normalize potential IGDB identifiers to a canonical string."""

    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
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
        if float(value).is_integer():
            return str(int(value))
        return str(value)
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    return text


def extract_igdb_id(row: pd.Series, allow_generic_id: bool = False) -> str:
    """Extract an IGDB identifier from a row, optionally falling back to ``id``."""

    for key in row.index:
        normalized = _normalize_column_name(key)
        if "igdb" in normalized and "id" in normalized:
            value = coerce_igdb_id(row.get(key))
            if value:
                return value
    if allow_generic_id:
        for key in row.index:
            key_str = str(key)
            if key_str.lower() == "id":
                value = coerce_igdb_id(row.get(key))
                if value:
                    return value
    return ""


def _normalize_column_name(name: str) -> str:
    return "".join(ch.lower() for ch in str(name) if ch.isalnum())


def map_igdb_genres(names: Iterable[str]) -> list[str]:
    """Return localized genre names derived from IGDB values."""

    return _map_igdb_values(names, IGDB_GENRE_TRANSLATIONS)


def map_igdb_modes(names: Iterable[str]) -> list[str]:
    """Return localized game mode names derived from IGDB values."""

    return _map_igdb_values(names, IGDB_MODE_TRANSLATIONS)


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


def _normalize_translation_key(value: str) -> str:
    key = str(value).strip().casefold()
    for old, new in (
        ("&", " and "),
        ("/", " "),
        ("-", " "),
        ("_", " "),
        ("'", ""),
        (",", " "),
        (".", " "),
        ("+", " "),
    ):
        key = key.replace(old, new)
    for char in "()[]{}":
        key = key.replace(char, " ")
    key = "".join(ch for ch in key if ch.isalnum() or ch.isspace())
    return " ".join(key.split())


IGDB_GENRE_TRANSLATIONS: dict[str, tuple[str, ...]] = {
    _normalize_translation_key("Action"): ("Ação e Aventura",),
    _normalize_translation_key("Action Adventure"): ("Ação e Aventura",),
    _normalize_translation_key("Adventure"): ("Ação e Aventura",),
    _normalize_translation_key("Point-and-click"): ("Ação e Aventura",),
    _normalize_translation_key("Stealth"): ("Ação e Aventura",),
    _normalize_translation_key("Survival"): ("Ação e Aventura",),
    _normalize_translation_key("Platform"): ("Plataformas",),
    _normalize_translation_key("Platformer"): ("Plataformas",),
    _normalize_translation_key("Shooter"): ("Tiro",),
    _normalize_translation_key("Shoot 'em up"): ("Tiro",),
    _normalize_translation_key("Fighting"): ("Luta",),
    _normalize_translation_key("Hack and slash/Beat 'em up"): ("Luta",),
    _normalize_translation_key("Brawler"): ("Luta",),
    _normalize_translation_key("Racing"): ("Corrida e Voo",),
    _normalize_translation_key("Driving Racing"): ("Corrida e Voo",),
    _normalize_translation_key("Flight"): ("Corrida e Voo",),
    _normalize_translation_key("Simulator"): ("Simulação",),
    _normalize_translation_key("Simulation"): ("Simulação",),
    _normalize_translation_key("Strategy"): ("Estratégia",),
    _normalize_translation_key("Real Time Strategy (RTS)"): ("Estratégia",),
    _normalize_translation_key("Turn-based strategy (TBS)"): ("Estratégia",),
    _normalize_translation_key("Tactical"): ("Estratégia",),
    _normalize_translation_key("MOBA"): ("Multijogador",),
    _normalize_translation_key("Massively Multiplayer Online (MMO)"): ("Multijogador",),
    _normalize_translation_key("Battle Royale"): ("Multijogador",),
    _normalize_translation_key("MMORPG"): ("RPG", "Multijogador"),
    _normalize_translation_key("Role-playing (RPG)"): ("RPG",),
    _normalize_translation_key("Role playing"): ("RPG",),
    _normalize_translation_key("Roguelike"): ("RPG",),
    _normalize_translation_key("Roguelite"): ("RPG",),
    _normalize_translation_key("Puzzle"): ("Quebra-cabeça e Trivia",),
    _normalize_translation_key("Quiz/Trivia"): ("Quebra-cabeça e Trivia",),
    _normalize_translation_key("Trivia"): ("Quebra-cabeça e Trivia",),
    _normalize_translation_key("Card & Board Game"): ("Cartas e Tabuleiro",),
    _normalize_translation_key("Board game"): ("Cartas e Tabuleiro",),
    _normalize_translation_key("Tabletop"): ("Cartas e Tabuleiro",),
    _normalize_translation_key("Family"): ("Família e Crianças",),
    _normalize_translation_key("Kids"): ("Família e Crianças",),
    _normalize_translation_key("Educational"): ("Família e Crianças",),
    _normalize_translation_key("Party"): ("Família e Crianças",),
    _normalize_translation_key("Music"): ("Família e Crianças",),
    _normalize_translation_key("Indie"): ("Indie",),
    _normalize_translation_key("Arcade"): ("Clássicos",),
    _normalize_translation_key("Pinball"): ("Clássicos",),
    _normalize_translation_key("Classic"): ("Clássicos",),
    _normalize_translation_key("Visual Novel"): ("Visual Novel",),
    _normalize_translation_key("ação e aventura"): ("Ação e Aventura",),
    _normalize_translation_key("plataformas"): ("Plataformas",),
    _normalize_translation_key("tiro"): ("Tiro",),
    _normalize_translation_key("luta"): ("Luta",),
    _normalize_translation_key("corrida e voo"): ("Corrida e Voo",),
    _normalize_translation_key("simulação"): ("Simulação",),
    _normalize_translation_key("estratégia"): ("Estratégia",),
    _normalize_translation_key("multijogador"): ("Multijogador",),
    _normalize_translation_key("rpg"): ("RPG",),
    _normalize_translation_key("quebra-cabeça e trivia"): ("Quebra-cabeça e Trivia",),
    _normalize_translation_key("cartas e tabuleiro"): ("Cartas e Tabuleiro",),
    _normalize_translation_key("família e crianças"): ("Família e Crianças",),
    _normalize_translation_key("indie"): ("Indie",),
    _normalize_translation_key("clássicos"): ("Clássicos",),
    _normalize_translation_key("visual novel"): ("Visual Novel",),
}


IGDB_MODE_TRANSLATIONS: dict[str, tuple[str, ...]] = {
    _normalize_translation_key("Single player"): ("Single-player",),
    _normalize_translation_key("Single-player"): ("Single-player",),
    _normalize_translation_key("Singleplayer"): ("Single-player",),
    _normalize_translation_key("Solo"): ("Single-player",),
    _normalize_translation_key("Campaign"): ("Single-player",),
    _normalize_translation_key("Co-operative"): ("Cooperativo (Co-op)",),
    _normalize_translation_key("Cooperative"): ("Cooperativo (Co-op)",),
    _normalize_translation_key("Co-op"): ("Cooperativo (Co-op)",),
    _normalize_translation_key("Co op"): ("Cooperativo (Co-op)",),
    _normalize_translation_key("Local co-op"): (
        "Cooperativo (Co-op)",
        "Multiplayer local",
    ),
    _normalize_translation_key("Offline co-op"): (
        "Cooperativo (Co-op)",
        "Multiplayer local",
    ),
    _normalize_translation_key("Online co-op"): (
        "Cooperativo (Co-op)",
        "Multiplayer online",
    ),
    _normalize_translation_key("Co-op campaign"): ("Cooperativo (Co-op)",),
    _normalize_translation_key("Multiplayer"): ("Multiplayer online",),
    _normalize_translation_key("Online multiplayer"): ("Multiplayer online",),
    _normalize_translation_key("Multiplayer online"): ("Multiplayer online",),
    _normalize_translation_key("Offline multiplayer"): ("Multiplayer local",),
    _normalize_translation_key("Local multiplayer"): ("Multiplayer local",),
    _normalize_translation_key("Split screen"): ("Multiplayer local",),
    _normalize_translation_key("Shared/Split screen"): ("Multiplayer local",),
    _normalize_translation_key("PvP"): ("Competitivo (PvP)",),
    _normalize_translation_key("Player vs Player"): ("Competitivo (PvP)",),
    _normalize_translation_key("Versus"): ("Competitivo (PvP)",),
    _normalize_translation_key("Competitive"): ("Competitivo (PvP)",),
    _normalize_translation_key("Battle Royale"): (
        "Multiplayer online",
        "Competitivo (PvP)",
    ),
    _normalize_translation_key("Massively Multiplayer Online (MMO)"): (
        "Multiplayer online",
    ),
    _normalize_translation_key("MMO"): ("Multiplayer online",),
    _normalize_translation_key("MMORPG"): (
        "RPG",
        "Multiplayer online",
    ),
    _normalize_translation_key("single-player"): ("Single-player",),
    _normalize_translation_key("cooperativo (co-op)"): ("Cooperativo (Co-op)",),
    _normalize_translation_key("multiplayer local"): ("Multiplayer local",),
    _normalize_translation_key("multiplayer online"): ("Multiplayer online",),
    _normalize_translation_key("competitivo (pvp)"): ("Competitivo (PvP)",),
}


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

