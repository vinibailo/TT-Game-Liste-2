"""IGDB client and external API integration helpers."""

from __future__ import annotations

import json
import logging
import numbers
import os
import time
from typing import Any, Callable, Iterable, Mapping

from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from helpers import (
    _normalize_lookup_name,
    _parse_company_names,
    _parse_iterable,
)

logger = logging.getLogger(__name__)


__all__ = [
    "IGDBClient",
    "IGDB_CATEGORY_LABELS",
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

    return IGDBClient.translate_genres(names)


def map_igdb_modes(names: Iterable[str]) -> list[str]:
    """Return localized game mode names derived from IGDB values."""

    return IGDBClient.translate_modes(names)


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


IGDB_CATEGORY_LABELS: dict[int, str] = {
    0: "Main Game",
    1: "DLC / Add-on",
    2: "Expansion",
    3: "Bundle",
    4: "Standalone Expansion",
    5: "Mod",
    6: "Episode",
    7: "Season",
    8: "Remake",
    9: "Remaster",
    10: "Expanded Game",
    11: "Port",
    12: "Fork",
    13: "Pack",
    14: "Update",
}


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


class IGDBClient:
    """High level helper that manages IGDB authentication and pagination."""

    BASE_URL = "https://api.igdb.com/v4"
    TOKEN_URL = "https://id.twitch.tv/oauth2/token"

    def __init__(
        self,
        *,
        client_id: str | None = None,
        client_secret: str | None = None,
        user_agent: str | None = None,
        max_page_size: int = 500,
        max_retries: int = 3,
        rate_limit_wait: float = 1.0,
        request_factory: Callable[..., Any] | None = None,
        opener: Callable[[Any], Any] | None = None,
        sleep: Callable[[float], None] | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self._client_id = (client_id or "").strip()
        self._client_secret = (client_secret or "").strip()
        self._user_agent = (user_agent or "").strip()
        self._max_page_size = max_page_size if max_page_size > 0 else 500
        self._max_retries = max(1, int(max_retries)) if max_retries else 3
        self._rate_limit_wait = rate_limit_wait if rate_limit_wait and rate_limit_wait > 0 else 1.0
        self._request_factory = request_factory
        self._opener = opener
        self._sleep = sleep or time.sleep
        self._env = env or os.environ

    @property
    def user_agent(self) -> str:
        if self._user_agent:
            return self._user_agent
        env_agent = self._env.get("IGDB_USER_AGENT")
        if env_agent:
            return env_agent.strip()
        return "TT-Game-Liste/1.0 (support@example.com)"

    def exchange_twitch_credentials(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        *,
        request_factory: Callable[..., Any] | None = None,
        opener: Callable[[Any], Any] | None = None,
    ) -> tuple[str, str]:
        """Return a Twitch access token paired with the resolved client id."""

        resolved_client_id = (client_id or self._client_id or self._env.get("TWITCH_CLIENT_ID") or "").strip()
        resolved_client_secret = (
            client_secret
            or self._client_secret
            or self._env.get("TWITCH_CLIENT_SECRET")
            or ""
        ).strip()
        if not resolved_client_id or not resolved_client_secret:
            raise RuntimeError("missing twitch client credentials")

        payload = urlencode(
            {
                "client_id": resolved_client_id,
                "client_secret": resolved_client_secret,
                "grant_type": "client_credentials",
            }
        ).encode("utf-8")

        build_request = self._resolve_request_factory(request_factory)
        open_request = self._resolve_opener(opener)

        request = build_request(
            self.TOKEN_URL,
            data=payload,
            method="POST",
        )
        request.add_header("Content-Type", "application/x-www-form-urlencoded")

        data = self._request_json(
            request,
            open_request,
            error_prefix="failed to obtain twitch token",
            generic_error="failed to obtain twitch token",
            allow_rate_limit=False,
        )

        token = data.get("access_token") if isinstance(data, Mapping) else None
        if not token:
            raise RuntimeError("missing access token in twitch response")
        return str(token), resolved_client_id

    def fetch_game_count(
        self,
        access_token: str,
        client_id: str,
        *,
        user_agent: str | None = None,
        request_factory: Callable[..., Any] | None = None,
        opener: Callable[[Any], Any] | None = None,
    ) -> int:
        """Return the total number of IGDB game records."""

        build_request = self._resolve_request_factory(request_factory)
        open_request = self._resolve_opener(opener)

        request = build_request(
            f"{self.BASE_URL}/games/count",
            data="where id != null;".encode("utf-8"),
            method="POST",
        )
        self._apply_headers(request, client_id, access_token, user_agent)

        payload = self._request_json(
            request,
            open_request,
            error_prefix="IGDB count request failed",
            generic_error="failed to query IGDB count",
        )

        if isinstance(payload, Mapping):
            count_value = payload.get("count")
        elif isinstance(payload, list) and payload:
            first = payload[0]
            count_value = first.get("count") if isinstance(first, Mapping) else None
        else:
            count_value = None

        try:
            return int(count_value)
        except (TypeError, ValueError):
            raise RuntimeError("invalid count payload from IGDB")

    def fetch_games(
        self,
        access_token: str,
        client_id: str,
        offset: int,
        limit: int,
        *,
        user_agent: str | None = None,
        request_factory: Callable[..., Any] | None = None,
        opener: Callable[[Any], Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Return a normalized page of IGDB games."""

        sanitized_limit = resolve_igdb_page_size(limit, max_page_size=self._max_page_size)
        if sanitized_limit <= 0:
            return []

        query = (
            "fields "
            "id,name,summary,updated_at,first_release_date,"
            "genres.name,platforms.name,game_modes.name,category,"
            "involved_companies.company.name,"
            "involved_companies.developer,"
            "involved_companies.publisher,"
            "cover.image_id,total_rating_count,rating_count; "
            f"limit {sanitized_limit}; "
            f"offset {max(0, int(offset))}; "
            "sort id asc;"
        )
        build_request = self._resolve_request_factory(request_factory)
        open_request = self._resolve_opener(opener)

        request = build_request(
            f"{self.BASE_URL}/games",
            data=query.encode("utf-8"),
            method="POST",
        )
        self._apply_headers(request, client_id, access_token, user_agent)

        payload = self._request_json(
            request,
            open_request,
            error_prefix="IGDB request failed",
            generic_error="failed to query IGDB games",
        )

        results: list[dict[str, Any]] = []
        for item in payload or []:
            normalized_item = self.normalize_game(item)
            if normalized_item is not None:
                results.append(normalized_item)
        return results

    def fetch_metadata_by_ids(
        self,
        access_token: str,
        client_id: str,
        igdb_ids: Iterable[str],
        *,
        batch_size: int | None = None,
        user_agent: str | None = None,
        request_factory: Callable[..., Any] | None = None,
        opener: Callable[[Any], Any] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Fetch detailed IGDB metadata for a collection of ids."""

        numeric_ids: list[int] = []
        seen: set[int] = set()
        for value in igdb_ids:
            normalized = coerce_igdb_id(value)
            if not normalized:
                continue
            try:
                numeric = int(normalized)
            except (TypeError, ValueError):
                logger.warning("Skipping invalid IGDB id %s", value)
                continue
            if numeric in seen:
                continue
            seen.add(numeric)
            numeric_ids.append(numeric)

        if not numeric_ids:
            return {}

        chunk_size = resolve_igdb_page_size(
            batch_size if batch_size is not None else self._max_page_size,
            max_page_size=self._max_page_size,
        )
        build_request = self._resolve_request_factory(request_factory)
        open_request = self._resolve_opener(opener)

        results: dict[str, dict[str, Any]] = {}
        for start in range(0, len(numeric_ids), chunk_size):
            chunk = numeric_ids[start : start + chunk_size]
            if not chunk:
                continue
            query = (
                "fields "
                "id,name,summary,updated_at,first_release_date,"
                "genres.name,platforms.name,game_modes.name,category,"
                "involved_companies.company.name,"
                "involved_companies.developer,"
                "involved_companies.publisher,"
                "cover.image_id,total_rating_count,rating_count; "
                f"where id = ({', '.join(str(v) for v in chunk)}); "
                f"limit {len(chunk)};"
            )

            request = build_request(
                f"{self.BASE_URL}/games",
                data=query.encode("utf-8"),
                method="POST",
            )
            self._apply_headers(request, client_id, access_token, user_agent)

            payload = self._request_json(
                request,
                open_request,
                error_prefix="IGDB request failed",
                generic_error="failed to query IGDB",
            )

            for item in payload or []:
                normalized_item = self.normalize_game(item)
                if not normalized_item:
                    continue
                results[str(normalized_item["id"])] = normalized_item
        return results

    def normalize_game(self, item: Mapping[str, Any]) -> dict[str, Any] | None:
        """Return a normalized representation of an IGDB payload."""

        if not isinstance(item, Mapping):
            return None

        raw_id = item.get("id")
        try:
            igdb_id = int(str(raw_id).strip())
        except (TypeError, ValueError):
            logger.warning("Skipping IGDB entry with invalid id %s", raw_id)
            return None

        name_value = item.get("name")
        name = name_value.strip() if isinstance(name_value, str) else ""

        summary_value = item.get("summary")
        summary = summary_value.strip() if isinstance(summary_value, str) else ""

        cover_obj = item.get("cover")
        cover: dict[str, Any] | None = None
        if isinstance(cover_obj, Mapping):
            image_id_value = cover_obj.get("image_id") or cover_obj.get("imageId")
            if image_id_value:
                cover = {"image_id": str(image_id_value)}
        elif isinstance(cover_obj, str):
            image_id = cover_obj.strip()
            if image_id:
                cover = {"image_id": image_id}

        rating_count = self._coerce_rating_count(
            item.get("total_rating_count"), item.get("rating_count")
        )

        developer_names, publisher_names = self._normalize_involved_companies(
            item.get("involved_companies")
        )

        if not developer_names:
            developer_names = _parse_company_names(item.get("developers"))
        if not publisher_names:
            publisher_names = _parse_company_names(item.get("publishers"))

        genres = _parse_iterable(item.get("genres"))
        platforms = _parse_iterable(item.get("platforms"))
        game_modes = _parse_iterable(item.get("game_modes"))

        return {
            "id": igdb_id,
            "name": name,
            "summary": summary,
            "updated_at": item.get("updated_at"),
            "first_release_date": item.get("first_release_date"),
            "category": item.get("category"),
            "cover": cover,
            "rating_count": rating_count,
            "developers": developer_names,
            "publishers": publisher_names,
            "genres": genres,
            "platforms": platforms,
            "game_modes": game_modes,
        }

    @staticmethod
    def _normalize_involved_companies(
        companies: Any,
    ) -> tuple[list[str], list[str]]:
        developer_names: list[str] = []
        publisher_names: list[str] = []
        if isinstance(companies, list):
            seen_dev: set[str] = set()
            seen_pub: set[str] = set()
            for company in companies:
                if not isinstance(company, Mapping):
                    continue
                company_obj = company.get("company")
                company_name: str | None = None
                if isinstance(company_obj, Mapping):
                    name_candidate = company_obj.get("name")
                    if isinstance(name_candidate, str):
                        company_name = name_candidate.strip()
                elif isinstance(company_obj, str):
                    company_name = company_obj.strip()
                if not company_name:
                    name_candidate = company.get("name")
                    if isinstance(name_candidate, str):
                        company_name = name_candidate.strip()
                if not company_name:
                    continue
                fingerprint = company_name.casefold()
                if company.get("developer") and fingerprint not in seen_dev:
                    seen_dev.add(fingerprint)
                    developer_names.append(company_name)
                if company.get("publisher") and fingerprint not in seen_pub:
                    seen_pub.add(fingerprint)
                    publisher_names.append(company_name)
        return developer_names, publisher_names

    @staticmethod
    def _coerce_rating_count(primary: Any, secondary: Any) -> int | None:
        for candidate in (primary, secondary):
            if candidate in (None, ""):
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

    def _apply_headers(
        self,
        request: Any,
        client_id: str,
        access_token: str,
        user_agent: str | None,
    ) -> None:
        request.add_header("Client-ID", client_id)
        request.add_header("Authorization", f"Bearer {access_token}")
        request.add_header("Accept", "application/json")
        request.add_header("User-Agent", (user_agent or self.user_agent).strip())

    def _resolve_request_factory(self, request_factory: Callable[..., Any] | None):
        return request_factory or self._request_factory or Request

    def _resolve_opener(self, opener: Callable[[Any], Any] | None):
        return opener or self._opener or urlopen

    def _request_json(
        self,
        request: Any,
        opener: Callable[[Any], Any],
        *,
        error_prefix: str,
        generic_error: str,
        allow_rate_limit: bool = True,
    ) -> Any:
        attempts = self._max_retries if allow_rate_limit else 1
        for attempt in range(attempts):
            try:
                with opener(request) as response:
                    body = response.read()
            except HTTPError as exc:
                if allow_rate_limit and exc.code == 429 and attempt + 1 < attempts:
                    delay = self._retry_delay(exc)
                    if delay > 0:
                        self._sleep(delay)
                    continue
                message = _format_http_error(error_prefix, exc)
                raise RuntimeError(message) from exc
            except Exception as exc:  # pragma: no cover - network failures surfaced
                raise RuntimeError(f"{generic_error}: {exc}") from exc
            try:
                text = body.decode("utf-8") if body else ""
            except Exception:  # pragma: no cover - unexpected decoding failures
                text = ""
            try:
                return json.loads(text) if text else []
            except Exception as exc:
                raise RuntimeError("invalid JSON response from IGDB") from exc
        return []

    def _retry_delay(self, error: HTTPError) -> float:
        headers = getattr(error, "headers", None)
        if headers is not None:
            for key in ("Retry-After", "retry-after"):
                value = headers.get(key)
                if value:
                    try:
                        delay = float(value)
                        if delay > 0:
                            return delay
                    except (TypeError, ValueError):
                        continue
            for key in ("X-RateLimit-Reset", "x-ratelimit-reset"):
                value = headers.get(key)
                if value:
                    try:
                        reset_timestamp = float(value)
                        delay = reset_timestamp - time.time()
                        if delay > 0:
                            return delay
                    except (TypeError, ValueError):
                        continue
        return self._rate_limit_wait

    @classmethod
    def translate_category(cls, value: Any) -> str:
        if value in (None, ""):
            return ""
        try:
            key = int(value)
        except (TypeError, ValueError):
            return str(value).strip()
        return IGDB_CATEGORY_LABELS.get(key, "Other")

    @classmethod
    def translate_genres(cls, names: Iterable[str]) -> list[str]:
        return _map_igdb_values(names, IGDB_GENRE_TRANSLATIONS)

    @classmethod
    def translate_modes(cls, names: Iterable[str]) -> list[str]:
        return _map_igdb_values(names, IGDB_MODE_TRANSLATIONS)


def exchange_twitch_credentials(
    client_id: str | None = None,
    client_secret: str | None = None,
    *,
    request_factory: Callable[..., Any] | None = None,
    opener: Callable[[Any], Any] | None = None,
) -> tuple[str, str]:
    """Compatibility wrapper that proxies to :class:`IGDBClient`."""

    client = IGDBClient(client_id=client_id, client_secret=client_secret)
    return client.exchange_twitch_credentials(
        client_id=client_id,
        client_secret=client_secret,
        request_factory=request_factory,
        opener=opener,
    )


def download_igdb_game_count(
    access_token: str,
    client_id: str,
    *,
    user_agent: str,
    request_factory: Callable[..., Any] | None = None,
    opener: Callable[[Any], Any] | None = None,
) -> int:
    client = IGDBClient(user_agent=user_agent, request_factory=request_factory, opener=opener)
    return client.fetch_game_count(
        access_token,
        client_id,
        user_agent=user_agent,
        request_factory=request_factory,
        opener=opener,
    )


def download_igdb_games(
    access_token: str,
    client_id: str,
    offset: int,
    limit: int,
    *,
    user_agent: str,
    request_factory: Callable[..., Any] | None = None,
    opener: Callable[[Any], Any] | None = None,
) -> list[dict[str, Any]]:
    client = IGDBClient(user_agent=user_agent, request_factory=request_factory, opener=opener)
    return client.fetch_games(
        access_token,
        client_id,
        offset,
        limit,
        user_agent=user_agent,
        request_factory=request_factory,
        opener=opener,
    )


def download_igdb_metadata(
    access_token: str,
    client_id: str,
    igdb_ids: Iterable[str],
    *,
    batch_size: int,
    user_agent: str,
    request_factory: Callable[..., Any] | None = None,
    opener: Callable[[Any], Any] | None = None,
) -> dict[str, dict[str, Any]]:
    client = IGDBClient(user_agent=user_agent, request_factory=request_factory, opener=opener)
    return client.fetch_metadata_by_ids(
        access_token,
        client_id,
        igdb_ids,
        batch_size=batch_size,
        user_agent=user_agent,
        request_factory=request_factory,
        opener=opener,
    )


def _format_http_error(prefix: str, error: HTTPError) -> str:
    message = f"{prefix}: {error.code}"
    error_message = ""
    try:
        error_body = error.read()
    except Exception:  # pragma: no cover - best effort to capture error body
        error_body = b""
    if error_body:
        try:
            error_message = error_body.decode("utf-8", errors="replace").strip()
        except Exception:  # pragma: no cover - unexpected decoding failures
            error_message = ""
    if not error_message and error.reason:
        error_message = str(error.reason)
    if error_message:
        message = f"{message} {error_message}"
    return message
