"""Shared testing helpers for loading the Flask app without live IGDB calls."""

from __future__ import annotations

import importlib.util
import json
import os
import uuid
from pathlib import Path
from typing import Iterable
from unittest.mock import patch

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


class _DummyResponse:
    def __init__(self, body: bytes):
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self) -> bytes:
        return self._body


def _default_game_payload() -> list[dict[str, object]]:
    return []


def _patch_urlopen(pages: Iterable[list[dict[str, object]]]):
    token_body = json.dumps({"access_token": "test-token"}).encode("utf-8")
    response_queue = [json.dumps(page).encode("utf-8") for page in pages]
    response_queue.append(b"[]")  # Ensure the pagination loop terminates.

    def fake_urlopen(request):
        url = getattr(request, "full_url", "")
        if "oauth2/token" in str(url):
            return _DummyResponse(token_body)
        try:
            body = response_queue.pop(0)
        except IndexError:
            body = b"[]"
        return _DummyResponse(body)

    return patch("urllib.request.urlopen", new=fake_urlopen)


def load_app(tmp_path: Path) -> object:
    """Import the application module using deterministic IGDB responses."""

    os.chdir(tmp_path)
    module_name = f"app_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, APP_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load app module specification")
    module = importlib.util.module_from_spec(spec)

    pages = [_default_game_payload()]

    original_env = {
        key: os.environ.get(key)
        for key in ("TWITCH_CLIENT_ID", "TWITCH_CLIENT_SECRET")
    }
    os.environ["TWITCH_CLIENT_ID"] = original_env.get("TWITCH_CLIENT_ID") or "dummy-id"
    os.environ["TWITCH_CLIENT_SECRET"] = (
        original_env.get("TWITCH_CLIENT_SECRET") or "dummy-secret"
    )

    with _patch_urlopen(pages):
        try:
            spec.loader.exec_module(module)
        finally:
            for key, value in original_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    module.games_df = module.games_df.copy()
    module.total_games = len(module.games_df)

    return module
