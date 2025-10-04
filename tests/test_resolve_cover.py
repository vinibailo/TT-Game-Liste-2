from __future__ import annotations

from pathlib import Path

from tests.app_helpers import load_app


def test_resolve_cover_uses_static_placeholder(tmp_path):
    app_module = load_app(tmp_path)

    result = app_module.resolve_cover()

    assert result == '/static/no-image.jpg'


def test_resolve_cover_handles_static_placeholder_override(tmp_path):
    app_module = load_app(tmp_path)

    result = app_module.resolve_cover(placeholder='/static/custom-placeholder.png')

    assert result == '/static/custom-placeholder.png'


def test_resolve_cover_preserves_absolute_placeholder(tmp_path):
    app_module = load_app(tmp_path)
    placeholder = 'https://example.com/fallback.png'

    result = app_module.resolve_cover(placeholder=placeholder)

    assert result == placeholder


def test_static_placeholder_file_exists(tmp_path):
    app_module = load_app(tmp_path)

    module_root = Path(app_module.__file__).resolve().parent
    static_path = module_root / 'static' / 'no-image.jpg'

    assert static_path.is_file()
