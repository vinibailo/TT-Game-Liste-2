"""Flask application factory and service client initialization."""
from __future__ import annotations

from typing import Callable

from flask import Flask


def create_app(
    flask_app: Flask | None = None,
    *,
    configure_blueprints: Callable[[Flask], None] | None = None,
) -> Flask:
    """Return a configured Flask application instance."""
    if flask_app is None or configure_blueprints is None:
        from app import app as default_app, configure_blueprints as default_configure

        if flask_app is None:
            flask_app = default_app
        if configure_blueprints is None:
            configure_blueprints = default_configure

    configure_blueprints(flask_app)
    return flask_app
