"""HTML-facing Flask routes and authentication flows."""
from __future__ import annotations

from typing import Any, Callable, Mapping

from flask import (
    Blueprint,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

web_blueprint = Blueprint("web", __name__)

_context: dict[str, Any] = {}


def configure(context: Mapping[str, Any]) -> None:
    """Provide shared state required by the HTML routes."""
    _context.update(context)


def _ctx(key: str) -> Any:
    if key not in _context:
        raise RuntimeError(f"web routes missing context value: {key}")
    return _context[key]


def _get_total_games() -> int:
    getter: Callable[[], int] = _ctx("get_total_games")
    return getter()


def _get_categories() -> list[str]:
    getter: Callable[[], list[str]] = _ctx("get_categories")
    return getter()


def _get_platforms() -> list[str]:
    getter: Callable[[], list[str]] = _ctx("get_platforms")
    return getter()


def _get_app_password() -> str:
    return _ctx("app_password")


@web_blueprint.before_app_request
def require_login():
    if request.endpoint in ("web.login", "static"):
        return None
    if session.get("authenticated"):
        return None
    return redirect(url_for("web.login"))


@web_blueprint.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        if request.form.get("password") == _get_app_password():
            session["authenticated"] = True
            return redirect(url_for("web.index"))
        error = "Invalid password"
    return render_template("login.html", error=error)


@web_blueprint.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("web.login"))


@web_blueprint.route("/")
def index():
    return render_template(
        "index.html",
        total=_get_total_games(),
        categories=_get_categories(),
        platforms=_get_platforms(),
    )
