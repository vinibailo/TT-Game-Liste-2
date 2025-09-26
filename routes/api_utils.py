"""Shared helpers for API routes (error handling and logging)."""

from __future__ import annotations

import json
from functools import wraps
from typing import Any, Callable, ParamSpec, TypeVar

from flask import current_app, jsonify, request, session
from werkzeug.exceptions import HTTPException

P = ParamSpec("P")
R = TypeVar("R")


class APIError(Exception):
    """Base class for API errors that includes an HTTP status code."""

    status_code: int = 500
    message: str = "An unexpected error occurred."

    def __init__(
        self,
        message: str | None = None,
        *,
        status_code: int | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message or self.message)
        self.message = message or self.message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload or {}

    def to_dict(self) -> dict[str, Any]:
        data = {"error": self.message}
        data.update(self.payload)
        return data


class BadRequestError(APIError):
    status_code = 400
    message = "Invalid request."


class UnauthorizedError(APIError):
    status_code = 401
    message = "Unauthorized."


class ForbiddenError(APIError):
    status_code = 403
    message = "Forbidden."


class NotFoundError(APIError):
    status_code = 404
    message = "Resource not found."


class MethodNotAllowedError(APIError):
    status_code = 405
    message = "Method not allowed."


class ConflictError(APIError):
    status_code = 409
    message = "Conflict detected."


class UpstreamServiceError(APIError):
    status_code = 502
    message = "Upstream service unavailable."


def _resolve_user() -> str:
    try:
        if "user" in session:
            return str(session.get("user"))
        if session.get("authenticated"):
            return "authenticated"
    except RuntimeError:
        return "unknown"
    return "anonymous"


def _collect_request_context() -> dict[str, Any]:
    context: dict[str, Any] = {
        "route": request.path,
        "endpoint": request.endpoint,
        "method": request.method,
        "user": _resolve_user(),
        "view_args": dict(request.view_args or {}),
        "args": request.args.to_dict(flat=False),
    }

    if request.form:
        context["form"] = request.form.to_dict(flat=False)

    try:
        json_payload = request.get_json(silent=True)
    except Exception:
        json_payload = None
    if json_payload is not None:
        context["json"] = json_payload

    return context


def _serialize_context(context: dict[str, Any]) -> str:
    try:
        return json.dumps(context, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return repr(context)


def _log_api_error(exc: Exception, *, status_code: int, handled: bool) -> None:
    context = _collect_request_context()
    context["status_code"] = status_code
    context_str = _serialize_context(context)
    if handled and status_code < 500:
        current_app.logger.warning(
            "Handled API error (%s): %s | context=%s", status_code, exc, context_str
        )
        return
    if handled:
        current_app.logger.error(
            "Handled API error (%s): %s | context=%s", status_code, exc, context_str,
            exc_info=exc,
        )
        return
    current_app.logger.exception(
        "Unhandled API error (%s): %s | context=%s", status_code, exc, context_str
    )


def handle_api_errors(func: Callable[P, R]) -> Callable[P, R]:
    """Decorator that centralizes API error handling and logging."""

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs):  # type: ignore[misc]
        try:
            return func(*args, **kwargs)
        except APIError as exc:
            status_code = exc.status_code
            _log_api_error(exc, status_code=status_code, handled=True)
            response = jsonify(exc.to_dict())
            return response, status_code
        except HTTPException as exc:
            status_code = exc.code or 500
            message = exc.description or str(exc)
            api_error = APIError(message=message, status_code=status_code)
            _log_api_error(exc, status_code=status_code, handled=True)
            return jsonify(api_error.to_dict()), status_code
        except Exception as exc:  # pragma: no cover - defensive guard
            _log_api_error(exc, status_code=500, handled=False)
            return jsonify({"error": "Internal server error"}), 500

    return wrapper


__all__ = [
    "APIError",
    "BadRequestError",
    "ConflictError",
    "ForbiddenError",
    "MethodNotAllowedError",
    "NotFoundError",
    "UnauthorizedError",
    "UpstreamServiceError",
    "handle_api_errors",
]

