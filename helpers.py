"""General-purpose helper utilities shared across the application."""

from __future__ import annotations

import numbers
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

import pandas as pd


__all__ = [
    "_collect_company_names",
    "_dedupe_preserve_order",
    "_format_first_release_date",
    "_format_name_list",
    "_normalize_lookup_name",
    "_parse_company_names",
    "_parse_iterable",
    "has_cover_path_value",
    "has_summary_text",
]


def has_summary_text(value: Any) -> bool:
    """Return ``True`` when ``value`` contains non-empty summary text."""

    if value is None:
        return False
    if isinstance(value, str):
        text = value.strip()
    else:
        try:
            if pd.isna(value):
                return False
        except Exception:
            pass
        text = str(value).strip()
    if not text:
        return False
    if text.lower() == "nan":
        return False
    return True


def has_cover_path_value(value: Any) -> bool:
    """Return ``True`` when ``value`` contains a usable cover path."""

    if value is None:
        return False
    if isinstance(value, str):
        text = value.strip()
    else:
        try:
            if pd.isna(value):
                return False
        except Exception:
            pass
        text = str(value).strip()
    if not text:
        return False
    if text.lower() == "nan":
        return False
    return True


def _dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def _format_name_list(value: Any) -> str:
    return ", ".join(_dedupe_preserve_order(_parse_iterable(value)))


def _collect_company_names(
    companies: Any, role_key: str
) -> list[str]:  # pragma: no cover - simple data formatter
    names: list[str] = []
    if isinstance(companies, list):
        for company in companies:
            if not isinstance(company, Mapping):
                continue
            if not company.get(role_key):
                continue
            company_obj = company.get("company")
            name_value: Any = None
            if isinstance(company_obj, Mapping):
                name_value = company_obj.get("name")
            elif isinstance(company_obj, str):
                name_value = company_obj
            if not name_value:
                continue
            text = str(name_value).strip()
            if text:
                names.append(text)
    return _dedupe_preserve_order(names)


def _format_first_release_date(value: Any) -> str:
    if value in (None, "", 0):
        return ""
    try:
        timestamp = float(value)
    except (TypeError, ValueError):
        try:
            timestamp = float(str(value).strip())
        except (TypeError, ValueError):
            return ""
    if timestamp <= 0:
        return ""
    try:
        dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return ""
    return dt.date().isoformat()


def _normalize_lookup_name(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def _parse_iterable(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [v.strip() for v in value.split(',') if v.strip()]
    if isinstance(value, numbers.Number):
        return [str(value)]
    try:
        iterator = iter(value)
    except TypeError:
        return [str(value)]
    items: list[str] = []
    for element in iterator:
        if isinstance(element, Mapping):
            name = element.get("name")
            if isinstance(name, str) and name.strip():
                items.append(name.strip())
            else:
                items.append(str(element).strip())
        else:
            items.append(str(element).strip())
    return [item for item in items if item]


def _parse_company_names(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        cleaned = value.strip()
        return [cleaned] if cleaned else []
    if isinstance(value, numbers.Number):
        return [str(value)]
    names: list[str] = []
    try:
        iterator = iter(value)
    except TypeError:
        text = str(value).strip()
        return [text] if text else []
    for element in iterator:
        name_value: Any = None
        if isinstance(element, Mapping):
            if isinstance(element.get("name"), str):
                name_value = element["name"]
            else:
                company_obj = element.get("company")
                if isinstance(company_obj, Mapping) and isinstance(
                    company_obj.get("name"), str
                ):
                    name_value = company_obj["name"]
                elif isinstance(company_obj, str):
                    name_value = company_obj
        else:
            name_value = element
        text = _normalize_lookup_name(name_value)
        if text:
            names.append(text)
    return names
