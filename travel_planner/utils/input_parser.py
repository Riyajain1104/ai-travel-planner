"""Helpers for passing structured travel requirements between workflow agents."""

from __future__ import annotations

import json
from datetime import date, timedelta
from typing import Any


def parse_agent_input(input_data: str | list[dict[str, Any]]) -> dict[str, Any]:
    """Parse the structured JSON produced by the workflow base node."""
    if isinstance(input_data, list):
        for message in reversed(input_data):
            if message.get("role") == "user":
                input_data = message.get("content", "")
                break
        else:
            return {}
    if not isinstance(input_data, str):
        return {}
    try:
        value = json.loads(input_data)
    except (TypeError, json.JSONDecodeError):
        return {"original_query": input_data}
    return value if isinstance(value, dict) else {"original_query": input_data}


def get_travel_query(input_data: str | list[dict[str, Any]]) -> dict[str, Any]:
    """Return the travel_query section from structured agent input."""
    payload = parse_agent_input(input_data)
    value = payload.get("travel_query", {})
    return value if isinstance(value, dict) else {}


def get_preferences(input_data: str | list[dict[str, Any]]) -> dict[str, Any]:
    """Return the preferences section from structured agent input."""
    payload = parse_agent_input(input_data)
    value = payload.get("preferences", {})
    return value if isinstance(value, dict) else {}


def get_original_query(input_data: str | list[dict[str, Any]]) -> str:
    """Return the original natural-language user request."""
    payload = parse_agent_input(input_data)
    value = payload.get("original_query")
    return str(value) if value else ""


def query_dates(query: dict[str, Any]) -> tuple[str | None, str | None]:
    """Get dates from a TravelQuery, using duration when available."""
    start = query.get("departure_date") or query.get("start_date")
    end = query.get("return_date") or query.get("end_date")

    if start and end:
        return str(start), str(end)

    duration = query.get("duration_days")
    if start and duration:
        try:
            start_date = date.fromisoformat(str(start))
            end_date = start_date + timedelta(days=max(int(duration) - 1, 0))
            return start_date.isoformat(), end_date.isoformat()
        except ValueError:
            pass

    return (str(start) if start else None, str(end) if end else None)
