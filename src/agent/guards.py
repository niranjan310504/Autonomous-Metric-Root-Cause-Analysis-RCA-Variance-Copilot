"""Guardrails for agent-generated analytical queries."""

import re


def validate_read_only_sql(query: str) -> str:
    """Reject mutating SQL and return a normalized read-only query."""

    normalized = query.strip().rstrip(";").strip()
    if not normalized:
        raise ValueError("query must not be empty")
    forbidden = (
        r"\b(drop|delete|insert|update|alter|truncate|create|replace|copy|"
        r"attach|detach|install|load)\b"
    )
    if ";" in normalized or re.search(forbidden, normalized, re.IGNORECASE):
        raise ValueError("only read-only SQL queries are allowed")
    if not re.match(r"^(select|with)\b", normalized, re.IGNORECASE):
        raise ValueError("query must start with SELECT or WITH")
    return normalized
