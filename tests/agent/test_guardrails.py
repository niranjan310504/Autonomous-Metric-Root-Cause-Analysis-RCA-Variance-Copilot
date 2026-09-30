"""Tests for agent query guardrails."""

import pytest

from agent.guards import validate_read_only_sql


def test_read_only_query_is_allowed() -> None:
    assert validate_read_only_sql("SELECT * FROM metrics") == "SELECT * FROM metrics"


@pytest.mark.parametrize("query", ["DROP TABLE metrics", "UPDATE metrics SET value = 1"])
def test_mutating_queries_are_rejected(query: str) -> None:
    with pytest.raises(ValueError, match="read-only"):
        validate_read_only_sql(query)
