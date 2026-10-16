"""Regression tests for the canonical package layout."""

from importlib.util import find_spec

from agents.state import RCAState
from alerting.slack_notifier import format_slack_alert
from db.duckdb_manager import validate_read_only_sql
from engine.variance_waterfall import conversion_rate


def _is_discoverable(module_name: str) -> bool:
    try:
        return find_spec(module_name) is not None
    except ModuleNotFoundError:
        return False


def test_active_packages_and_guardrail_are_importable() -> None:
    assert RCAState.__module__ == "agents.state"
    assert format_slack_alert.__module__ == "alerting.slack_notifier"
    assert conversion_rate.__module__ == "engine.variance_waterfall"
    assert validate_read_only_sql("SELECT 1") == "SELECT 1"


def test_removed_scaffold_packages_are_not_discoverable() -> None:
    for module_name in ("agent.guards", "core.config", "data.schema", "ui.app"):
        assert not _is_discoverable(module_name)
