import pandas as pd
import pytest

from agents.sql_investigator import _apply_cevr_gate, investigate_cross_dimensions
from agents.state import Finding, RCAState, SchemaProfile
from db.duckdb_manager import DuckDBManager


def _state() -> RCAState:
    records = [
        {"device": "mobile", "region": "APAC", "visits": 100, "conversions": 5},
        {"device": "mobile", "region": "EMEA", "visits": 100, "conversions": 20},
        {"device": "desktop", "region": "APAC", "visits": 100, "conversions": 20},
        {"device": "desktop", "region": "EMEA", "visits": 100, "conversions": 20},
    ]
    return RCAState(
        baseline=records,
        current=records,
        schema_profile=SchemaProfile(
            columns=[],
            cardinalities={"device": 2, "region": 2},
            dimensions=["device", "region"],
        ),
    )


def test_cevr_gate_selects_two_dimensions_when_coverage_is_low() -> None:
    state = _state()
    baseline = pd.DataFrame(state.baseline)
    current = baseline.copy()
    current["conversions"] = [0, 20, 20, 20]
    findings = [
        Finding(
            dimension="device",
            segment="mobile",
            contribution=-0.001,
            is_statistically_significant=True,
            evidence="test",
        )
    ]

    _apply_cevr_gate(state, findings, baseline, current)

    assert state.requires_deep_dive is True
    assert state.deep_dive_dimension == "device x region"


def test_cross_investigation_appends_tagged_findings_and_closes_loop() -> None:
    state = _state()
    baseline = pd.DataFrame(state.baseline)
    current = baseline.copy()
    state.deep_dive_dimension = "device x region"

    with DuckDBManager() as manager:
        manager.register_dataframe("current_metrics_baseline", baseline)
        manager.register_dataframe("current_metrics", current)
        result = investigate_cross_dimensions(state, manager)

    assert result.loop_count == 1
    assert result.requires_deep_dive is False
    assert result.trace == ["cross_dimensional_investigator"]
    assert result.findings
    assert all(finding.dimension == "device x region" for finding in result.findings)
    assert all(" x " in finding.segment for finding in result.findings)


def test_cross_investigation_rejects_unknown_dimensions() -> None:
    state = _state()
    state.deep_dive_dimension = "device x country"
    baseline = pd.DataFrame(state.baseline)

    with DuckDBManager() as manager:
        manager.register_dataframe("current_metrics_baseline", baseline)
        manager.register_dataframe("current_metrics", baseline)
        with pytest.raises(ValueError, match="schema profile"):
            investigate_cross_dimensions(state, manager)
