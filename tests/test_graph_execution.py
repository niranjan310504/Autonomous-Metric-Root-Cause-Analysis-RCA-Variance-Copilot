from typing import Any, cast

import pandas as pd
import pytest
from data.generate_synthetic_data import generate_dataset, generate_timeseries_dataset

import graph
from agents import llm_adapter
from agents.state import RCAState


def _generate_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    return generate_dataset()


def _generate_timeseries_dataset(days: int = 30) -> pd.DataFrame:
    return generate_timeseries_dataset(days=days)


def test_graph_produces_trace_and_finding(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    baseline, current = _generate_dataset()
    result = graph.run_analysis(baseline, current)
    assert result.trace == [
        "schema_inspector",
        "hypothesis_generator",
        "sql_investigator",
        "narrative_synthesizer",
    ]
    assert result.findings
    assert any(
        finding.rate_effect != 0.0 or finding.mix_effect != 0.0
        for finding in result.findings
    )
    assert all(
        finding.rate_effect + finding.mix_effect + finding.interaction_effect
        == pytest.approx(finding.contribution)
        for finding in result.findings
    )
    assert all(0.0 <= finding.p_value <= 1.0 for finding in result.findings)
    assert all(
        finding.confidence_tier in {"high", "moderate", "noise"}
        for finding in result.findings
    )
    assert all(
        "raw p-value" in finding.evidence and "adjusted p-value" in finding.evidence
        for finding in result.findings
    )
    assert result.subgroups
    assert all(
        {"rule", "support", "conversion_rate", "lift_vs_overall"} <= subgroup.keys()
        for subgroup in result.subgroups
    )
    assert "driver" in result.narrative


def test_graph_with_mocked_gemini(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-only-key")
    monkeypatch.setattr(
        llm_adapter,
        "_generate_with_gemini",
        lambda prompt, api_key: "Executive narrative from mocked Gemini.",
    )
    baseline, current = _generate_dataset()

    result = graph.run_analysis(baseline, current)

    assert result.trace == [
        "schema_inspector",
        "hypothesis_generator",
        "sql_investigator",
        "narrative_synthesizer",
    ]
    assert result.findings
    assert result.subgroups
    assert result.narrative == "[Gemini] Executive narrative from mocked Gemini."


def test_graph_triggers_cross_dimensional_investigation_for_interaction_anomaly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    baseline = pd.DataFrame(
        [
            {"device": "mobile", "region": "APAC", "visits": 100, "conversions": 20},
            {"device": "mobile", "region": "EMEA", "visits": 100, "conversions": 20},
            {"device": "desktop", "region": "APAC", "visits": 100, "conversions": 20},
            {"device": "desktop", "region": "EMEA", "visits": 100, "conversions": 20},
        ]
    )
    current = baseline.copy()
    current["conversions"] = [60, 0, 0, 60]

    result = graph.run_analysis(baseline, current)

    assert result.loop_count == 1
    assert result.deep_dive_dimension == "device x region"
    assert "cross_dimensional_investigator" in result.trace
    assert any(
        finding.dimension == "device x region" and finding.segment == "mobile x APAC"
        for finding in result.findings
    )


def test_timeseries_graph_runs_wow_analysis(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    frame = _generate_timeseries_dataset()

    result, baseline, current = graph.run_timeseries_analysis(frame)

    assert result.findings
    assert not baseline.empty
    assert not current.empty
    assert baseline["timestamp"].max() < current["timestamp"].min()


def test_timeseries_graph_rejects_invalid_mode() -> None:
    with pytest.raises(ValueError, match="mode must be either"):
        graph.run_timeseries_analysis(_generate_timeseries_dataset(), mode="daily")


def test_timeseries_graph_rejects_insufficient_yoy_data() -> None:
    with pytest.raises(ValueError, match="two non-empty comparison windows"):
        graph.run_timeseries_analysis(_generate_timeseries_dataset(), mode="yoy")


def test_timeseries_graph_runs_yoy_analysis(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    frame = _generate_timeseries_dataset(days=730)
    captured: dict[str, pd.DataFrame] = {}

    def fake_run_analysis(baseline: pd.DataFrame, current: pd.DataFrame) -> RCAState:
        captured["baseline"] = baseline
        captured["current"] = current
        return RCAState(
            baseline=cast(list[dict[str, Any]], baseline.to_dict("records")),
            current=cast(list[dict[str, Any]], current.to_dict("records")),
        )

    monkeypatch.setattr(graph, "run_analysis", fake_run_analysis)

    result, baseline, current = graph.run_timeseries_analysis(frame, mode="yoy")

    assert result.baseline == baseline.to_dict("records")
    assert result.current == current.to_dict("records")
    assert captured["baseline"].equals(baseline)
    assert captured["current"].equals(current)
    assert baseline["timestamp"].max() < current["timestamp"].min()
    assert baseline["timestamp"].nunique() >= 300
    assert current["timestamp"].nunique() >= 300
