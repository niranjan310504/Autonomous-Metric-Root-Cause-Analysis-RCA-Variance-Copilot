import importlib.util
from typing import Any, cast

import pandas as pd
import pytest

from agents import llm_adapter
from graph import run_analysis, run_timeseries_analysis


def _generate_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    spec = importlib.util.spec_from_file_location("generator", "data/generate_synthetic_data.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return cast(tuple[pd.DataFrame, pd.DataFrame], module.generate_dataset())


def _generate_timeseries_dataset(days: int = 30) -> pd.DataFrame:
    spec = importlib.util.spec_from_file_location(
        "timeseries_generator", "data/generate_synthetic_data.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return cast(pd.DataFrame, cast(Any, module).generate_timeseries_dataset(days=days))


def test_graph_produces_trace_and_finding(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    baseline, current = _generate_dataset()
    result = run_analysis(baseline, current)
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

    result = run_analysis(baseline, current)

    assert result.trace == [
        "schema_inspector",
        "hypothesis_generator",
        "sql_investigator",
        "narrative_synthesizer",
    ]
    assert result.findings
    assert result.subgroups
    assert result.narrative == "[Gemini] Executive narrative from mocked Gemini."


def test_timeseries_graph_runs_wow_analysis(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    frame = _generate_timeseries_dataset()

    result, baseline, current = run_timeseries_analysis(frame)

    assert result.findings
    assert not baseline.empty
    assert not current.empty
    assert baseline["timestamp"].max() < current["timestamp"].min()


def test_timeseries_graph_rejects_invalid_mode() -> None:
    with pytest.raises(ValueError, match="mode must be either"):
        run_timeseries_analysis(_generate_timeseries_dataset(), mode="daily")


def test_timeseries_graph_rejects_insufficient_yoy_data() -> None:
    with pytest.raises(ValueError, match="two non-empty comparison windows"):
        run_timeseries_analysis(_generate_timeseries_dataset(), mode="yoy")


def test_timeseries_graph_runs_yoy_analysis(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    frame = _generate_timeseries_dataset(days=730)

    result, baseline, current = run_timeseries_analysis(frame, mode="yoy")

    assert result.findings
    assert baseline["timestamp"].max() < current["timestamp"].min()
    assert baseline["timestamp"].nunique() >= 300
    assert current["timestamp"].nunique() >= 300
