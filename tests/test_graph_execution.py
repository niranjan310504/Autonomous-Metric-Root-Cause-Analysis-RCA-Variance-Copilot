import importlib.util
from typing import cast

import pandas as pd
import pytest

from agents import llm_adapter
from graph import run_analysis


def _generate_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    spec = importlib.util.spec_from_file_location("generator", "data/generate_synthetic_data.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return cast(tuple[pd.DataFrame, pd.DataFrame], module.generate_dataset())


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
