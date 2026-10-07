from collections.abc import Mapping

import pytest

from agents import llm_adapter

FINDINGS: list[Mapping[str, object]] = [
    {
        "dimension": "device",
        "segment": "ios",
        "contribution": -0.12,
        "evidence": "4.00% current vs 16.00% baseline",
    }
]
SUBGROUPS: list[Mapping[str, object]] = [
    {
        "rule": "device=ios AND gateway=stripe",
        "support": 50,
        "conversion_rate": 0.04,
        "lift_vs_overall": 0.25,
    }
]


def test_missing_key_returns_labeled_deterministic_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    narrative = llm_adapter.generate_actionable_narrative(
        "conversion_rate", FINDINGS, SUBGROUPS
    )

    assert narrative.startswith("[Deterministic fallback]")
    assert "device=ios" in narrative
    assert "Recommended next step" in narrative


def test_empty_findings_returns_labeled_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    narrative = llm_adapter.generate_actionable_narrative("conversion_rate", [], [])

    assert narrative.startswith("[Deterministic fallback]")
    assert "No major driver" in narrative
    assert "Recommended next step" in narrative


def test_successful_provider_response_is_labeled_and_uses_evidence_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, str] = {}
    monkeypatch.setenv("GEMINI_API_KEY", "test-only-key")

    def fake_generate(prompt: str, api_key: str) -> str:
        captured["prompt"] = prompt
        captured["api_key"] = api_key
        return "Investigate the iOS payment path first."

    monkeypatch.setattr(llm_adapter, "_generate_with_gemini", fake_generate)

    narrative = llm_adapter.generate_actionable_narrative(
        "conversion_rate", FINDINGS, SUBGROUPS
    )

    assert narrative == "[Gemini] Investigate the iOS payment path first."
    assert captured["api_key"] == "test-only-key"
    assert "device" in captured["prompt"]
    assert "device=ios AND gateway=stripe" in captured["prompt"]
    assert "validated evidence" in captured["prompt"]


def test_provider_failure_returns_deterministic_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-only-key")

    def failing_generate(prompt: str, api_key: str) -> str:
        raise RuntimeError("simulated provider outage")

    monkeypatch.setattr(llm_adapter, "_generate_with_gemini", failing_generate)

    narrative = llm_adapter.generate_actionable_narrative(
        "conversion_rate", FINDINGS, SUBGROUPS
    )

    assert narrative.startswith("[Deterministic fallback]")
    assert "device=ios" in narrative


def test_strict_mode_rejects_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(llm_adapter.GeminiProviderError, match="not configured"):
        llm_adapter.generate_actionable_narrative(
            "conversion_rate", FINDINGS, SUBGROUPS, require_gemini=True
        )


def test_strict_mode_rejects_provider_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-only-key")

    def failing_generate(prompt: str, api_key: str) -> str:
        raise RuntimeError("simulated provider outage")

    monkeypatch.setattr(llm_adapter, "_generate_with_gemini", failing_generate)

    with pytest.raises(llm_adapter.GeminiProviderError, match="generation failed"):
        llm_adapter.generate_actionable_narrative(
            "conversion_rate", FINDINGS, SUBGROUPS, require_gemini=True
        )
