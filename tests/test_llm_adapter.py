import sys
from collections.abc import Mapping
from types import SimpleNamespace

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


def test_gemini_uses_content_generation_api(monkeypatch: pytest.MonkeyPatch) -> None:
    request: dict[str, str] = {}

    class FakeHttpOptions:
        def __init__(self, timeout: int) -> None:
            request["timeout"] = str(timeout)

    class FakeModels:
        def generate_content(self, **kwargs: str) -> SimpleNamespace:
            request.update(kwargs)
            return SimpleNamespace(text="Generated through content generation.")

    class FakeGenai:
        def __init__(self, api_key: str, http_options: FakeHttpOptions) -> None:
            assert api_key == "test-only-key"
            assert http_options is not None
            self.models = FakeModels()

    monkeypatch.setitem(
        sys.modules,
        "google.genai",
        SimpleNamespace(Client=FakeGenai, types=SimpleNamespace(HttpOptions=FakeHttpOptions)),
    )
    monkeypatch.setenv("GEMINI_MODEL", "test-model")

    narrative = llm_adapter._generate_with_gemini("validated prompt", "test-only-key")

    assert narrative == "Generated through content generation."
    assert request == {
        "model": "test-model",
        "contents": "validated prompt",
        "timeout": "120000",
    }


def test_timeout_can_be_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    request: dict[str, int] = {}

    class FakeHttpOptions:
        def __init__(self, timeout: int) -> None:
            request["timeout"] = timeout

    class FakeModels:
        def generate_content(self, **kwargs: str) -> SimpleNamespace:
            return SimpleNamespace(text="Configured timeout response.")

    class FakeGenai:
        def __init__(self, api_key: str, http_options: FakeHttpOptions) -> None:
            self.models = FakeModels()

    monkeypatch.setitem(
        sys.modules,
        "google.genai",
        SimpleNamespace(Client=FakeGenai, types=SimpleNamespace(HttpOptions=FakeHttpOptions)),
    )
    monkeypatch.setenv("GEMINI_TIMEOUT_MS", "45000")

    assert llm_adapter._generate_with_gemini("prompt", "test-only-key")
    assert request == {"timeout": 45000}


def test_successful_result_reports_gemini_source(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-only-key")
    monkeypatch.setattr(
        llm_adapter,
        "_generate_with_gemini",
        lambda prompt, api_key: "Provider response.",
    )

    result = llm_adapter.generate_narrative_result("conversion_rate", FINDINGS, SUBGROUPS)

    assert result.narrative == "[Gemini] Provider response."
    assert result.source == "gemini"
    assert result.provider_error is None


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


def test_timeout_failure_is_classified_and_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-only-key")

    def timed_out(prompt: str, api_key: str) -> str:
        raise TimeoutError()

    monkeypatch.setattr(llm_adapter, "_generate_with_gemini", timed_out)

    result = llm_adapter.generate_narrative_result("conversion_rate", FINDINGS, SUBGROUPS)

    assert result.source == "deterministic_fallback"
    assert result.provider_error == "Gemini request timed out after 120000 ms"
    with pytest.raises(llm_adapter.GeminiProviderError, match="timed out"):
        llm_adapter.generate_narrative_result(
            "conversion_rate", FINDINGS, SUBGROUPS, require_gemini=True
        )


@pytest.mark.parametrize(
    ("code", "expected"),
    [(401, "authentication failed"), (404, "model or API endpoint was not found")],
)
def test_api_failures_are_classified(
    monkeypatch: pytest.MonkeyPatch, code: int, expected: str
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-only-key")

    class FakeAPIError(Exception):
        def __init__(self) -> None:
            self.code = code

    def failing_generate(prompt: str, api_key: str) -> str:
        raise FakeAPIError()

    monkeypatch.setattr(
        llm_adapter,
        "_generate_with_gemini",
        failing_generate,
    )

    with pytest.raises(llm_adapter.GeminiProviderError, match=expected):
        llm_adapter.generate_narrative_result(
            "conversion_rate", FINDINGS, SUBGROUPS, require_gemini=True
        )


def test_invalid_timeout_uses_fallback_or_strict_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-only-key")
    monkeypatch.setenv("GEMINI_TIMEOUT_MS", "not-a-number")

    result = llm_adapter.generate_narrative_result("conversion_rate", FINDINGS, SUBGROUPS)
    assert result.source == "deterministic_fallback"
    assert result.provider_error == "GEMINI_TIMEOUT_MS must be a positive integer in milliseconds"

    with pytest.raises(llm_adapter.GeminiProviderError, match="positive integer"):
        llm_adapter.generate_narrative_result(
            "conversion_rate", FINDINGS, SUBGROUPS, require_gemini=True
        )


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
