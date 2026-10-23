"""Optional Gemini narrative generation with a deterministic fallback."""

from __future__ import annotations

import importlib
import json
import logging
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal

from dotenv import load_dotenv

_DEFAULT_MODEL = "gemini-flash-latest"
_DEFAULT_TIMEOUT_MS = 120_000
_LOGGER = logging.getLogger(__name__)

load_dotenv()


class GeminiProviderError(RuntimeError):
    """Raised when strict Gemini verification cannot reach the provider."""

    def __init__(self, message: str, category: GeminiFailureCategory) -> None:
        super().__init__(message)
        self.category = category


class GeminiFailureCategory(StrEnum):
    """Safe categories for failures reaching Gemini."""

    CONFIGURATION = "configuration"
    TIMEOUT = "timeout"
    AUTHENTICATION = "authentication"
    MODEL_API = "model/API"
    QUOTA = "quota"
    NETWORK = "network"
    PROVIDER = "provider"


@dataclass(frozen=True)
class NarrativeResult:
    """Narrative text plus safe provider/fallback status for the UI."""

    narrative: str
    source: Literal["gemini", "deterministic_fallback"]
    provider_error: str | None = None


def _fallback_narrative(
    metric_name: str,
    findings: Sequence[Mapping[str, object]],
) -> str:
    """Build a stable narrative when Gemini is unavailable."""

    top = findings[0] if findings else None
    if top is None:
        return (
            "[Deterministic fallback] No major driver was identified for "
            f"{metric_name}. Recommended next step: validate data freshness and coverage."
        )

    dimension = str(top.get("dimension", "unknown dimension"))
    segment = str(top.get("segment", "unknown segment"))
    evidence = str(top.get("evidence", "no additional evidence"))
    contribution = top.get("contribution")
    contribution_text = (
        f" with an estimated contribution of {float(contribution):+.2%}"
        if isinstance(contribution, (int, float))
        else ""
    )
    return (
        f"[Deterministic fallback] {metric_name} changed from the baseline to the current period. "
        f"The strongest observed driver is {dimension}={segment}{contribution_text}; {evidence}. "
        f"Recommended next step: investigate the {dimension}={segment} segment and "
        "validate its upstream data."
    )


def _build_prompt(
    metric_name: str,
    findings: Sequence[Mapping[str, object]],
    subgroups: Sequence[Mapping[str, object]],
) -> str:
    """Build an evidence-only prompt for executive narrative synthesis."""

    evidence = {
        "metric_name": metric_name,
        "findings": findings,
        "subgroups": subgroups,
    }
    return (
        "You are an executive analytics narrator. Use only the validated evidence below. "
        "Do not calculate, revise, or infer metric values or attribution. Write a concise "
        "C-suite summary followed by 1-3 actionable diagnostic next steps. Mention the "
        "strongest driver and distinguish rate, mix, and interaction evidence when present. "
        "Return plain text only.\n\n"
        f"Validated evidence:\n{json.dumps(evidence, sort_keys=True, default=str)}"
    )


def _timeout_ms() -> int:
    """Read and validate the provider timeout without exposing credentials."""

    raw_timeout = os.getenv("GEMINI_TIMEOUT_MS")
    if raw_timeout is None or not raw_timeout.strip():
        return _DEFAULT_TIMEOUT_MS
    try:
        timeout_ms = int(raw_timeout)
    except ValueError as exc:
        raise GeminiProviderError(
            "GEMINI_TIMEOUT_MS must be a positive integer in milliseconds",
            GeminiFailureCategory.CONFIGURATION,
        ) from exc
    if timeout_ms <= 0:
        raise GeminiProviderError(
            "GEMINI_TIMEOUT_MS must be a positive integer in milliseconds",
            GeminiFailureCategory.CONFIGURATION,
        )
    return timeout_ms


def _generate_with_gemini(prompt: str, api_key: str) -> str:
    """Call Gemini's standard content-generation API."""

    timeout_ms = _timeout_ms()
    genai: Any = importlib.import_module("google.genai")
    genai_types: Any = getattr(genai, "types", None)
    if genai_types is None:
        genai_types = importlib.import_module("google.genai.types")
    client: Any = genai.Client(
        api_key=api_key,
        http_options=genai_types.HttpOptions(timeout=timeout_ms),
    )
    configured_model = os.getenv("GEMINI_MODEL", _DEFAULT_MODEL)
    try:
        response: Any = client.models.generate_content(
            model=configured_model,
            contents=prompt,
        )
    except Exception as exc:
        if getattr(exc, "code", None) != 404 or configured_model == _DEFAULT_MODEL:
            raise
        _LOGGER.warning(
            "Configured Gemini model %s was not found; retrying with %s",
            configured_model,
            _DEFAULT_MODEL,
        )
        response = client.models.generate_content(
            model=_DEFAULT_MODEL,
            contents=prompt,
        )
    text = getattr(response, "text", None)
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Gemini returned an empty narrative")
    return text.strip()


def _classify_provider_error(exc: Exception, timeout_ms: int) -> tuple[GeminiFailureCategory, str]:
    """Convert SDK and transport exceptions into safe user-facing diagnostics."""

    exception_name = type(exc).__name__.lower()
    code = getattr(exc, "code", None)
    message = str(getattr(exc, "message", "") or exc).lower()
    if isinstance(exc, TimeoutError) or "timeout" in exception_name or "timed out" in message:
        return (
            GeminiFailureCategory.TIMEOUT,
            f"Gemini request timed out after {timeout_ms} ms",
        )
    if code in (401, 403) or any(
        phrase in message for phrase in ("unauthenticated", "unauthorized", "invalid api key")
    ):
        return (
            GeminiFailureCategory.AUTHENTICATION,
            "Gemini authentication failed; verify GEMINI_API_KEY",
        )
    if code == 429 or any(
        phrase in message for phrase in ("quota", "rate limit", "resource exhausted")
    ):
        return GeminiFailureCategory.QUOTA, "Gemini quota or rate limit was exceeded"
    if code == 404 or "not found" in message or "model" in message and "invalid" in message:
        return GeminiFailureCategory.MODEL_API, "Gemini model or API endpoint was not found"
    if any(term in exception_name for term in ("requesterror", "connection", "connect")):
        return GeminiFailureCategory.NETWORK, "A network error prevented contact with Gemini"
    return (
        GeminiFailureCategory.PROVIDER,
        "Gemini narrative generation failed: unexpected provider error",
    )


def generate_narrative_result(
    metric_name: str,
    findings: Sequence[Mapping[str, object]],
    subgroups: Sequence[Mapping[str, object]],
    *,
    require_gemini: bool = False,
) -> NarrativeResult:
    """Generate a narrative and retain a safe provider status for callers."""

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        error = "Gemini authentication is not configured: GEMINI_API_KEY is missing"
        if require_gemini:
            raise GeminiProviderError(error, GeminiFailureCategory.CONFIGURATION)
        return NarrativeResult(
            narrative=_fallback_narrative(metric_name, findings),
            source="deterministic_fallback",
            provider_error=error,
        )

    timeout_ms = _DEFAULT_TIMEOUT_MS
    try:
        timeout_ms = _timeout_ms()
        narrative = _generate_with_gemini(
            _build_prompt(metric_name, findings, subgroups), api_key
        )
    except GeminiProviderError as exc:
        if require_gemini:
            raise
        return NarrativeResult(
            narrative=_fallback_narrative(metric_name, findings),
            source="deterministic_fallback",
            provider_error=str(exc),
        )
    except Exception as exc:
        category, error = _classify_provider_error(exc, timeout_ms)
        _LOGGER.warning("Gemini narrative generation failed (%s): %s", category, error)
        if require_gemini:
            raise GeminiProviderError(error, category) from exc
        return NarrativeResult(
            narrative=_fallback_narrative(metric_name, findings),
            source="deterministic_fallback",
            provider_error=error,
        )
    return NarrativeResult(narrative=f"[Gemini] {narrative}", source="gemini")


def generate_actionable_narrative(
    metric_name: str,
    findings: Sequence[Mapping[str, object]],
    subgroups: Sequence[Mapping[str, object]],
    *,
    require_gemini: bool = False,
) -> str:
    """Generate an evidence-grounded narrative or return a deterministic fallback.

    ``require_gemini`` is intended for smoke tests and deployment checks. Normal
    application execution keeps the deterministic fallback for provider outages.
    """

    return generate_narrative_result(
        metric_name,
        findings,
        subgroups,
        require_gemini=require_gemini,
    ).narrative
