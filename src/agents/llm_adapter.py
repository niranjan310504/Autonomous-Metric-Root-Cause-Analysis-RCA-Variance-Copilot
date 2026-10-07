"""Optional Gemini narrative generation with a deterministic fallback."""

from __future__ import annotations

import importlib
import json
import logging
import os
from collections.abc import Mapping, Sequence
from typing import Any

from dotenv import load_dotenv

_DEFAULT_MODEL = "gemini-3.8-flash"
_LOGGER = logging.getLogger(__name__)

load_dotenv()


class GeminiProviderError(RuntimeError):
    """Raised when strict Gemini verification cannot reach the provider."""


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


def _generate_with_gemini(prompt: str, api_key: str) -> str:
    """Call Gemini without making the SDK a required import."""

    genai: Any = importlib.import_module("google.genai")
    client: Any = genai.Client(api_key=api_key)
    response: Any = client.models.generate_content(
        model=os.getenv("GEMINI_MODEL", _DEFAULT_MODEL),
        contents=prompt,
    )
    text = getattr(response, "text", None)
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Gemini returned an empty narrative")
    return text.strip()


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

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        if require_gemini:
            raise GeminiProviderError("GEMINI_API_KEY is not configured")
        return _fallback_narrative(metric_name, findings)

    try:
        return "[Gemini] " + _generate_with_gemini(
            _build_prompt(metric_name, findings, subgroups), api_key
        )
    except Exception as exc:
        _LOGGER.warning("Gemini narrative generation failed: %s", exc)
        if require_gemini:
            raise GeminiProviderError("Gemini narrative generation failed") from exc
        return _fallback_narrative(metric_name, findings)
