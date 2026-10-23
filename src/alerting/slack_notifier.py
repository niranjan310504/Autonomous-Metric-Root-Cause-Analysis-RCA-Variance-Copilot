"""Slack Incoming Webhook notification helpers."""

from __future__ import annotations

from collections.abc import Mapping

import requests


def _numeric(value: object) -> float:
    if isinstance(value, (int, float, str)):
        return float(value)
    return 0.0


def format_slack_alert(
    metric_name: str,
    narrative: str,
    top_finding: Mapping[str, object],
    *,
    confidence_tier: str | None = None,
    deep_dive_triggered: bool | None = None,
    deep_dive_dimension: str | None = None,
) -> str:
    """Return the exact markdown text used by the Slack webhook payload."""

    lines = [
        f"🚨 *Metric Alert: {metric_name}*\n"
        f"{narrative}\n"
        f"*Rate Effect*: {_numeric(top_finding.get('rate_effect', 0)):.2%} | "
        f"*Mix Effect*: {_numeric(top_finding.get('mix_effect', 0)):.2%}"
    ]
    if confidence_tier is not None or "confidence_tier" in top_finding:
        confidence = confidence_tier or str(top_finding.get("confidence_tier", "noise"))
        confidence_labels = {
            "high": "🟢 High Confidence",
            "moderate": "🟡 Moderate Confidence",
            "noise": "⚪ Statistical Noise",
        }
        lines.append(f"*Confidence*: {confidence_labels.get(confidence, confidence)}")
    if deep_dive_triggered is not None or deep_dive_dimension is not None:
        status = "Triggered" if deep_dive_triggered else "Not triggered"
        lines.append(f"*Autonomous deep dive*: {status}")
        if deep_dive_triggered and deep_dive_dimension:
            lines.append(f"*Cross-dimensions*: {deep_dive_dimension}")
    return "\n".join(lines)


def send_slack_alert(
    webhook_url: str,
    metric_name: str,
    narrative: str,
    top_finding: Mapping[str, object],
    *,
    confidence_tier: str | None = None,
    deep_dive_triggered: bool | None = None,
    deep_dive_dimension: str | None = None,
) -> bool:
    """Post a metric alert to Slack and report whether Slack accepted it."""

    payload = {
        "text": format_slack_alert(
            metric_name,
            narrative,
            top_finding,
            confidence_tier=confidence_tier,
            deep_dive_triggered=deep_dive_triggered,
            deep_dive_dimension=deep_dive_dimension,
        )
    }
    try:
        response = requests.post(webhook_url, json=payload, timeout=5)
    except requests.RequestException:
        return False
    return bool(response.status_code == 200)
