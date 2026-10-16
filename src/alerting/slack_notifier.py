"""Slack Incoming Webhook notification helpers."""

from __future__ import annotations

from typing import Any

import requests


def format_slack_alert(
    metric_name: str, narrative: str, top_finding: dict[str, Any]
) -> str:
    """Return the exact markdown text used by the Slack webhook payload."""

    return (
        f"🚨 *Metric Alert: {metric_name}*\n"
        f"{narrative}\n"
        f"*Rate Effect*: {float(top_finding.get('rate_effect', 0)):.2%} | "
        f"*Mix Effect*: {float(top_finding.get('mix_effect', 0)):.2%}"
    )


def send_slack_alert(
    webhook_url: str,
    metric_name: str,
    narrative: str,
    top_finding: dict[str, Any],
) -> bool:
    """Post a metric alert to Slack and report whether Slack accepted it."""

    payload = {"text": format_slack_alert(metric_name, narrative, top_finding)}
    try:
        response = requests.post(webhook_url, json=payload, timeout=5)
    except requests.RequestException:
        return False
    return response.status_code == 200
