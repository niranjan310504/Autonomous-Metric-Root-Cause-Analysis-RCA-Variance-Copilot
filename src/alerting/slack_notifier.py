"""Slack Incoming Webhook notification helpers."""

from __future__ import annotations

from typing import Any

import requests


def send_slack_alert(
    webhook_url: str,
    metric_name: str,
    narrative: str,
    top_finding: dict[str, Any],
) -> bool:
    """Post a metric alert to Slack and report whether Slack accepted it."""

    payload = {
        "text": (
            f"🚨 *Metric Alert: {metric_name}*\n"
            f"{narrative}\n"
            f"*Rate Effect*: {float(top_finding.get('rate_effect', 0)):.2%} | "
            f"*Mix Effect*: {float(top_finding.get('mix_effect', 0)):.2%}"
        )
    }
    try:
        response = requests.post(webhook_url, json=payload, timeout=5)
    except requests.RequestException:
        return False
    return response.status_code == 200
