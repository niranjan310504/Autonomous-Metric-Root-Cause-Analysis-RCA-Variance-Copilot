import pytest

from alerting.slack_notifier import format_slack_alert, send_slack_alert


class FakeResponse:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


def test_send_slack_alert_posts_formatted_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        captured["url"] = url
        captured.update(kwargs)
        return FakeResponse(200)

    monkeypatch.setattr("alerting.slack_notifier.requests.post", fake_post)

    assert send_slack_alert(
        "https://hooks.slack.test/example",
        "conversion_rate",
        "Investigate the payment path.",
        {"rate_effect": -0.125, "mix_effect": 0.025},
    ) is True
    assert captured["url"] == "https://hooks.slack.test/example"
    assert captured["timeout"] == 5
    assert captured["json"] == {
        "text": (
            "🚨 *Metric Alert: conversion_rate*\n"
            "Investigate the payment path.\n"
            "*Rate Effect*: -12.50% | *Mix Effect*: 2.50%"
        )
    }


def test_format_slack_alert_matches_webhook_text() -> None:
    assert format_slack_alert(
        "conversion_rate",
        "Investigate the payment path.",
        {"rate_effect": -0.125, "mix_effect": 0.025},
    ) == (
        "🚨 *Metric Alert: conversion_rate*\n"
        "Investigate the payment path.\n"
        "*Rate Effect*: -12.50% | *Mix Effect*: 2.50%"
    )


def test_format_slack_alert_includes_confidence_and_deep_dive() -> None:
    formatted = format_slack_alert(
        "conversion_rate",
        "Investigate the payment path.",
        {"rate_effect": -0.125, "mix_effect": 0.025},
        confidence_tier="high",
        deep_dive_triggered=True,
        deep_dive_dimension="device x region",
    )

    assert "*Confidence*: 🟢 High Confidence" in formatted
    assert "*Autonomous deep dive*: Triggered" in formatted
    assert "*Cross-dimensions*: device x region" in formatted


def test_format_slack_alert_reports_deep_dive_not_triggered() -> None:
    formatted = format_slack_alert(
        "conversion_rate",
        "No interaction detected.",
        {"rate_effect": 0, "mix_effect": 0},
        confidence_tier="noise",
        deep_dive_triggered=False,
    )

    assert "*Confidence*: ⚪ Statistical Noise" in formatted
    assert "*Autonomous deep dive*: Not triggered" in formatted


def test_send_slack_alert_returns_false_for_non_200(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "alerting.slack_notifier.requests.post",
        lambda *args, **kwargs: FakeResponse(500),
    )

    assert send_slack_alert("https://hooks.slack.test/example", "metric", "text", {}) is False


def test_send_slack_alert_returns_false_for_request_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failing_post(*args: object, **kwargs: object) -> FakeResponse:
        import requests

        raise requests.RequestException("network unavailable")

    monkeypatch.setattr("alerting.slack_notifier.requests.post", failing_post)

    assert send_slack_alert("https://hooks.slack.test/example", "metric", "text", {}) is False
