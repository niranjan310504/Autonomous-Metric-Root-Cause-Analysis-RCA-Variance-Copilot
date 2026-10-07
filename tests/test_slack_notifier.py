import pytest

from alerting.slack_notifier import send_slack_alert


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
