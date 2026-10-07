"""Alert delivery integrations."""

from alerting.slack_notifier import send_slack_alert

__all__ = ["send_slack_alert"]
