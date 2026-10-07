"""Streamlit UI for the autonomous metric RCA copilot."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import plotly.express as px
import streamlit as st

# Streamlit may execute with ``src`` rather than the repository root as the
# import root, so make both application and data packages explicit.
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from data.generate_synthetic_data import generate_dataset  # noqa: E402

from alerting.slack_notifier import send_slack_alert  # noqa: E402
from graph import run_analysis  # noqa: E402

st.set_page_config(page_title="Metric RCA Copilot", layout="wide")
st.title("Autonomous Metric RCA & Variance Copilot")
st.caption("Deterministic computation, bounded SQL, and traceable agent reasoning.")

baseline, current = generate_dataset()
if st.button("Run diagnostic", type="primary"):
    try:
        analysis_result = run_analysis(baseline, current)
        st.session_state["analysis_result"] = analysis_result
    except (KeyError, ValueError, TimeoutError) as exc:
        st.error(f"Analysis failed: {exc}")

result = st.session_state.get("analysis_result")
if result is not None:
    metric_columns = st.columns(2)
    metric_columns[0].metric(
        "Baseline conversion", f"{baseline.conversions.sum() / baseline.visits.sum():.2%}"
    )
    metric_columns[1].metric(
        "Current conversion", f"{current.conversions.sum() / current.visits.sum():.2%}"
    )
    st.success(result.narrative)
    st.write("Trace: " + " -> ".join(result.trace))

    findings = [finding.model_dump() for finding in result.findings]
    if findings:
        st.subheader("Dimensional contributions")
        st.dataframe(findings, use_container_width=True)
        contribution_chart = px.bar(
            findings[:10],
            x="segment",
            y="contribution",
            color="dimension",
            title="Top dimensional contributions",
        )
        st.plotly_chart(contribution_chart, use_container_width=True)

        effect_rows = [
            {
                "segment": finding["segment"],
                "dimension": finding["dimension"],
                "effect_type": effect_type,
                "effect": finding[effect_key],
            }
            for finding in findings[:10]
            for effect_type, effect_key in (
                ("Rate effect", "rate_effect"),
                ("Mix effect", "mix_effect"),
            )
        ]
        effect_chart = px.bar(
            effect_rows,
            x="segment",
            y="effect",
            color="effect_type",
            barmode="relative",
            hover_data=["dimension"],
            title="Rate vs. mix effect breakdown",
        )
        effect_chart.update_yaxes(tickformat=".2%")
        st.plotly_chart(effect_chart, use_container_width=True)
    else:
        st.info("No dimensional findings were identified.")

    st.subheader("Decision-tree subgroup rules")
    if result.subgroups:
        st.dataframe(result.subgroups, use_container_width=True)
    else:
        st.info("No decision-tree subgroup rules were identified.")

    st.subheader("Slack alert")
    if st.button("Send Slack Alert", type="secondary"):
        webhook_url = os.getenv("SLACK_WEBHOOK_URL")
        if not webhook_url:
            st.error("SLACK_WEBHOOK_URL is not configured.")
        elif not findings:
            st.error("A Slack alert requires at least one dimensional finding.")
        elif send_slack_alert(webhook_url, result.metric_name, result.narrative, findings[0]):
            st.success("Slack alert sent.")
        else:
            st.error("Slack alert could not be delivered.")
