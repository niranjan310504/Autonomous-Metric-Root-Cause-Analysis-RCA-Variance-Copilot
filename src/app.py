"""Streamlit UI for the autonomous metric RCA copilot."""

from __future__ import annotations

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

from graph import run_analysis  # noqa: E402

st.set_page_config(page_title="Metric RCA Copilot", layout="wide")
st.title("Autonomous Metric RCA & Variance Copilot")
st.caption("Deterministic computation, bounded SQL, and traceable agent reasoning.")

baseline, current = generate_dataset()
if st.button("Run diagnostic", type="primary"):
    try:
        result = run_analysis(baseline, current)
        st.metric(
            "Baseline conversion", f"{baseline.conversions.sum() / baseline.visits.sum():.2%}"
        )
        st.metric("Current conversion", f"{current.conversions.sum() / current.visits.sum():.2%}")
        st.success(result.narrative)
        st.write("Trace: " + " -> ".join(result.trace))
        if result.findings:
            findings = [finding.model_dump() for finding in result.findings]
            st.dataframe(findings, use_container_width=True)
            chart = px.bar(
                findings[:10],
                x="segment",
                y="contribution",
                color="dimension",
                title="Top dimensional contributions",
            )
            st.plotly_chart(chart, use_container_width=True)
    except (KeyError, ValueError, TimeoutError) as exc:
        st.error(f"Analysis failed: {exc}")
