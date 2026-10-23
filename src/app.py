"""Streamlit UI for the autonomous metric RCA copilot."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Streamlit may execute with ``src`` rather than the repository root as the
# import root, so make both application and data packages explicit.
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from data.generate_synthetic_data import (  # noqa: E402
    generate_dataset,
    generate_timeseries_dataset,
)

from alerting.slack_notifier import format_slack_alert, send_slack_alert  # noqa: E402
from db.duckdb_manager import DuckDBManager  # noqa: E402
from engine.time_series_baseline import slice_wow_baseline, slice_yoy_baseline  # noqa: E402
from engine.variance_waterfall import conversion_rate, dimension_waterfall  # noqa: E402
from graph import run_analysis, run_timeseries_analysis  # noqa: E402

st.set_page_config(page_title="Metric RCA Copilot", layout="wide")
st.title("Autonomous Metric RCA & Variance Copilot")
st.caption("Deterministic computation, bounded SQL, and traceable agent reasoning.")

def _usable_dimensions(frame: pd.DataFrame) -> set[str]:
    return set(frame.columns) - {"timestamp", "visits", "conversions"}


def _valid_metric_frame(frame: pd.DataFrame, require_dimension: bool = False) -> bool:
    has_metrics = {"visits", "conversions"}.issubset(frame.columns)
    return has_metrics and (not require_dimension or bool(_usable_dimensions(frame)))


def _confidence_badge(finding: dict[str, Any]) -> str:
    labels = {
        "high": "🟢 High Confidence",
        "moderate": "🟡 Moderate Confidence",
        "noise": "⚪ Statistical Noise",
    }
    return labels.get(str(finding.get("confidence_tier", "noise")), "⚪ Statistical Noise")


def _safe_connection_label(connection: str) -> str:
    return f"{connection[:3]}...{connection[-3:]}" if len(connection) > 8 else "***"


def _load_database_metrics(
    manager: DuckDBManager,
    table: str,
    timestamp_column: str,
    numerator: str,
    denominator: str,
    dimensions: list[str],
) -> pd.DataFrame:
    frame = manager.query_aggregated_metric(
        table,
        [timestamp_column, *dimensions],
        numerator,
        denominator,
    )
    renamed = frame.rename(
        columns={
            timestamp_column: "timestamp",
            numerator: "conversions",
            denominator: "visits",
        }
    )
    renamed["timestamp"] = pd.to_datetime(renamed["timestamp"], errors="raise")
    return renamed.sort_values("timestamp").reset_index(drop=True)


dataset_mode = st.sidebar.radio(
    "Dataset mode",
    [
        "Preloaded Aggregate Benchmark",
        "Time-Series (WoW / YoY Slicing)",
        "Custom CSV Upload",
        "Direct Database (SQLite / Postgres)",
    ],
)
timeseries_mode = "wow"
baseline: pd.DataFrame | None = None
current: pd.DataFrame | None = None
timeseries_frame: pd.DataFrame | None = None
preview_baseline: pd.DataFrame | None = None
preview_current: pd.DataFrame | None = None
input_error: str | None = None
database_manager: DuckDBManager | None = st.session_state.get("database_manager")
database_table = ""
database_timestamp = ""
database_numerator = ""
database_denominator = ""
database_dimensions: list[str] = []

if dataset_mode == "Preloaded Aggregate Benchmark":
    baseline, current = generate_dataset()
    context_key = dataset_mode
elif dataset_mode == "Time-Series (WoW / YoY Slicing)":
    timeseries_mode = st.sidebar.radio("Comparison window", ["wow", "yoy"])
    timeseries_frame = generate_timeseries_dataset()
    slicer = slice_wow_baseline if timeseries_mode == "wow" else slice_yoy_baseline
    try:
        preview_baseline, preview_current = slicer(timeseries_frame)
        if preview_baseline.empty or preview_current.empty:
            input_error = "The selected comparison mode requires two non-empty windows."
    except (KeyError, ValueError) as exc:
        input_error = str(exc)
    context_key = f"{dataset_mode}:{timeseries_mode}"
elif dataset_mode == "Custom CSV Upload":
    upload_shape = st.sidebar.radio("CSV shape", ["Time-series CSV", "Aggregate baseline/current"])
    if upload_shape == "Time-series CSV":
        uploaded = st.sidebar.file_uploader("Upload timestamped CSV", type="csv")
        timeseries_mode = st.sidebar.radio("Comparison window", ["wow", "yoy"])
        context_key = (
            f"{dataset_mode}:timeseries:{timeseries_mode}:"
            f"{uploaded.name if uploaded else ''}"
        )
        if uploaded is not None:
            timeseries_frame = pd.read_csv(uploaded)
            if "timestamp" not in timeseries_frame:
                input_error = "Time-series CSV requires a timestamp column."
            elif not _valid_metric_frame(timeseries_frame, require_dimension=True):
                input_error = (
                    "Time-series CSV requires visits, conversions, and at least one "
                    "dimension column."
                )
            else:
                slicer = slice_wow_baseline if timeseries_mode == "wow" else slice_yoy_baseline
                try:
                    preview_baseline, preview_current = slicer(timeseries_frame)
                    if preview_baseline.empty or preview_current.empty:
                        input_error = "The selected comparison mode requires two non-empty windows."
                except (KeyError, ValueError) as exc:
                    input_error = str(exc)
    else:
        baseline_file = st.sidebar.file_uploader("Upload baseline CSV", type="csv")
        current_file = st.sidebar.file_uploader("Upload current CSV", type="csv")
        context_key = (
            f"{dataset_mode}:aggregate:{baseline_file.name if baseline_file else ''}:"
            f"{current_file.name if current_file else ''}"
        )
        if baseline_file is not None and current_file is not None:
            baseline = pd.read_csv(baseline_file)
            current = pd.read_csv(current_file)
            common_dimensions = _usable_dimensions(baseline) & _usable_dimensions(current)
            if not _valid_metric_frame(baseline) or not _valid_metric_frame(current):
                input_error = "Aggregate CSVs require visits and conversions columns."
            elif not common_dimensions:
                input_error = "Aggregate CSVs require at least one shared dimension column."
else:
    database_type = st.sidebar.selectbox("Database type", ["SQLite", "PostgreSQL"])
    connection_string = st.sidebar.text_input(
        "Connection path / URI",
        type="password" if database_type == "PostgreSQL" else "default",
    )
    if st.sidebar.button("Test Connection & Introspect"):
        if not connection_string.strip():
            st.session_state["database_error"] = "A connection path or URI is required."
        else:
            manager = None
            old_manager = st.session_state.pop("database_manager", None)
            if old_manager is not None:
                old_manager.close()
            try:
                manager = DuckDBManager()
                manager.attach_database(database_type, connection_string)
                st.session_state["database_manager"] = manager
                st.session_state["database_tables"] = manager.list_tables()
                st.session_state["database_error"] = None
                st.session_state["database_connection_label"] = _safe_connection_label(
                    connection_string
                )
            except (OSError, ValueError) as exc:
                if manager is not None:
                    manager.close()
                st.session_state["database_error"] = str(exc)
    if st.session_state.get("database_error"):
        st.sidebar.error(st.session_state["database_error"])
    database_manager = st.session_state.get("database_manager")
    tables = st.session_state.get("database_tables", [])
    if database_manager is not None and tables:
        database_table = st.sidebar.selectbox("Target table", tables)
        try:
            schema = database_manager.schema(f"remote_db.{database_table}")
            columns = [item["name"] for item in schema]
            numeric_columns = [
                item["name"]
                for item in schema
                if any(
                    token in item["type"].upper()
                    for token in ("INT", "DECIMAL", "DOUBLE", "FLOAT")
                )
            ]
            if len(columns) < 1:
                raise ValueError("selected table has no columns")
            if len(numeric_columns) < 2:
                raise ValueError("selected table requires at least two numeric columns")
            database_timestamp = st.sidebar.selectbox("Timestamp column", columns)
            database_numerator = st.sidebar.selectbox("Numerator column", numeric_columns)
            denominator_options = [item for item in numeric_columns if item != database_numerator]
            database_denominator = st.sidebar.selectbox(
                "Denominator column",
                denominator_options,
            )
            database_dimensions = st.sidebar.multiselect(
                "Dimensions",
                [
                    item
                    for item in columns
                    if item
                    not in {
                        database_timestamp,
                        database_numerator,
                        database_denominator,
                    }
                ],
            )
            timeseries_mode = st.sidebar.radio("Comparison window", ["wow", "yoy"])
            if database_dimensions:
                database_frame = _load_database_metrics(
                    database_manager,
                    f"remote_db.{database_table}",
                    database_timestamp,
                    database_numerator,
                    database_denominator,
                    database_dimensions,
                )
                timeseries_frame = database_frame
                context_key = (
                    f"{dataset_mode}:{database_table}:{database_timestamp}:"
                    f"{database_numerator}:{database_denominator}:{database_dimensions}:"
                    f"{timeseries_mode}"
                )
            else:
                input_error = "Select at least one dimension for dimensional attribution."
                context_key = f"{dataset_mode}:incomplete"
        except (IndexError, KeyError, ValueError, TypeError) as exc:
            input_error = f"Database schema could not be loaded: {exc}"
            context_key = f"{dataset_mode}:invalid"
    else:
        input_error = "Test a database connection to introspect available tables."
        context_key = f"{dataset_mode}:unconnected"

if st.session_state.get("analysis_context") != context_key:
    st.session_state.pop("analysis_result", None)
    st.session_state["analysis_context"] = context_key

if preview_baseline is not None and preview_current is not None:
    st.subheader("Comparison window preview")
    preview_columns = st.columns(2)
    preview_columns[0].metric("Baseline rows", len(preview_baseline))
    preview_columns[1].metric("Current rows", len(preview_current))
    preview_columns[0].caption(
        f"{preview_baseline['timestamp'].min()} to {preview_baseline['timestamp'].max()}"
    )
    preview_columns[1].caption(
        f"{preview_current['timestamp'].min()} to {preview_current['timestamp'].max()}"
    )
    preview_columns[0].dataframe(preview_baseline.head(10), use_container_width=True)
    preview_columns[1].dataframe(preview_current.head(10), use_container_width=True)

if input_error:
    st.warning(input_error)

if st.button("Run diagnostic", type="primary"):
    st.session_state.pop("analysis_result", None)
    st.session_state.pop("analysis_baseline", None)
    st.session_state.pop("analysis_current", None)
    if input_error:
        st.error(f"Diagnostic cannot start: {input_error}")
    else:
        with st.status("Running diagnostic...", expanded=True) as status:
            try:
                if timeseries_frame is not None:
                    analysis_result, baseline, current = run_timeseries_analysis(
                        timeseries_frame, mode=timeseries_mode
                    )
                elif baseline is not None and current is not None:
                    analysis_result = run_analysis(baseline, current)
                else:
                    raise ValueError("Select or upload a complete dataset before running analysis.")
                st.session_state["analysis_result"] = analysis_result
                st.session_state["analysis_baseline"] = baseline
                st.session_state["analysis_current"] = current
                status.update(
                    label="Diagnostic completed",
                    state="complete",
                    expanded=False,
                )
            except Exception as exc:
                status.update(
                    label="Diagnostic failed",
                    state="error",
                    expanded=True,
                )
                st.error(f"Diagnostic failed: {type(exc).__name__}: {exc}")
                with st.expander("Technical details"):
                    st.exception(exc)

result = st.session_state.get("analysis_result")
baseline = st.session_state.get("analysis_baseline", baseline)
current = st.session_state.get("analysis_current", current)
if result is not None:
    metric_columns = st.columns(2)
    metric_columns[0].metric(
        "Baseline conversion", f"{baseline.conversions.sum() / baseline.visits.sum():.2%}"
    )
    metric_columns[1].metric(
        "Current conversion", f"{current.conversions.sum() / current.visits.sum():.2%}"
    )
    if result.narrative_source == "gemini":
        st.success("Gemini narrative generated.")
    elif result.narrative_source == "deterministic_fallback":
        st.warning(
            "Gemini was unavailable, so the deterministic fallback was used."
        )
        if result.provider_error:
            st.caption(result.provider_error)
    st.write(result.narrative)
    st.write("Trace: " + " -> ".join(result.trace))

    if baseline is not None and current is not None and result.schema_profile:
        waterfall_dimension = result.schema_profile.dimensions[0]
        waterfall = dimension_waterfall(baseline, current, waterfall_dimension)
        baseline_rate = conversion_rate(baseline, "conversions", "visits")
        current_rate = conversion_rate(current, "conversions", "visits")
        net_effects = [
            sum(item.rate_effect for item in waterfall),
            sum(item.mix_effect for item in waterfall),
            sum(item.interaction_effect for item in waterfall),
        ]
        waterfall_chart = go.Figure(
            go.Waterfall(
                measure=["absolute", "relative", "relative", "relative", "total"],
                x=[
                    "Baseline Rate",
                    "Net Rate Effect",
                    "Net Mix Effect",
                    "Net Interaction Effect",
                    "Current Rate",
                ],
                y=[baseline_rate, *net_effects, current_rate],
                text=[f"{value:+.2%}" for value in [baseline_rate, *net_effects, current_rate]],
                textposition="outside",
                connector={"line": {"color": "#94a3b8"}},
            )
        )
        waterfall_chart.update_layout(
            title=f"Executive Kitsberger Waterfall ({waterfall_dimension})",
            yaxis_tickformat=".2%",
            showlegend=False,
        )
        st.plotly_chart(waterfall_chart, use_container_width=True)

    findings = [finding.model_dump() for finding in result.findings]
    deep_dive_findings = [
        finding
        for finding in findings
        if result.deep_dive_dimension
        and result.loop_count > 0
        and finding["dimension"] == result.deep_dive_dimension
    ]
    if deep_dive_findings:
        st.subheader("Cross-Dimensional Interaction Waterfall")
        st.info(
            "This drilldown was triggered automatically because significant "
            "one-dimensional findings explained less than 70% of the metric change."
        )
        deep_dive_chart = px.bar(
            deep_dive_findings,
            x="segment",
            y="contribution",
            color="confidence_tier",
            hover_data=["p_value", "rate_effect", "mix_effect", "interaction_effect"],
            title=f"Cross-dimensional contributions ({result.deep_dive_dimension})",
            labels={"p_value": "BH-adjusted p-value"},
        )
        deep_dive_chart.update_yaxes(tickformat=".2%")
        st.plotly_chart(deep_dive_chart, use_container_width=True)

    if findings:
        st.subheader("Dimensional contributions")
        finding_rows = []
        for finding in findings:
            display_finding = dict(finding)
            display_finding["BH-adjusted p-value"] = display_finding.pop("p_value")
            display_finding["Statistically significant"] = display_finding.pop(
                "is_statistically_significant"
            )
            display_finding["Confidence tier"] = display_finding.pop("confidence_tier")
            display_finding["Confidence"] = _confidence_badge(finding)
            finding_rows.append(display_finding)
        significant_count = sum(
            finding["Statistically significant"] for finding in finding_rows
        )
        st.metric(
            "Statistically significant findings",
            f"{significant_count}/{len(finding_rows)}",
        )
        st.dataframe(finding_rows, use_container_width=True)
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
                ("Interaction effect", "interaction_effect"),
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
        elif send_slack_alert(
            webhook_url,
            result.metric_name,
            result.narrative,
            findings[0],
            confidence_tier=str(findings[0].get("confidence_tier", "noise")),
            deep_dive_triggered=result.loop_count > 0,
            deep_dive_dimension=result.deep_dive_dimension,
        ):
            st.success("Slack alert sent.")
        else:
            st.error("Slack alert could not be delivered.")
    if findings:
        st.markdown("**Slack live preview**")
        st.code(
            format_slack_alert(
                result.metric_name,
                result.narrative,
                findings[0],
                confidence_tier=str(findings[0].get("confidence_tier", "noise")),
                deep_dive_triggered=result.loop_count > 0,
                deep_dive_dimension=result.deep_dive_dimension,
            ),
            language="markdown",
        )
