"""LangGraph orchestration entrypoint."""

from __future__ import annotations

from typing import Any, TypedDict, cast

import pandas as pd

from agents.hypothesis_generator import generate_hypotheses
from agents.narrative_synthesizer import synthesize_narrative
from agents.schema_inspector import inspect_schema
from agents.sql_investigator import investigate, investigate_cross_dimensions
from agents.state import RCAState
from db.duckdb_manager import DuckDBManager
from engine.time_series_baseline import slice_wow_baseline, slice_yoy_baseline


class GraphState(TypedDict, total=False):
    """Serialized RCA state accepted by the LangGraph workflow."""

    metric_name: str
    baseline: list[dict[str, Any]]
    current: list[dict[str, Any]]
    numerator: str
    denominator: str
    table_name: str
    schema_profile: dict[str, Any] | None
    hypotheses: list[dict[str, Any]]
    findings: list[dict[str, Any]]
    subgroups: list[dict[str, Any]]
    narrative: str
    narrative_source: str
    provider_error: str | None
    trace: list[str]
    error: str | None
    requires_deep_dive: bool
    loop_count: int
    deep_dive_dimension: str | None


def run_analysis(baseline: pd.DataFrame, current: pd.DataFrame) -> RCAState:
    """Run the complete deterministic RCA workflow."""

    baseline_records = cast(list[dict[str, Any]], baseline.to_dict("records"))
    current_records = cast(list[dict[str, Any]], current.to_dict("records"))
    state = RCAState(baseline=baseline_records, current=current_records)
    with DuckDBManager() as manager:
        manager.register_dataframe("current_metrics", current)
        manager.register_dataframe("current_metrics_baseline", baseline)
        state = inspect_schema(state, manager)
        state = generate_hypotheses(state)
        state = investigate(state, manager)
        if state.requires_deep_dive and state.loop_count < 1:
            state = investigate_cross_dimensions(state, manager)
        state = synthesize_narrative(state)
    return state


def run_timeseries_analysis(
    df: pd.DataFrame, mode: str = "wow", timestamp_col: str = "timestamp"
) -> tuple[RCAState, pd.DataFrame, pd.DataFrame]:
    """Slice timestamped data and run the complete RCA workflow."""

    slicers = {
        "wow": slice_wow_baseline,
        "yoy": slice_yoy_baseline,
    }
    slicer = slicers.get(mode.lower())
    if slicer is None:
        raise ValueError("mode must be either 'wow' or 'yoy'")

    baseline, current = slicer(df, timestamp_col)
    if baseline.empty or current.empty:
        raise ValueError(f"{mode.lower()} requires two non-empty comparison windows")
    for frame_name, frame in (("baseline", baseline), ("current", current)):
        missing = {"visits", "conversions"} - set(frame.columns)
        if missing:
            raise KeyError(f"missing metric columns in {frame_name}: {sorted(missing)}")

    return run_analysis(baseline, current), baseline, current


def build_graph(manager: DuckDBManager) -> object:
    """Compile a StateGraph for callers that need node-level orchestration."""

    # Keep the common direct-analysis path independent of LangGraph's import cost.
    # pyrefly: ignore [missing-import]
    from langgraph.graph import END, StateGraph

    workflow = StateGraph(GraphState)

    def schema_node(values: GraphState) -> GraphState:
        state = RCAState.model_validate(values)
        return cast(GraphState, inspect_schema(state, manager).model_dump())

    def hypothesis_node(values: GraphState) -> GraphState:
        return cast(GraphState, generate_hypotheses(RCAState.model_validate(values)).model_dump())

    def investigation_node(values: GraphState) -> GraphState:
        return cast(GraphState, investigate(RCAState.model_validate(values), manager).model_dump())

    def cross_investigation_node(values: GraphState) -> GraphState:
        return cast(
            GraphState,
            investigate_cross_dimensions(RCAState.model_validate(values), manager).model_dump(),
        )

    def narrative_node(values: GraphState) -> GraphState:
        return cast(GraphState, synthesize_narrative(RCAState.model_validate(values)).model_dump())

    def routing_condition(values: GraphState) -> str:
        if values.get("requires_deep_dive", False) and values.get("loop_count", 0) < 1:
            return "cross_dimensional_investigation"
        return "narrative"

    workflow.add_node("schema", cast(Any, schema_node))
    workflow.add_node("hypotheses", cast(Any, hypothesis_node))
    workflow.add_node("investigation", cast(Any, investigation_node))
    workflow.add_node(
        "cross_dimensional_investigation",
        cast(Any, cross_investigation_node),
    )
    workflow.add_node("narrative", cast(Any, narrative_node))
    workflow.set_entry_point("schema")
    workflow.add_edge("schema", "hypotheses")
    workflow.add_edge("hypotheses", "investigation")
    workflow.add_conditional_edges(
        "investigation",
        routing_condition,
        {
            "cross_dimensional_investigation": "cross_dimensional_investigation",
            "narrative": "narrative",
        },
    )
    workflow.add_edge("cross_dimensional_investigation", "narrative")
    workflow.add_edge("narrative", END)
    return workflow.compile()
