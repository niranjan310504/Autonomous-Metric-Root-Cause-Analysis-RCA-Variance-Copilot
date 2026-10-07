"""LangGraph orchestration entrypoint."""

from __future__ import annotations

from typing import Any, TypedDict, cast

import pandas as pd

# pyrefly: ignore [missing-import]
from langgraph.graph import END, StateGraph

from agents.hypothesis_generator import generate_hypotheses
from agents.narrative_synthesizer import synthesize_narrative
from agents.schema_inspector import inspect_schema
from agents.sql_investigator import investigate
from agents.state import RCAState
from db.duckdb_manager import DuckDBManager


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
    trace: list[str]
    error: str | None


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
        state = synthesize_narrative(state)
    return state


def build_graph(manager: DuckDBManager) -> object:
    """Compile a StateGraph for callers that need node-level orchestration."""

    workflow = StateGraph(GraphState)

    def schema_node(values: GraphState) -> GraphState:
        state = RCAState.model_validate(values)
        return cast(GraphState, inspect_schema(state, manager).model_dump())

    def hypothesis_node(values: GraphState) -> GraphState:
        return cast(GraphState, generate_hypotheses(RCAState.model_validate(values)).model_dump())

    def investigation_node(values: GraphState) -> GraphState:
        return cast(GraphState, investigate(RCAState.model_validate(values), manager).model_dump())

    def narrative_node(values: GraphState) -> GraphState:
        return cast(GraphState, synthesize_narrative(RCAState.model_validate(values)).model_dump())

    workflow.add_node("schema", cast(Any, schema_node))
    workflow.add_node("hypotheses", cast(Any, hypothesis_node))
    workflow.add_node("investigation", cast(Any, investigation_node))
    workflow.add_node("narrative", cast(Any, narrative_node))
    workflow.set_entry_point("schema")
    workflow.add_edge("schema", "hypotheses")
    workflow.add_edge("hypotheses", "investigation")
    workflow.add_edge("investigation", "narrative")
    workflow.add_edge("narrative", END)
    return workflow.compile()
