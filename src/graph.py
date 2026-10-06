"""LangGraph orchestration entrypoint."""

from __future__ import annotations

import pandas as pd

# pyrefly: ignore [missing-import]
from langgraph.graph import END, StateGraph

from agents.hypothesis_generator import generate_hypotheses
from agents.narrative_synthesizer import synthesize_narrative
from agents.schema_inspector import inspect_schema
from agents.sql_investigator import investigate
from agents.state import RCAState
from db.duckdb_manager import DuckDBManager


def run_analysis(baseline: pd.DataFrame, current: pd.DataFrame) -> RCAState:
    """Run the complete deterministic RCA workflow."""

    state = RCAState(baseline=baseline.to_dict("records"), current=current.to_dict("records"))
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

    workflow = StateGraph(dict)

    def schema_node(values: dict[str, object]) -> dict[str, object]:
        state = RCAState.model_validate(values)
        return inspect_schema(state, manager).model_dump()

    def hypothesis_node(values: dict[str, object]) -> dict[str, object]:
        return generate_hypotheses(RCAState.model_validate(values)).model_dump()

    def investigation_node(values: dict[str, object]) -> dict[str, object]:
        return investigate(RCAState.model_validate(values), manager).model_dump()

    def narrative_node(values: dict[str, object]) -> dict[str, object]:
        return synthesize_narrative(RCAState.model_validate(values)).model_dump()

    workflow.add_node("schema", schema_node)
    workflow.add_node("hypotheses", hypothesis_node)
    workflow.add_node("investigation", investigation_node)
    workflow.add_node("narrative", narrative_node)
    workflow.set_entry_point("schema")
    workflow.add_edge("schema", "hypotheses")
    workflow.add_edge("hypotheses", "investigation")
    workflow.add_edge("investigation", "narrative")
    workflow.add_edge("narrative", END)
    return workflow.compile()
