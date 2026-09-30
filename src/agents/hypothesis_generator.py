"""Deterministic structured hypothesis generation."""

from agents.state import Hypothesis, RCAState


def generate_hypotheses(state: RCAState) -> RCAState:
    """Prioritize dimensional cuts without asking an LLM to calculate."""

    if state.schema_profile is None:
        raise ValueError("schema profile is required before hypotheses")
    state.hypotheses = [
        Hypothesis(
            dimension=dimension,
            question=f"Did conversion change materially for a segment of {dimension}?",
            priority=index + 1,
        )
        for index, dimension in enumerate(state.schema_profile.dimensions)
    ]
    state.trace.append("hypothesis_generator")
    return state
