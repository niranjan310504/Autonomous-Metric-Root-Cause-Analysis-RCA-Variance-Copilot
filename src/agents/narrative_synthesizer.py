"""Executive narrative synthesis with a deterministic default."""

from agents.llm_adapter import generate_actionable_narrative
from agents.state import RCAState


def synthesize_narrative(state: RCAState) -> RCAState:
    """Create an executive diagnostic from validated computed findings."""

    state.narrative = generate_actionable_narrative(
        metric_name=state.metric_name,
        findings=[finding.model_dump() for finding in state.findings],
        subgroups=state.subgroups,
    )
    state.trace.append("narrative_synthesizer")
    return state
