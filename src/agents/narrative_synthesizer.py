"""Executive narrative synthesis with a deterministic default."""

from agents.llm_adapter import generate_narrative_result
from agents.state import RCAState


def synthesize_narrative(state: RCAState) -> RCAState:
    """Create an executive diagnostic from validated computed findings."""

    result = generate_narrative_result(
        metric_name=state.metric_name,
        findings=[finding.model_dump() for finding in state.findings],
        subgroups=state.subgroups,
    )
    state.narrative = result.narrative
    state.narrative_source = result.source
    state.provider_error = result.provider_error
    state.trace.append("narrative_synthesizer")
    return state
