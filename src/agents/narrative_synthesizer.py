"""Executive narrative synthesis with a deterministic default."""

from agents.state import RCAState


def synthesize_narrative(state: RCAState) -> RCAState:
    """Create a concise executive diagnostic from computed findings."""

    if not state.findings:
        state.narrative = "No material dimensional driver was identified."
    else:
        top = state.findings[0]
        state.narrative = (
            f"{state.metric_name} changed from the baseline to the current period. "
            f"The strongest observed driver is {top.dimension}={top.segment}, "
            f"with {top.evidence} and an estimated contribution of {top.contribution:+.2%}."
        )
    state.trace.append("narrative_synthesizer")
    return state
