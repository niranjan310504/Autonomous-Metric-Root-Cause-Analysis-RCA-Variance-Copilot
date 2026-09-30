"""Safe SQL investigation node."""

from agents.state import Finding, RCAState
from db.duckdb_manager import DuckDBManager
from engine.variance_waterfall import dimension_waterfall


def investigate(state: RCAState, manager: DuckDBManager) -> RCAState:
    """Run bounded SQL-backed dimensional attribution."""

    baseline_frame = manager.run_read_only(f"SELECT * FROM {state.table_name}_baseline")
    current_frame = manager.run_read_only(f"SELECT * FROM {state.table_name}")
    findings: list[Finding] = []
    for hypothesis in state.hypotheses:
        waterfall = dimension_waterfall(
            baseline_frame,
            current_frame,
            hypothesis.dimension,
            state.numerator,
            state.denominator,
        )
        for item in waterfall[:5]:
            findings.append(
                Finding(
                    dimension=item.dimension,
                    segment=item.segment,
                    contribution=item.contribution,
                    evidence=(
                        f"{item.current_rate:.2%} current vs {item.baseline_rate:.2%} baseline"
                    ),
                )
            )
    state.findings = sorted(findings, key=lambda finding: abs(finding.contribution), reverse=True)
    state.trace.append("sql_investigator")
    return state
