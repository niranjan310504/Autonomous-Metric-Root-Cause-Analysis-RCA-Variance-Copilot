"""Safe SQL investigation node."""

from dataclasses import asdict

from agents.state import Finding, RCAState
from db.duckdb_manager import DuckDBManager
from engine.statistical_significance import evaluate_findings_significance
from engine.subgroup_miner import mine_subgroups
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
        significance = evaluate_findings_significance(waterfall, alpha=0.05)
        for assessment in significance[:5]:
            item = assessment.contribution
            contribution = item.rate_effect + item.mix_effect + item.interaction_effect
            findings.append(
                Finding(
                    dimension=item.dimension,
                    segment=item.segment,
                    contribution=contribution,
                    rate_effect=item.rate_effect,
                    mix_effect=item.mix_effect,
                    interaction_effect=item.interaction_effect,
                    p_value=assessment.adjusted_p_value,
                    is_statistically_significant=assessment.is_statistically_significant,
                    confidence_tier=assessment.confidence_tier,
                    evidence=(
                        f"{item.current_rate:.2%} current vs {item.baseline_rate:.2%} baseline; "
                        f"rate effect {item.rate_effect:+.2%}, "
                        f"mix effect {item.mix_effect:+.2%}, "
                        f"interaction effect {item.interaction_effect:+.2%}; "
                        f"raw p-value {assessment.raw_p_value:.4g}; "
                        f"adjusted p-value {assessment.adjusted_p_value:.4g}; "
                        f"statistically significant: {assessment.is_statistically_significant}; "
                        f"confidence: {assessment.confidence_tier}"
                    ),
                )
            )
    state.findings = sorted(findings, key=lambda finding: abs(finding.contribution), reverse=True)
    dimensions = state.schema_profile.dimensions if state.schema_profile else []
    state.subgroups = (
        [
            asdict(subgroup)
            for subgroup in mine_subgroups(
                current_frame,
                dimensions,
                state.numerator,
                state.denominator,
            )
        ]
        if dimensions
        else []
    )
    state.trace.append("sql_investigator")
    return state
