"""Safe SQL investigation node."""

from dataclasses import asdict

from agents.state import Finding, RCAState
from db.duckdb_manager import DuckDBManager
from engine.statistical_significance import evaluate_findings_significance
from engine.subgroup_miner import mine_subgroups
from engine.variance_waterfall import VarianceContribution, dimension_waterfall


def investigate(state: RCAState, manager: DuckDBManager) -> RCAState:
    """Run bounded SQL-backed dimensional attribution."""

    baseline_frame = manager.run_read_only(f"SELECT * FROM {state.table_name}_baseline")
    current_frame = manager.run_read_only(f"SELECT * FROM {state.table_name}")
    contributions: list[VarianceContribution] = []
    for hypothesis in state.hypotheses:
        contributions.extend(
            dimension_waterfall(
                baseline_frame,
                current_frame,
                hypothesis.dimension,
                state.numerator,
                state.denominator,
            )
        )
    significance = evaluate_findings_significance(contributions, alpha=0.05)
    findings = [
        Finding(
            dimension=assessment.contribution.dimension,
            segment=assessment.contribution.segment,
            contribution=(
                assessment.contribution.rate_effect
                + assessment.contribution.mix_effect
                + assessment.contribution.interaction_effect
            ),
            rate_effect=assessment.contribution.rate_effect,
            mix_effect=assessment.contribution.mix_effect,
            interaction_effect=assessment.contribution.interaction_effect,
            p_value=assessment.adjusted_p_value,
            is_statistically_significant=assessment.is_statistically_significant,
            confidence_tier=assessment.confidence_tier,
            evidence=(
                f"{assessment.contribution.current_rate:.2%} current vs "
                f"{assessment.contribution.baseline_rate:.2%} baseline; "
                f"rate effect {assessment.contribution.rate_effect:+.2%}, "
                f"mix effect {assessment.contribution.mix_effect:+.2%}, "
                f"interaction effect {assessment.contribution.interaction_effect:+.2%}; "
                f"raw p-value {assessment.raw_p_value:.4g}; "
                f"adjusted p-value {assessment.adjusted_p_value:.4g}; "
                f"statistically significant: {assessment.is_statistically_significant}; "
                f"confidence: {assessment.confidence_tier}"
            ),
        )
        for assessment in significance
    ]
    state.findings = sorted(
        findings,
        key=lambda finding: abs(finding.contribution),
        reverse=True,
    )
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
