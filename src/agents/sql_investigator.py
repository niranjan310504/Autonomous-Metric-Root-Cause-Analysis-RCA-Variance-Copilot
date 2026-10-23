"""Safe SQL investigation node."""

from collections.abc import Sequence
from dataclasses import asdict

import pandas as pd

from agents.state import Finding, RCAState
from db.duckdb_manager import DuckDBManager
from engine.statistical_significance import (
    StatisticalFinding,
    evaluate_findings_significance,
)
from engine.subgroup_miner import mine_subgroups
from engine.variance_waterfall import (
    VarianceContribution,
    conversion_rate,
    dimension_waterfall,
)

_COVERAGE_THRESHOLD = 0.70
_CROSS_DIMENSION = "__rca_cross_dimension__"


def _finding_from_assessment(
    assessment: StatisticalFinding,
    dimension: str | None = None,
) -> Finding:
    contribution = assessment.contribution
    total_contribution = (
        contribution.rate_effect
        + contribution.mix_effect
        + contribution.interaction_effect
    )
    return Finding(
        dimension=dimension or contribution.dimension,
        segment=contribution.segment,
        contribution=total_contribution,
        rate_effect=contribution.rate_effect,
        mix_effect=contribution.mix_effect,
        interaction_effect=contribution.interaction_effect,
        p_value=assessment.adjusted_p_value,
        is_statistically_significant=assessment.is_statistically_significant,
        confidence_tier=assessment.confidence_tier,
        evidence=(
            f"{contribution.current_rate:.2%} current vs "
            f"{contribution.baseline_rate:.2%} baseline; "
            f"rate effect {contribution.rate_effect:+.2%}, "
            f"mix effect {contribution.mix_effect:+.2%}, "
            f"interaction effect {contribution.interaction_effect:+.2%}; "
            f"raw p-value {assessment.raw_p_value:.4g}; "
            f"adjusted p-value {assessment.adjusted_p_value:.4g}; "
            f"statistically significant: {assessment.is_statistically_significant}; "
            f"confidence: {assessment.confidence_tier}"
        ),
    )


def _candidate_dimensions(
    state: RCAState,
    findings: Sequence[Finding],
) -> tuple[str, str] | None:
    dimensions = state.schema_profile.dimensions if state.schema_profile else []
    if len(dimensions) < 2:
        return None
    significant = [finding for finding in findings if finding.is_statistically_significant]
    ranked: list[tuple[str, float]] = []
    source = significant or list(findings)
    for dimension in dimensions:
        contribution = max(
            (
                abs(finding.contribution)
                for finding in source
                if finding.dimension == dimension
            ),
            default=0.0,
        )
        if contribution > 0.0:
            ranked.append((dimension, contribution))
    ranked.sort(key=lambda item: item[1], reverse=True)
    selected = [dimension for dimension, _ in ranked[:2]]
    if len(selected) < 2:
        selected = dimensions[:2]
    return selected[0], selected[1]


def _apply_cevr_gate(
    state: RCAState,
    findings: Sequence[Finding],
    baseline_frame: pd.DataFrame,
    current_frame: pd.DataFrame,
) -> None:
    state.requires_deep_dive = False
    state.deep_dive_dimension = None
    if state.loop_count >= 1:
        return
    dimensions = state.schema_profile.dimensions if state.schema_profile else []
    if len(dimensions) < 2:
        return
    delta = conversion_rate(current_frame, state.numerator, state.denominator) - (
        conversion_rate(baseline_frame, state.numerator, state.denominator)
    )
    if abs(delta) <= 1e-12:
        return
    significant = [finding for finding in findings if finding.is_statistically_significant]
    coverage_by_dimension = {
        dimension: max(
            abs(finding.contribution)
            for finding in significant
            if finding.dimension == dimension
        ) if any(finding.dimension == dimension for finding in significant) else 0.0
        for dimension in dimensions
    }
    coverage = max(coverage_by_dimension.values(), default=0.0) / abs(delta)
    if coverage >= _COVERAGE_THRESHOLD:
        return
    candidates = _candidate_dimensions(state, findings)
    if candidates is None:
        return
    state.requires_deep_dive = True
    state.deep_dive_dimension = " x ".join(candidates)


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
    findings = [_finding_from_assessment(assessment) for assessment in significance]
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
    _apply_cevr_gate(state, state.findings, baseline_frame, current_frame)
    state.trace.append("sql_investigator")
    return state


def investigate_cross_dimensions(
    state: RCAState,
    manager: DuckDBManager,
) -> RCAState:
    """Run one bounded 2D cross-dimensional attribution pass."""

    if not state.deep_dive_dimension:
        raise ValueError("deep_dive_dimension is required for cross-dimensional investigation")
    dimensions = state.deep_dive_dimension.split(" x ")
    if len(dimensions) != 2 or dimensions[0] == dimensions[1]:
        raise ValueError("deep_dive_dimension must contain two distinct dimensions")
    if state.schema_profile is None or any(
        dimension not in state.schema_profile.dimensions for dimension in dimensions
    ):
        raise ValueError("deep-dive dimensions must be present in the schema profile")

    baseline_frame = manager.run_read_only(f"SELECT * FROM {state.table_name}_baseline")
    current_frame = manager.run_read_only(f"SELECT * FROM {state.table_name}")
    baseline_frame = baseline_frame.copy()
    current_frame = current_frame.copy()
    for frame in (baseline_frame, current_frame):
        frame[_CROSS_DIMENSION] = (
            frame[dimensions[0]].astype("string").fillna("<NA>")
            + " x "
            + frame[dimensions[1]].astype("string").fillna("<NA>")
        )

    contributions = dimension_waterfall(
        baseline_frame,
        current_frame,
        _CROSS_DIMENSION,
        state.numerator,
        state.denominator,
    )
    significance = evaluate_findings_significance(contributions, alpha=0.05)
    label = state.deep_dive_dimension
    state.findings.extend(
        _finding_from_assessment(assessment, dimension=label)
        for assessment in significance
    )
    state.findings.sort(key=lambda finding: abs(finding.contribution), reverse=True)
    state.loop_count += 1
    state.requires_deep_dive = False
    state.trace.append("cross_dimensional_investigator")
    return state
