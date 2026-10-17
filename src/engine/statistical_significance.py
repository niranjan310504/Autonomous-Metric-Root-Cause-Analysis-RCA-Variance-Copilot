"""Statistical significance tests for dimensional variance findings."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Literal

from scipy.stats import norm  # type: ignore[import-untyped]

from engine.variance_waterfall import VarianceContribution

ConfidenceTier = Literal["high", "moderate", "noise"]


@dataclass(frozen=True)
class ProportionTest:
    """Two-proportion z-test result for one segment."""

    z_score: float
    raw_p_value: float
    standard_error: float


@dataclass(frozen=True)
class StatisticalFinding:
    """Multiple-comparison assessment for one variance contribution."""

    contribution: VarianceContribution
    raw_p_value: float
    adjusted_p_value: float
    is_statistically_significant: bool
    confidence_tier: ConfidenceTier


def two_proportion_z_test(
    baseline_successes: float,
    baseline_trials: float,
    current_successes: float,
    current_trials: float,
) -> ProportionTest:
    """Run a two-tailed pooled two-proportion z-test."""

    values = (
        baseline_successes,
        baseline_trials,
        current_successes,
        current_trials,
    )
    if not all(isfinite(value) for value in values):
        raise ValueError("test counts must be finite")
    if any(value < 0 for value in values):
        raise ValueError("test counts cannot be negative")
    if baseline_successes > baseline_trials or current_successes > current_trials:
        raise ValueError("successes cannot exceed trials")

    if baseline_trials == 0 or current_trials == 0:
        return ProportionTest(z_score=0.0, raw_p_value=1.0, standard_error=0.0)

    pooled_rate = (baseline_successes + current_successes) / (
        baseline_trials + current_trials
    )
    if pooled_rate in (0.0, 1.0):
        return ProportionTest(z_score=0.0, raw_p_value=1.0, standard_error=0.0)

    standard_error = sqrt(
        pooled_rate
        * (1.0 - pooled_rate)
        * (1.0 / baseline_trials + 1.0 / current_trials)
    )
    baseline_rate = baseline_successes / baseline_trials
    current_rate = current_successes / current_trials
    z_score = (current_rate - baseline_rate) / standard_error
    raw_p_value = float(2.0 * norm.sf(abs(z_score)))
    return ProportionTest(
        z_score=z_score,
        raw_p_value=raw_p_value,
        standard_error=standard_error,
    )


def _confidence_tier(raw_p_value: float, significant: bool) -> ConfidenceTier:
    if not significant:
        return "noise"
    if raw_p_value <= 0.01:
        return "high"
    if raw_p_value <= 0.05:
        return "moderate"
    return "noise"


def _adjusted_p_values(raw_p_values: Sequence[float]) -> list[float]:
    count = len(raw_p_values)
    adjusted = [1.0] * count
    running_minimum = 1.0
    for index in range(count - 1, -1, -1):
        rank = index + 1
        running_minimum = min(running_minimum, raw_p_values[index] * count / rank)
        adjusted[index] = min(1.0, running_minimum)
    return adjusted


def evaluate_findings_significance(
    findings: Sequence[VarianceContribution],
    alpha: float = 0.05,
) -> list[StatisticalFinding]:
    """Apply pooled z-tests and Benjamini-Hochberg FDR to all findings."""

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1")
    if not findings:
        return []

    tests = [
        two_proportion_z_test(
            finding.baseline_rate * finding.baseline_denominator,
            finding.baseline_denominator,
            finding.current_rate * finding.current_denominator,
            finding.current_denominator,
        )
        for finding in findings
    ]
    ranked_indices = sorted(range(len(findings)), key=lambda index: tests[index].raw_p_value)
    ranked_p_values = [tests[index].raw_p_value for index in ranked_indices]
    adjusted_ranked = _adjusted_p_values(ranked_p_values)

    passing_rank = 0
    for rank, raw_p_value in enumerate(ranked_p_values, start=1):
        if raw_p_value <= rank * alpha / len(findings):
            passing_rank = rank

    results: list[StatisticalFinding | None] = [None] * len(findings)
    for rank, (index, adjusted_p_value) in enumerate(
        zip(ranked_indices, adjusted_ranked, strict=True),
        start=1,
    ):
        significant = rank <= passing_rank
        results[index] = StatisticalFinding(
            contribution=findings[index],
            raw_p_value=tests[index].raw_p_value,
            adjusted_p_value=adjusted_p_value,
            is_statistically_significant=significant,
            confidence_tier=_confidence_tier(tests[index].raw_p_value, significant),
        )

    return [result for result in results if result is not None]
