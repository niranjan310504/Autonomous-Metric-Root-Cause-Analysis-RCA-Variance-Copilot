import pytest

from engine.statistical_significance import (
    evaluate_findings_significance,
    two_proportion_z_test,
)
from engine.variance_waterfall import VarianceContribution


def _contribution(
    segment: str,
    baseline_successes: int,
    baseline_trials: int,
    current_successes: int,
    current_trials: int,
) -> VarianceContribution:
    return VarianceContribution(
        dimension="device",
        segment=segment,
        baseline_rate=baseline_successes / baseline_trials if baseline_trials else 0.0,
        current_rate=current_successes / current_trials if current_trials else 0.0,
        baseline_weight=1.0,
        current_weight=1.0,
        rate_effect=0.0,
        mix_effect=0.0,
        interaction_effect=0.0,
        baseline_denominator=baseline_trials,
        current_denominator=current_trials,
    )


def test_two_proportion_z_test_matches_pooled_formula() -> None:
    result = two_proportion_z_test(10, 100, 20, 100)

    assert result.z_score == pytest.approx(1.980295086, rel=1e-6)
    assert result.raw_p_value == pytest.approx(0.047670381, rel=1e-6)
    assert result.standard_error == pytest.approx(0.050497525, rel=1e-6)


@pytest.mark.parametrize(
    ("baseline_trials", "current_trials", "expected"),
    [(0, 100, 1.0), (100, 0, 1.0)],
)
def test_two_proportion_z_test_handles_empty_segments(
    baseline_trials: int,
    current_trials: int,
    expected: float,
) -> None:
    result = two_proportion_z_test(0, baseline_trials, 0, current_trials)

    assert result.z_score == 0.0
    assert result.standard_error == 0.0
    assert result.raw_p_value == expected


@pytest.mark.parametrize(
    ("baseline_successes", "current_successes"),
    [(0, 0), (100, 100)],
)
def test_two_proportion_z_test_handles_degenerate_pooled_rates(
    baseline_successes: int,
    current_successes: int,
) -> None:
    result = two_proportion_z_test(baseline_successes, 100, current_successes, 100)

    assert result.z_score == 0.0
    assert result.standard_error == 0.0
    assert result.raw_p_value == 1.0


def test_benjamini_hochberg_assigns_significance_and_tiers() -> None:
    findings = [
        _contribution("strong", 50, 100, 80, 100),
        _contribution("moderate", 50, 100, 65, 100),
        _contribution("noise", 50, 100, 52, 100),
    ]

    results = evaluate_findings_significance(findings)
    by_segment = {result.contribution.segment: result for result in results}

    assert by_segment["strong"].is_statistically_significant
    assert by_segment["strong"].confidence_tier == "high"
    assert by_segment["moderate"].is_statistically_significant
    assert by_segment["moderate"].confidence_tier == "moderate"
    assert not by_segment["noise"].is_statistically_significant
    assert by_segment["noise"].confidence_tier == "noise"
    assert all(0.0 <= result.adjusted_p_value <= 1.0 for result in results)


def test_low_volume_zero_conversion_segment_is_noise() -> None:
    result = evaluate_findings_significance(
        [_contribution("small", 0, 3, 0, 3)]
    )[0]

    assert result.raw_p_value == 1.0
    assert result.adjusted_p_value == 1.0
    assert not result.is_statistically_significant
    assert result.confidence_tier == "noise"


def test_low_volume_significant_delta_is_noise() -> None:
    result = evaluate_findings_significance(
        [_contribution("small", 8, 29, 20, 29)]
    )[0]

    assert result.raw_p_value < 0.05
    assert not result.is_statistically_significant
    assert result.confidence_tier == "noise"


def test_benjamini_hochberg_uses_complete_hypothesis_family() -> None:
    single_result = evaluate_findings_significance(
        [_contribution("candidate", 10, 100, 20, 100)]
    )[0]
    family_results = evaluate_findings_significance(
        [
            _contribution("candidate", 10, 100, 20, 100),
            _contribution("noise-1", 50, 100, 50, 100),
            _contribution("noise-2", 50, 100, 50, 100),
            _contribution("noise-3", 50, 100, 50, 100),
        ]
    )

    assert single_result.raw_p_value == pytest.approx(0.047670381, rel=1e-6)
    assert single_result.is_statistically_significant
    candidate = next(
        result for result in family_results if result.contribution.segment == "candidate"
    )
    assert not candidate.is_statistically_significant
    assert candidate.confidence_tier == "noise"


def test_significance_rejects_invalid_alpha_and_counts() -> None:
    with pytest.raises(ValueError, match="alpha"):
        evaluate_findings_significance([], alpha=1.0)
    with pytest.raises(ValueError, match="successes cannot exceed trials"):
        two_proportion_z_test(2, 1, 0, 1)
