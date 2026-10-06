import pandas as pd
import pytest

from engine.variance_waterfall import conversion_rate, dimension_waterfall


def test_waterfall_identifies_rate_drop_and_preserves_rates() -> None:
    baseline = pd.DataFrame(
        {"device": ["mobile", "desktop"], "visits": [100, 100], "conversions": [10, 10]}
    )
    current = pd.DataFrame(
        {"device": ["mobile", "desktop"], "visits": [100, 100], "conversions": [5, 10]}
    )
    results = dimension_waterfall(baseline, current, "device")
    assert conversion_rate(current, "conversions", "visits") == pytest.approx(0.075)
    assert results[0].segment == "mobile"
    assert results[0].rate_effect == pytest.approx(-0.025)
    assert results[0].mix_effect == pytest.approx(0.0)
    assert results[0].interaction_effect == pytest.approx(0.0)


def test_waterfall_reconciles_rate_mix_and_interaction_effects() -> None:
    baseline = pd.DataFrame(
        {"device": ["mobile", "desktop"], "visits": [100, 100], "conversions": [10, 10]}
    )
    current = pd.DataFrame(
        {"device": ["mobile", "desktop"], "visits": [200, 100], "conversions": [10, 15]}
    )

    results = dimension_waterfall(baseline, current, "device")
    total_effect = sum(
        item.rate_effect + item.mix_effect + item.interaction_effect for item in results
    )

    assert total_effect == pytest.approx(
        conversion_rate(current, "conversions", "visits")
        - conversion_rate(baseline, "conversions", "visits")
    )
    mobile = next(item for item in results if item.segment == "mobile")
    assert mobile.rate_effect == pytest.approx(-0.025)
    assert mobile.mix_effect == pytest.approx(1 / 60)
    assert mobile.interaction_effect == pytest.approx(-1 / 120)


def test_waterfall_includes_segments_added_or_removed() -> None:
    baseline = pd.DataFrame(
        {"device": ["mobile", "desktop"], "visits": [100, 100], "conversions": [10, 10]}
    )
    current = pd.DataFrame(
        {"device": ["mobile", "tablet"], "visits": [100, 100], "conversions": [5, 20]}
    )

    results = dimension_waterfall(baseline, current, "device")

    assert {item.segment for item in results} == {"mobile", "desktop", "tablet"}
    assert next(item for item in results if item.segment == "desktop").current_denominator == 0
    assert next(item for item in results if item.segment == "tablet").baseline_denominator == 0


def test_waterfall_rejects_empty_denominator() -> None:
    with pytest.raises(ValueError, match="denominator"):
        conversion_rate(pd.DataFrame({"visits": [0], "conversions": [0]}), "conversions", "visits")
