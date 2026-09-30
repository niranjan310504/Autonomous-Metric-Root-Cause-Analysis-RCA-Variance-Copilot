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
    assert results[0].contribution == pytest.approx(-0.025)


def test_waterfall_rejects_empty_denominator() -> None:
    with pytest.raises(ValueError, match="denominator"):
        conversion_rate(pd.DataFrame({"visits": [0], "conversions": [0]}), "conversions", "visits")
