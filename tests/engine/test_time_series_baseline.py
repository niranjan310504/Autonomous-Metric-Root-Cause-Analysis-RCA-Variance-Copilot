"""Tests for deterministic time-series baseline slicing."""

import pandas as pd
import pytest

from engine.time_series_baseline import slice_wow_baseline, slice_yoy_baseline


def test_slice_wow_baseline_uses_adjacent_seven_day_windows() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-01-01", "2026-01-07", "2026-01-08", "2026-01-14", "2026-01-15"]
            ),
            "value": [1, 2, 3, 4, 5],
        }
    )

    baseline, current = slice_wow_baseline(frame)

    assert baseline["value"].tolist() == [1, 2]
    assert current["value"].tolist() == [3, 4, 5]


def test_slice_yoy_baseline_uses_adjacent_365_day_windows() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-02", "2024-12-31", "2025-01-01", "2025-12-31", "2026-01-01"]
            ),
            "value": [1, 2, 3, 4, 5],
        }
    )

    baseline, current = slice_yoy_baseline(frame)

    assert baseline["value"].tolist() == [1, 2]
    assert current["value"].tolist() == [3, 4, 5]


def test_slicer_supports_custom_timestamp_column_without_mutating_input() -> None:
    frame = pd.DataFrame(
        {
            "event_time": ["2026-01-01", "2026-01-15"],
            "value": [1, 2],
        }
    )
    original = frame.copy(deep=True)

    baseline, current = slice_wow_baseline(frame, "event_time")

    pd.testing.assert_frame_equal(frame, original)
    assert baseline["value"].tolist() == [1]
    assert current["value"].tolist() == [2]
    assert pd.api.types.is_datetime64_any_dtype(current["event_time"])


@pytest.mark.parametrize(
    "frame", [pd.DataFrame({"value": [1]}), pd.DataFrame({"timestamp": ["invalid"]})]
)
def test_slicer_rejects_missing_or_invalid_timestamps(frame: pd.DataFrame) -> None:
    with pytest.raises((KeyError, ValueError)):
        slice_wow_baseline(frame)
