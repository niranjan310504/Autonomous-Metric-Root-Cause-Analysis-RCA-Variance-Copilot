"""Tests for the timestamped benchmark data generator."""

import importlib.util
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest


def _generate_timeseries_dataset(days: int = 30) -> pd.DataFrame:
    generator_path = Path(__file__).parents[2] / "data" / "generate_synthetic_data.py"
    spec = importlib.util.spec_from_file_location("synthetic_generator", generator_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    generator = cast(Any, module).generate_timeseries_dataset
    return cast(pd.DataFrame, generator(days=days))


def test_generate_timeseries_dataset_is_reproducible_and_timestamped() -> None:
    first = _generate_timeseries_dataset()
    second = _generate_timeseries_dataset()

    pd.testing.assert_frame_equal(first, second)
    assert set(first.columns) == {
        "timestamp",
        "device",
        "channel",
        "region",
        "visits",
        "conversions",
    }
    assert first["timestamp"].nunique() == 30
    assert len(first) == 30 * 3 * 2
    assert first["visits"].gt(0).all()
    assert first["conversions"].ge(0).all()


def test_generate_timeseries_dataset_documents_final_weekend_shift() -> None:
    frame = _generate_timeseries_dataset()
    daily = frame.groupby("timestamp", as_index=False)["visits"].sum()
    mobile = frame[frame["device"] == "mobile"]
    mobile_daily = mobile.groupby("timestamp", as_index=False)["visits"].sum()
    mobile_daily["share"] = mobile_daily["visits"] / daily["visits"]

    final_two = mobile_daily.tail(2)
    earlier = mobile_daily.iloc[:-2]
    assert final_two["share"].min() > earlier["share"].max()

    mobile = mobile.copy()
    mobile["rate"] = mobile["conversions"] / mobile["visits"]
    final_rate = mobile[mobile["timestamp"] >= mobile["timestamp"].max() - pd.Timedelta(days=1)][
        "rate"
    ].mean()
    earlier_rate = mobile[mobile["timestamp"] < mobile["timestamp"].max() - pd.Timedelta(days=1)][
        "rate"
    ].mean()
    assert final_rate < earlier_rate


@pytest.mark.parametrize("days", [0, -1])
def test_generate_timeseries_dataset_rejects_non_positive_days(days: int) -> None:
    with pytest.raises(ValueError, match="days must be positive"):
        _generate_timeseries_dataset(days=days)