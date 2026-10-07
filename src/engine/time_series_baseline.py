"""Deterministic time-series baseline slicing utilities."""

from __future__ import annotations

import pandas as pd


def _slice_baseline(
    frame: pd.DataFrame, timestamp_col: str, window: pd.Timedelta
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return adjacent baseline and current windows ending at the latest timestamp."""

    if timestamp_col not in frame:
        raise KeyError(f"missing timestamp column: {timestamp_col}")
    sliced = frame.copy()
    timestamps = pd.to_datetime(sliced[timestamp_col], errors="raise")
    if timestamps.empty or timestamps.isna().all():
        raise ValueError("timestamp column must contain at least one valid timestamp")
    sliced[timestamp_col] = timestamps
    max_date = timestamps.max()
    current_start = max_date - window
    baseline_start = max_date - (window * 2)
    current = sliced[sliced[timestamp_col] >= current_start]
    baseline = sliced[
        (sliced[timestamp_col] >= baseline_start)
        & (sliced[timestamp_col] < current_start)
    ]
    return baseline, current


def slice_wow_baseline(
    df: pd.DataFrame, timestamp_col: str = "timestamp"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Slice the preceding seven days and latest seven days from timestamped data."""

    return _slice_baseline(df, timestamp_col, pd.Timedelta(days=7))


def slice_yoy_baseline(
    df: pd.DataFrame, timestamp_col: str = "timestamp"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Slice the preceding 365 days and latest 365 days from timestamped data."""

    return _slice_baseline(df, timestamp_col, pd.Timedelta(days=365))
