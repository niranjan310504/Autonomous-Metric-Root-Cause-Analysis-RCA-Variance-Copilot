"""Deterministic dimension-level variance decomposition."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class VarianceContribution:
    """Contribution of one dimensional cut to the conversion-rate delta."""

    dimension: str
    segment: str
    contribution: float
    current_rate: float
    baseline_rate: float
    current_denominator: int


def conversion_rate(frame: pd.DataFrame, numerator: str, denominator: str) -> float:
    """Calculate a conversion rate from validated numeric columns."""

    if numerator not in frame or denominator not in frame:
        raise KeyError("numerator and denominator columns are required")
    total = float(frame[denominator].sum())
    if total <= 0:
        raise ValueError("denominator total must be positive")
    return float(frame[numerator].sum()) / total


def dimension_waterfall(
    baseline: pd.DataFrame,
    current: pd.DataFrame,
    dimension: str,
    numerator: str = "conversions",
    denominator: str = "visits",
) -> list[VarianceContribution]:
    """Attribute current-vs-baseline rate change to dimension segments."""

    if dimension not in baseline or dimension not in current:
        raise KeyError(f"missing dimension: {dimension}")
    baseline_groups = baseline.groupby(dimension, dropna=False)[[numerator, denominator]].sum()
    current_groups = current.groupby(dimension, dropna=False)[[numerator, denominator]].sum()
    total_current_denominator = float(current_groups[denominator].sum())
    if total_current_denominator <= 0:
        raise ValueError("current denominator total must be positive")
    contributions: list[VarianceContribution] = []
    for segment, current_row in current_groups.iterrows():
        baseline_row = baseline_groups.loc[segment] if segment in baseline_groups.index else None
        current_rate = float(current_row[numerator]) / float(current_row[denominator])
        baseline_rate = (
            float(baseline_row[numerator]) / float(baseline_row[denominator])
            if baseline_row is not None and baseline_row[denominator] > 0
            else 0.0
        )
        contribution = (
            float(current_row[denominator])
            / total_current_denominator
            * (current_rate - baseline_rate)
        )
        contributions.append(
            VarianceContribution(
                dimension=dimension,
                segment=str(segment),
                contribution=contribution,
                current_rate=current_rate,
                baseline_rate=baseline_rate,
                current_denominator=int(current_row[denominator]),
            )
        )
    return sorted(contributions, key=lambda item: abs(item.contribution), reverse=True)
