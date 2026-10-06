"""Deterministic dimension-level variance decomposition."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class VarianceContribution:
    """Kitsberger variance components for one dimensional segment."""

    dimension: str
    segment: str
    baseline_rate: float
    current_rate: float
    baseline_weight: float
    current_weight: float
    rate_effect: float
    mix_effect: float
    interaction_effect: float
    baseline_denominator: int
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
    """Decompose current-vs-baseline rate change by dimension segment."""

    if dimension not in baseline or dimension not in current:
        raise KeyError(f"missing dimension: {dimension}")
    if numerator not in baseline or denominator not in baseline:
        raise KeyError("numerator and denominator columns are required in baseline")
    if numerator not in current or denominator not in current:
        raise KeyError("numerator and denominator columns are required in current")
    baseline_groups = baseline.groupby(dimension, dropna=False)[[numerator, denominator]].sum()
    current_groups = current.groupby(dimension, dropna=False)[[numerator, denominator]].sum()
    total_baseline_denominator = float(baseline_groups[denominator].sum())
    total_current_denominator = float(current_groups[denominator].sum())
    if total_baseline_denominator <= 0:
        raise ValueError("baseline denominator total must be positive")
    if total_current_denominator <= 0:
        raise ValueError("current denominator total must be positive")
    segments = baseline_groups.index.union(current_groups.index, sort=False)
    baseline_groups = baseline_groups.reindex(segments, fill_value=0)
    current_groups = current_groups.reindex(segments, fill_value=0)
    contributions: list[VarianceContribution] = []
    for position, (segment, baseline_row) in enumerate(baseline_groups.iterrows()):
        current_row = current_groups.iloc[position]
        baseline_denominator = float(baseline_row[denominator])
        current_denominator = float(current_row[denominator])
        baseline_rate = (
            float(baseline_row[numerator]) / baseline_denominator
            if baseline_denominator > 0
            else 0.0
        )
        current_rate = (
            float(current_row[numerator]) / current_denominator
            if current_denominator > 0
            else 0.0
        )
        baseline_weight = baseline_denominator / total_baseline_denominator
        current_weight = current_denominator / total_current_denominator
        rate_delta = current_rate - baseline_rate
        weight_delta = current_weight - baseline_weight
        contributions.append(
            VarianceContribution(
                dimension=dimension,
                segment=str(segment),
                baseline_rate=baseline_rate,
                current_rate=current_rate,
                baseline_weight=baseline_weight,
                current_weight=current_weight,
                rate_effect=baseline_weight * rate_delta,
                mix_effect=baseline_rate * weight_delta,
                interaction_effect=rate_delta * weight_delta,
                baseline_denominator=int(baseline_denominator),
                current_denominator=int(current_denominator),
            )
        )
    return sorted(
        contributions,
        key=lambda item: abs(
            item.rate_effect + item.mix_effect + item.interaction_effect
        ),
        reverse=True,
    )
