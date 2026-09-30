"""Deterministic synthetic metric data generation."""

from collections.abc import Sequence

from .schema import MetricObservation


def generate_observations(
    metric_name: str,
    values: Sequence[float],
    dimensions: Sequence[dict[str, str]],
) -> list[MetricObservation]:
    """Build validated observations from deterministic input sequences."""

    if len(values) != len(dimensions):
        raise ValueError("values and dimensions must have the same length")

    return [
        MetricObservation(metric_name=metric_name, value=value, dimensions=dimension)
        for value, dimension in zip(values, dimensions, strict=True)
    ]
