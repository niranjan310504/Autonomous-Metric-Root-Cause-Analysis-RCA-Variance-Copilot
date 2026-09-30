"""Canonical data schemas used by the analytical engine."""

from pydantic import BaseModel, ConfigDict


class MetricObservation(BaseModel):
    """A single metric observation with dimensions and a value."""

    model_config = ConfigDict(extra="forbid")

    metric_name: str
    value: float
    dimensions: dict[str, str]
