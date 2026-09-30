"""Pydantic state models shared by agent nodes."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RCAState(BaseModel):
    """State passed between deterministic analysis and reasoning nodes."""

    model_config = ConfigDict(extra="forbid")

    metric_name: str
    current_value: float
    comparison_value: float
    variance: float | None = None
    findings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
