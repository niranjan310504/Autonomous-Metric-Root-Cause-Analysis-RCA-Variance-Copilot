"""Strict state contracts for the RCA graph."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SchemaProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    columns: list[dict[str, str]]
    cardinalities: dict[str, int]
    dimensions: list[str]


class Hypothesis(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dimension: str
    question: str
    priority: int = Field(ge=1)


class Finding(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dimension: str
    segment: str
    contribution: float
    rate_effect: float = 0.0
    mix_effect: float = 0.0
    evidence: str


class RCAState(BaseModel):
    model_config = ConfigDict(extra="forbid")
    metric_name: str = "conversion_rate"
    baseline: list[dict[str, Any]]
    current: list[dict[str, Any]]
    numerator: str = "conversions"
    denominator: str = "visits"
    table_name: str = "current_metrics"
    schema_profile: SchemaProfile | None = None
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    subgroups: list[dict[str, Any]] = Field(default_factory=list)
    narrative: str = ""
    trace: list[str] = Field(default_factory=list)
    error: str | None = None
