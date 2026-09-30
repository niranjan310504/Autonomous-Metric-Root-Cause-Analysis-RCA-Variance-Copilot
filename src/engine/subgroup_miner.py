"""Decision-tree subgroup discovery for conversion-rate changes."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text


@dataclass(frozen=True)
class SubgroupRule:
    """A human-readable subgroup rule and its observed conversion rate."""

    rule: str
    support: int
    conversion_rate: float
    lift_vs_overall: float


def mine_subgroups(
    frame: pd.DataFrame,
    dimensions: list[str],
    numerator: str = "conversions",
    denominator: str = "visits",
    max_depth: int = 3,
    min_samples_leaf: int = 20,
) -> list[SubgroupRule]:
    """Mine deterministic low-conversion leaves using a shallow classifier."""

    required = [*dimensions, numerator, denominator]
    if not dimensions or any(column not in frame for column in required):
        raise KeyError("all dimensions and metric columns must exist")
    rows: list[dict[str, object]] = []
    for record in frame.to_dict("records"):
        visits = int(record[denominator])
        conversions = int(record[numerator])
        rows.extend(
            [{**{column: str(record[column]) for column in dimensions}, "_outcome": 1}]
            * conversions
        )
        rows.extend(
            [{**{column: str(record[column]) for column in dimensions}, "_outcome": 0}]
            * (visits - conversions)
        )
    expanded = pd.DataFrame(rows)
    if expanded.empty:
        return []
    encoded = pd.get_dummies(expanded[dimensions], dtype=float)
    tree = DecisionTreeClassifier(
        max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=0
    )
    tree.fit(encoded, expanded["_outcome"])
    leaf_ids = tree.apply(encoded)
    leaf_rates = expanded.assign(_leaf=leaf_ids).groupby("_leaf")["_outcome"].agg(["mean", "size"])
    overall = float(expanded["_outcome"].mean())
    readable = export_text(tree, feature_names=list(encoded.columns)).splitlines()
    return [
        SubgroupRule(
            rule=line.strip(),
            support=int(leaf_rates.iloc[index]["size"]),
            conversion_rate=float(leaf_rates.iloc[index]["mean"]),
            lift_vs_overall=float(leaf_rates.iloc[index]["mean"] / overall) if overall else 0.0,
        )
        for index, line in enumerate(readable)
        if line.strip() and "|" not in line and index < len(leaf_rates)
    ]
