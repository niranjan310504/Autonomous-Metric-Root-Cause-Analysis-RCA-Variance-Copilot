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
    sample_weights: list[int] = []
    for record in frame.to_dict("records"):
        visits = int(record[denominator])
        conversions = int(record[numerator])
        segment = {column: str(record[column]) for column in dimensions}
        non_conversions = visits - conversions
        if conversions > 0:
            rows.append({**segment, "_outcome": 1})
            sample_weights.append(conversions)
        if non_conversions > 0:
            rows.append({**segment, "_outcome": 0})
            sample_weights.append(non_conversions)
    aggregated = pd.DataFrame(rows)
    if aggregated.empty:
        return []
    encoded = pd.get_dummies(aggregated[dimensions], dtype=float)
    total_weight = sum(sample_weights)
    min_weight_fraction_leaf = min(min_samples_leaf / total_weight, 0.5)
    tree = DecisionTreeClassifier(
        max_depth=max_depth,
        min_weight_fraction_leaf=min_weight_fraction_leaf,
        random_state=0,
    )
    tree.fit(encoded, aggregated["_outcome"], sample_weight=sample_weights)
    leaf_ids = tree.apply(encoded)
    weighted = aggregated.assign(
        _leaf=leaf_ids,
        _weight=sample_weights,
        _weighted_outcome=aggregated["_outcome"] * sample_weights,
    )
    leaf_stats = weighted.groupby("_leaf").agg(
        support=("_weight", "sum"),
        conversions=("_weighted_outcome", "sum"),
    )
    leaf_stats["conversion_rate"] = (
        leaf_stats["conversions"] / leaf_stats["support"]
    )
    overall = sum(
        weight * int(outcome)
        for weight, outcome in zip(sample_weights, aggregated["_outcome"], strict=True)
    ) / total_weight
    readable = export_text(tree, feature_names=list(encoded.columns)).splitlines()
    leaf_rules: list[str] = []
    path: list[str] = []
    for line in readable:
        if "|---" not in line:
            continue
        prefix, value = line.split("|---", maxsplit=1)
        depth = prefix.count("|   ")
        value = value.strip()
        path = path[:depth]
        if value.startswith("class:"):
            leaf_rules.append(" AND ".join(path))
        else:
            path.append(value)
    ordered_leaf_ids = sorted(leaf_stats.index)
    return [
        SubgroupRule(
            rule=rule,
            support=int(leaf_stats.loc[leaf_id, "support"]),
            conversion_rate=float(leaf_stats.loc[leaf_id, "conversion_rate"]),
            lift_vs_overall=(
                float(leaf_stats.loc[leaf_id, "conversion_rate"] / overall)
                if overall
                else 0.0
            ),
        )
        for rule, leaf_id in zip(leaf_rules, ordered_leaf_ids, strict=True)
    ]
