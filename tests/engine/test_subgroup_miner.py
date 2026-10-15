import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

from engine import subgroup_miner


def test_mine_subgroups_uses_weighted_leaf_statistics() -> None:
    frame = pd.DataFrame(
        [
            {"device": "ios", "conversions": 2, "visits": 10},
            {"device": "web", "conversions": 8, "visits": 10},
        ]
    )

    rules = subgroup_miner.mine_subgroups(
        frame,
        ["device"],
        max_depth=1,
        min_samples_leaf=1,
    )

    assert sorted(rule.support for rule in rules) == [10, 10]
    assert sorted(rule.conversion_rate for rule in rules) == [0.2, 0.8]
    assert sorted(rule.lift_vs_overall for rule in rules) == [0.4, 1.6]


def test_mine_subgroups_keeps_high_volume_inputs_compact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    original_fit = DecisionTreeClassifier.fit

    def spy_fit(
        estimator: DecisionTreeClassifier,
        features: pd.DataFrame,
        outcomes: pd.Series,
        sample_weight: list[int] | None = None,
    ) -> DecisionTreeClassifier:
        captured["rows"] = len(features)
        captured["weights"] = list(sample_weight) if sample_weight is not None else []
        return original_fit(
            estimator,
            features,
            outcomes,
            sample_weight=sample_weight,
        )

    monkeypatch.setattr(DecisionTreeClassifier, "fit", spy_fit)
    frame = pd.DataFrame(
        [
            {"device": "ios", "conversions": 0, "visits": 10_000_000},
            {"device": "web", "conversions": 10_000_000, "visits": 10_000_000},
        ]
    )

    rules = subgroup_miner.mine_subgroups(
        frame,
        ["device"],
        max_depth=1,
        min_samples_leaf=1,
    )

    assert len(rules) == 2
    assert captured == {"rows": 2, "weights": [10_000_000, 10_000_000]}
