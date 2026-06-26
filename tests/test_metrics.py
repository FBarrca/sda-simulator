import numpy as np
import pytest

from sda import MetricStore


def test_metric_series_distribution_queries():
    store = MetricStore()
    store.log("loss", [1, 2, 3, 4, 100], scenario_ids=[0, 1, 2, 3, 4], level="trajectory")
    series = store.metric("loss")

    assert series.values().tolist() == [1, 2, 3, 4, 100]
    assert series.mean() == pytest.approx(22)
    assert series.std() == pytest.approx(np.std([1, 2, 3, 4, 100]))
    assert series.percentile(95) == pytest.approx(np.percentile([1, 2, 3, 4, 100], 95))
    assert series.quantile(0.95) == pytest.approx(series.percentile(95))
    assert series.cvar(0.95) == pytest.approx(100)
    assert series.summary()["p95"] == pytest.approx(np.percentile([1, 2, 3, 4, 100], 95))


def test_metric_series_filters_by_time_and_level():
    store = MetricStore()
    store.log("x", [1, 2], scenario_ids=[1, 2], t=3, level="step")
    store.log("x", [10, 20], scenario_ids=[1, 2], t=4, level="step")
    store.log("x", [100, 200], scenario_ids=[1, 2], level="trajectory")

    assert store.metric("x").at_time(3).values().tolist() == [1, 2]
    assert store.metric("x").step_level().values().tolist() == [1, 2, 10, 20]
    assert store.metric("x").trajectory_level().values().tolist() == [100, 200]


def test_metric_store_validates_broadcasts_and_names():
    store = MetricStore()
    store.log("x", 5, scenario_ids=[10, 11], level="step")

    assert store.names() == ["x"]
    assert store.metric("x").values().tolist() == [5, 5]

    with pytest.raises(ValueError, match="received 3 values"):
        store.log("bad", [1, 2, 3], scenario_ids=[1, 2])

    with pytest.raises(ValueError, match="level"):
        store.log("bad", 1, level="bad")
