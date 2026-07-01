import numpy as np
import pytest

from sda import (
    MetricStore,
    StepRecord,
    TrajectoryRecord,
    info_metric,
    step_metric,
    trajectory_metric,
)


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


def test_metric_series_returns_rows():
    store = MetricStore()
    store.log("x", [1, 2], scenario_ids=[1, 2], t=3, level="step")

    rows = store.metric("x").rows()

    assert [row.value for row in rows] == [1, 2]
    assert [row.scenario_id for row in rows] == [1, 2]
    assert [row.t for row in rows] == [3, 3]


def test_metric_series_supports_lightweight_export_and_counting():
    store = MetricStore()
    store.log("x", [1, 2], scenario_ids=[1, 2], t=3, level="step")
    series = store.metric("x")

    assert len(series) == 2
    assert series.count() == 2
    assert [row.value for row in series] == [1, 2]
    assert series.records() == [
        {
            "name": "x",
            "value": 1.0,
            "scenario_id": 1,
            "t": 3,
            "level": "step",
        },
        {
            "name": "x",
            "value": 2.0,
            "scenario_id": 2,
            "t": 3,
            "level": "step",
        },
    ]


def test_step_metric_factory_logs_step_values():
    metric = step_metric("inventory", lambda step: step.next_state)
    store = MetricStore()
    step = StepRecord(
        scenario_ids=[10, 11],
        t=2,
        state=np.array([3, 4]),
        decision=np.array([0, 0]),
        exogenous={},
        next_state=np.array([5, 6]),
        cost=np.array([1, 1]),
        info={},
    )

    metric.on_step(step, store)

    assert store.metric("inventory").values().tolist() == [5, 6]
    assert [row.t for row in store.metric("inventory")] == [2, 2]


def test_info_metric_logs_step_info_by_name_or_key():
    store = MetricStore()
    step = StepRecord(
        scenario_ids=[1, 2],
        t=0,
        state=None,
        decision=None,
        exogenous={},
        next_state=None,
        cost=np.array([0, 0]),
        info={
            "fill_rate": np.array([0.9, 1.0]),
            "lost_sales": np.array([0, 3]),
        },
    )

    info_metric("fill_rate").on_step(step, store)
    info_metric("stockout", key="lost_sales").on_step(step, store)

    assert store.metric("fill_rate").values().tolist() == [0.9, 1.0]
    assert store.metric("stockout").values().tolist() == [0, 3]


def test_trajectory_metric_factory_logs_trajectory_values():
    metric = trajectory_metric(
        "ending_inventory",
        lambda trajectory: trajectory.final_state,
    )
    store = MetricStore()
    trajectory = TrajectoryRecord(
        scenario_ids=[4, 5],
        total_cost=np.array([10, 20]),
        final_state=np.array([7, 8]),
        steps=[],
    )

    metric.on_trajectory(trajectory, store)

    assert store.metric("ending_inventory").values().tolist() == [7, 8]
    assert store.metric("ending_inventory").trajectory_level().count() == 2


def test_metric_series_builds_trajectory_matrix():
    store = MetricStore()
    store.log("inventory", [5, 8], scenario_ids=[20, 10], t=1, level="step")
    store.log("inventory", [7, 9], scenario_ids=[20, 10], t=0, level="step")

    scenario_ids, times, values = store.metric("inventory").to_trajectory_matrix()

    assert scenario_ids.tolist() == [10, 20]
    assert times.tolist() == [0, 1]
    assert values.tolist() == [
        [9, 8],
        [7, 5],
    ]


def test_metric_series_trajectory_matrix_handles_missing_time_steps():
    store = MetricStore()
    store.log("inventory", [5], scenario_ids=[1], t=0, level="step")
    store.log("inventory", [8], scenario_ids=[2], t=1, level="step")

    _, _, values = store.metric("inventory").to_trajectory_matrix()

    assert values[0, 0] == pytest.approx(5)
    assert np.isnan(values[0, 1])
    assert np.isnan(values[1, 0])
    assert values[1, 1] == pytest.approx(8)


def test_metric_series_trajectory_matrix_validates_labeled_step_rows():
    store = MetricStore()
    store.log("system_metric", 1, t=0, level="step")

    with pytest.raises(ValueError, match="scenario_id and t"):
        store.metric("system_metric").to_trajectory_matrix()


def test_metric_series_trajectory_matrix_rejects_duplicate_cells():
    store = MetricStore()
    store.log("inventory", [5], scenario_ids=[1], t=0, level="step")
    store.log("inventory", [6], scenario_ids=[1], t=0, level="step")

    with pytest.raises(ValueError, match="duplicate"):
        store.metric("inventory").to_trajectory_matrix()


def test_metric_store_validates_broadcasts_and_names():
    store = MetricStore()
    store.log("x", 5, scenario_ids=[10, 11], level="step")

    assert store.names() == ["x"]
    assert store.metric("x").values().tolist() == [5, 5]

    with pytest.raises(ValueError, match="received 3 values"):
        store.log("bad", [1, 2, 3], scenario_ids=[1, 2])

    with pytest.raises(ValueError, match="level"):
        store.log("bad", 1, level="bad")
