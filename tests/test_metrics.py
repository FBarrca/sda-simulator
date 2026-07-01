import numpy as np
import pytest
import simpy

from sda import MetricStore, Recorder


def test_metric_series_distribution_queries():
    store = MetricStore()
    store.log(
        "loss",
        [1, 2, 3, 4, 100],
        scenario_ids=[0, 1, 2, 3, 4],
        level="trajectory",
    )
    series = store.metric("loss")

    assert series.values().tolist() == [1, 2, 3, 4, 100]
    assert series.mean() == pytest.approx(22)
    assert series.std() == pytest.approx(np.std([1, 2, 3, 4, 100]))
    assert series.percentile(95) == pytest.approx(np.percentile([1, 2, 3, 4, 100], 95))
    assert series.quantile(0.95) == pytest.approx(series.percentile(95))
    assert series.cvar(0.95) == pytest.approx(100)
    assert series.summary()["p95"] == pytest.approx(np.percentile([1, 2, 3, 4, 100], 95))


def test_metric_series_filters_by_time_level_and_tag():
    store = MetricStore()
    store.log("x", [1, 2], scenario_ids=[1, 2], time=3, tags={"node": "A"})
    store.log("x", [10, 20], scenario_ids=[1, 2], time=4, tags={"node": "B"})
    store.log("x", [100, 200], scenario_ids=[1, 2], level="trajectory")

    assert store.metric("x").at_time(3).values().tolist() == [1, 2]
    assert store.metric("x").event_level().values().tolist() == [1, 2, 10, 20]
    assert store.metric("x").trajectory_level().values().tolist() == [100, 200]
    assert store.metric("x").with_tag("node", "B").values().tolist() == [10, 20]


def test_metric_series_returns_rows_and_records():
    store = MetricStore()
    store.log("x", [1, 2], scenario_ids=[1, 2], time=3, tags={"kind": "demo"})
    series = store.metric("x")

    rows = series.rows()

    assert [row.value for row in rows] == [1, 2]
    assert [row.scenario_id for row in rows] == [1, 2]
    assert [row.time for row in rows] == [3, 3]
    assert len(series) == 2
    assert series.count() == 2
    assert [row.value for row in series] == [1, 2]
    assert series.records() == [
        {
            "name": "x",
            "value": 1.0,
            "scenario_id": 1,
            "time": 3.0,
            "level": "event",
            "tags": {"kind": "demo"},
        },
        {
            "name": "x",
            "value": 2.0,
            "scenario_id": 2,
            "time": 3.0,
            "level": "event",
            "tags": {"kind": "demo"},
        },
    ]


def test_metric_series_builds_trajectory_matrix():
    store = MetricStore()
    store.log("inventory", [5, 8], scenario_ids=[20, 10], time=1)
    store.log("inventory", [7, 9], scenario_ids=[20, 10], time=0)

    scenario_ids, times, values = store.metric("inventory").to_trajectory_matrix()

    assert scenario_ids.tolist() == [10, 20]
    assert times.tolist() == [0, 1]
    assert values.tolist() == [
        [9, 8],
        [7, 5],
    ]


def test_metric_series_trajectory_matrix_handles_missing_time_steps():
    store = MetricStore()
    store.log("inventory", [5], scenario_ids=[1], time=0)
    store.log("inventory", [8], scenario_ids=[2], time=1)

    _, _, values = store.metric("inventory").to_trajectory_matrix()

    assert values[0, 0] == pytest.approx(5)
    assert np.isnan(values[0, 1])
    assert np.isnan(values[1, 0])
    assert values[1, 1] == pytest.approx(8)


def test_metric_series_trajectory_matrix_validates_event_rows():
    store = MetricStore()
    store.log("system_metric", 1, time=0)

    with pytest.raises(ValueError, match="scenario_id"):
        store.metric("system_metric").to_trajectory_matrix()

    store = MetricStore()
    store.log("inventory", [5], scenario_ids=[1], time=0)
    store.log("inventory", [6], scenario_ids=[1], time=0)
    with pytest.raises(ValueError, match="duplicate"):
        store.metric("inventory").to_trajectory_matrix()


def test_metric_store_validates_broadcasts_and_names():
    store = MetricStore()
    store.log("x", 5, scenario_ids=[10, 11])

    assert store.names() == ["x"]
    assert store.metric("x").values().tolist() == [5, 5]

    with pytest.raises(ValueError, match="received 3 values"):
        store.log("bad", [1, 2, 3], scenario_ids=[1, 2])
    with pytest.raises(ValueError, match="level"):
        store.log("bad", 1, level="bad")
    with pytest.raises(ValueError, match="scenario_ids"):
        store.log("bad", [1, 2])


def test_recorder_logs_events_and_total_cost_at_env_time():
    env = simpy.Environment()
    store = MetricStore()
    recorder = Recorder(store, scenario_id=42, env=env)

    def process():
        recorder.log("inventory", 10)
        yield env.timeout(2.5)
        recorder.cost(3)
        recorder.trajectory("final_inventory", 7)

    env.process(process())
    env.run(until=3)
    recorder.close()

    assert store.metric("inventory").records()[0]["time"] == pytest.approx(0)
    assert store.metric("cost").records()[0]["time"] == pytest.approx(2.5)
    assert store.metric("total_cost").trajectory_level().values().tolist() == [3]
    assert [record.name for record in recorder.history] == [
        "inventory",
        "cost",
        "final_inventory",
        "total_cost",
    ]


def test_recorder_append_does_not_read_store_rows(monkeypatch):
    env = simpy.Environment()
    store = MetricStore()
    recorder = Recorder(store, scenario_id=7, env=env)

    def rows_unavailable():
        raise AssertionError("Recorder append should not copy store rows")

    monkeypatch.setattr(store, "rows", rows_unavailable)

    inventory = recorder.log("inventory", 10, tags={"node": 1})
    cost = recorder.cost(3)
    final = recorder.trajectory("final_inventory", 7)
    recorder.close()

    assert [record.name for record in recorder.history] == [
        "inventory",
        "cost",
        "final_inventory",
        "total_cost",
    ]
    assert recorder.history[:3] == [inventory, cost, final]
    assert store.metric("inventory").records() == [
        {
            "name": "inventory",
            "value": 10.0,
            "scenario_id": 7,
            "time": 0.0,
            "level": "event",
            "tags": {"node": "1"},
        }
    ]
    assert store.metric("total_cost").trajectory_level().values().tolist() == [3]

    monkeypatch.undo()
    assert [record.name for record in store.rows()] == [
        "inventory",
        "cost",
        "final_inventory",
        "total_cost",
    ]
