import numpy as np
import pytest

import sda.simulation as simulation
from sda import (
    ArrayDataModule,
    DataModule,
    GeneratorDataModule,
    Metric,
    MetricStore,
    Policy,
    SDAModel,
    StepRecord,
    Simulator,
    StepCostMetric,
    TotalCostMetric,
    evaluate,
)


class ZeroPolicy(Policy):
    def act(self, state, t, history):
        return np.zeros_like(state, dtype=float)


class DemandAccumulationModel(SDAModel):
    def transition(self, state, decision, exogenous, t):
        return np.asarray(state, dtype=float) + np.asarray(exogenous["demand"], dtype=float)

    def cost(self, state, decision, exogenous, next_state, t):
        return np.asarray(exogenous["demand"], dtype=float)


class StateMetric(Metric):
    name = "state"

    def on_step(self, step: StepRecord, store: MetricStore) -> None:
        store.log(self.name, step.next_state, step.scenario_ids, step.t, "step")


class TrajectoryStepCountMetric(Metric):
    name = "trajectory_steps"

    def on_trajectory(self, trajectory, store: MetricStore) -> None:
        store.log(self.name, len(trajectory.steps), level="trajectory")


class HistoryPolicy(Policy):
    def __init__(self):
        self.history_lengths = []

    def act(self, state, t, history):
        self.history_lengths.append(len(history))
        return np.zeros_like(state, dtype=float)


class RecordingTracker:
    def __init__(self):
        self.calls = []

    def log_result(self, result, *, params=None, tags=None):
        self.calls.append(
            {
                "result": result,
                "params": dict(params or {}),
                "tags": dict(tags or {}),
            }
        )
        return "run-123"


def make_data(demand):
    return ArrayDataModule(
        {"demand": demand},
        initial_state=np.zeros(demand.shape[0]),
        batch_size=2,
    )


def test_simulator_runs_deterministic_model_and_logs_metric_shapes():
    demand = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ],
        dtype=float,
    )
    result = Simulator(metrics=[StepCostMetric(), TotalCostMetric()]).evaluate(
        DemandAccumulationModel(ZeroPolicy()),
        make_data(demand),
    )

    assert result.metric("total_cost").values().tolist() == [6, 15, 24]
    assert len(result.metric("total_cost").values()) == 3
    assert len(result.metric("step_cost").values()) == 9
    assert result.metric("step_cost").at_time(1).values().tolist() == [2, 5, 8]


def test_evaluate_delegates_to_configured_simulator(monkeypatch):
    calls = {}
    model = DemandAccumulationModel(ZeroPolicy())
    data = make_data(np.ones((1, 2)))
    tracker = object()
    sentinel = object()

    class FakeSimulator:
        def __init__(self, metrics=None, keep_history=True, tracking=None):
            calls["metrics"] = metrics
            calls["keep_history"] = keep_history
            calls["tracking"] = tracking

        def evaluate(self, model, data, *, stage="evaluate"):
            calls["model"] = model
            calls["data"] = data
            calls["stage"] = stage
            return sentinel

    monkeypatch.setattr(simulation, "Simulator", FakeSimulator)

    result = simulation.evaluate(
        model,
        data,
        keep_history=False,
        stage="stress",
        tracking=tracker,
    )

    assert result is sentinel
    assert [type(metric) for metric in calls["metrics"]] == [
        StepCostMetric,
        TotalCostMetric,
    ]
    assert calls["keep_history"] is False
    assert calls["tracking"] is tracker
    assert calls["model"] is model
    assert calls["data"] is data
    assert calls["stage"] == "stress"


def test_simulator_owns_data_lifecycle_with_stage():
    class DemandDataModule(DataModule):
        def __init__(self, demand):
            self.demand = demand
            self.events = []

        def prepare_data(self) -> None:
            self.events.append("prepare")

        def setup(self, stage=None) -> None:
            self.events.append(f"setup:{stage}")

        def batches(self, stage="evaluate"):
            self.events.append(f"batches:{stage}")
            yield from ArrayDataModule(
                {"demand": self.demand},
                initial_state=np.zeros(self.demand.shape[0]),
                batch_size=2,
            ).batches(stage=stage)

    demand = np.array(
        [
            [1, 2],
            [3, 4],
        ],
        dtype=float,
    )
    data = DemandDataModule(demand)

    result = Simulator(metrics=[TotalCostMetric()]).evaluate(
        DemandAccumulationModel(ZeroPolicy()),
        data,
        stage="stress",
    )

    assert data.events == ["prepare", "setup:stress", "batches:stress"]
    assert result["total_cost"].values().tolist() == [3, 7]


def test_simulator_keep_history_controls_policy_history_and_trajectory_steps():
    demand = np.ones((1, 3), dtype=float)

    with_history_policy = HistoryPolicy()
    with_history = Simulator(
        metrics=[TotalCostMetric(), TrajectoryStepCountMetric()],
        keep_history=True,
    ).evaluate(DemandAccumulationModel(with_history_policy), make_data(demand))

    without_history_policy = HistoryPolicy()
    without_history = Simulator(
        metrics=[TotalCostMetric(), TrajectoryStepCountMetric()],
        keep_history=False,
    ).evaluate(DemandAccumulationModel(without_history_policy), make_data(demand))

    assert with_history_policy.history_lengths == [0, 1, 2]
    assert without_history_policy.history_lengths == [0, 0, 0]
    assert with_history["trajectory_steps"].values().tolist() == [3]
    assert without_history["trajectory_steps"].values().tolist() == [0]
    assert with_history["total_cost"].values().tolist() == [3]
    assert without_history["total_cost"].values().tolist() == [3]


def test_simulator_tracking_runs_once_after_successful_evaluation():
    tracker = RecordingTracker()

    result = Simulator(
        metrics=[TotalCostMetric()],
        keep_history=False,
        tracking=tracker,
    ).evaluate(
        DemandAccumulationModel(ZeroPolicy()),
        make_data(np.ones((2, 2), dtype=float)),
        stage="stress",
    )

    assert len(tracker.calls) == 1
    assert tracker.calls[0]["result"] is result
    assert tracker.calls[0]["params"] == {
        "sda.model": "DemandAccumulationModel",
        "sda.policy": "ZeroPolicy",
        "sda.data": "ArrayDataModule",
        "sda.stage": "stress",
        "sda.keep_history": False,
    }
    assert tracker.calls[0]["tags"] == {}


def test_simulator_does_not_track_when_rollout_fails():
    class FailingModel(DemandAccumulationModel):
        def transition(self, state, decision, exogenous, t):
            raise RuntimeError("boom")

    tracker = RecordingTracker()

    with pytest.raises(RuntimeError, match="boom"):
        Simulator(metrics=[TotalCostMetric()], tracking=tracker).evaluate(
            FailingModel(ZeroPolicy()),
            make_data(np.ones((1, 2), dtype=float)),
        )

    assert tracker.calls == []


def test_data_modules_models_and_metrics_are_swappable():
    first_data = make_data(np.ones((2, 2)))
    second_data = make_data(np.full((2, 3), 2.0))
    model = DemandAccumulationModel(ZeroPolicy())

    only_total = Simulator(metrics=[TotalCostMetric()]).evaluate(model, first_data)
    with_steps = Simulator(metrics=[StepCostMetric(), TotalCostMetric()]).evaluate(
        model,
        second_data,
    )

    assert only_total.metric("total_cost").values().tolist() == [2, 2]
    assert only_total.metric("step_cost").values().size == 0
    assert with_steps.metric("total_cost").values().tolist() == [6, 6]
    assert with_steps.metric("step_cost").values().tolist() == [2, 2, 2, 2, 2, 2]


def test_evaluate_uses_default_cost_metrics():
    demand = np.array(
        [
            [1, 2],
            [3, 4],
        ],
        dtype=float,
    )

    result = evaluate(DemandAccumulationModel(ZeroPolicy()), make_data(demand))

    assert result.metric("total_cost").values().tolist() == [3, 7]
    assert result.metric("step_cost").values().tolist() == [1, 3, 2, 4]


def test_evaluate_accepts_generator_data_module():
    def constant_demand(*, rng, scenario_ids, horizon):
        del rng
        return {
            "demand": np.full(
                (len(scenario_ids), horizon),
                2.0,
            )
        }

    data = GeneratorDataModule(
        constant_demand,
        horizon=3,
        n_scenarios=4,
        initial_state=0,
        batch_size=2,
    )

    result = evaluate(DemandAccumulationModel(ZeroPolicy()), data)

    assert result["total_cost"].values().tolist() == [6, 6, 6, 6]


def test_evaluate_calls_data_module_hooks_with_stage():
    class DemandDataModule(DataModule):
        def __init__(self, demand):
            self.demand = demand
            self.events = []

        def prepare_data(self) -> None:
            self.events.append("prepare")

        def setup(self, stage=None) -> None:
            self.events.append(f"setup:{stage}")

        def batches(self, stage="evaluate"):
            self.events.append(f"batches:{stage}")
            yield from ArrayDataModule(
                {"demand": self.demand},
                initial_state=np.zeros(self.demand.shape[0]),
                batch_size=2,
            ).batches(stage=stage)

    demand = np.array(
        [
            [1, 2],
            [3, 4],
        ],
        dtype=float,
    )
    data = DemandDataModule(demand)

    result = evaluate(DemandAccumulationModel(ZeroPolicy()), data, stage="stress")

    assert data.events == ["prepare", "setup:stress", "batches:stress"]
    assert result["total_cost"].values().tolist() == [3, 7]


def test_data_module_branches_on_stage_inside_batches():
    class DemandDataModule(DataModule):
        def __init__(self):
            self.events = []

        def setup(self, stage=None) -> None:
            self.events.append(f"setup:{stage}")

        def batches(self, stage="evaluate"):
            self.events.append(f"batches:{stage}")
            demand = (
                np.array([[5, 5]], dtype=float)
                if stage == "stress"
                else np.array([[2, 2], [3, 3]], dtype=float)
            )
            yield from ArrayDataModule({"demand": demand}, initial_state=0).batches()

    data = DemandDataModule()

    result = evaluate(DemandAccumulationModel(ZeroPolicy()), data, stage="stress")

    assert data.events == ["setup:stress", "batches:stress"]
    assert result["total_cost"].values().tolist() == [10]


def test_evaluate_rejects_non_data_module_inputs():
    with pytest.raises(TypeError, match="DataModule"):
        evaluate(DemandAccumulationModel(ZeroPolicy()), object())


def test_simulation_result_supports_metric_collection_access():
    demand = np.array(
        [
            [1, 2],
            [3, 4],
        ],
        dtype=float,
    )

    result = evaluate(DemandAccumulationModel(ZeroPolicy()), make_data(demand))

    assert result.names() == ["step_cost", "total_cost"]
    assert "total_cost" in result
    assert "missing" not in result
    assert result["total_cost"].values().tolist() == result.metric(
        "total_cost"
    ).values().tolist()
    assert result["total_cost"].values().tolist() == [3, 7]


def test_simulation_result_exports_raw_rows_and_records():
    demand = np.array(
        [
            [1, 2],
            [3, 4],
        ],
        dtype=float,
    )

    result = evaluate(DemandAccumulationModel(ZeroPolicy()), make_data(demand))

    rows = result.rows("total_cost")
    assert [row.value for row in rows] == [3, 7]
    assert result.records("total_cost") == [
        {
            "name": "total_cost",
            "value": 3.0,
            "scenario_id": 0,
            "t": None,
            "level": "trajectory",
        },
        {
            "name": "total_cost",
            "value": 7.0,
            "scenario_id": 1,
            "t": None,
            "level": "trajectory",
        },
    ]
    assert len(result.rows()) == 6
    assert len(result.records()) == 6


def test_evaluate_extra_metrics_extend_default_cost_metrics():
    demand = np.array(
        [
            [1, 2],
            [3, 4],
        ],
        dtype=float,
    )

    result = evaluate(
        DemandAccumulationModel(ZeroPolicy()),
        make_data(demand),
        extra_metrics=[StateMetric()],
    )

    assert result.metric("total_cost").values().tolist() == [3, 7]
    assert result.metric("step_cost").values().tolist() == [1, 3, 2, 4]
    assert result.metric("state").values().tolist() == [1, 3, 3, 7]


def test_evaluate_explicit_metrics_replace_default_cost_metrics():
    demand = np.array(
        [
            [1, 2],
            [3, 4],
        ],
        dtype=float,
    )

    result = evaluate(
        DemandAccumulationModel(ZeroPolicy()),
        make_data(demand),
        metrics=[],
    )

    assert result.metric("total_cost").values().size == 0
    assert result.metric("step_cost").values().size == 0
