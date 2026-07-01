import numpy as np
import pytest

import sda.simulation as simulation
from sda import (
    ArrayDataModule,
    DataModule,
    GeneratorDataModule,
    Policy,
    SDAModel,
    ScenarioBatch,
    ScenarioSpec,
    Simulator,
    evaluate,
)


class ZeroPolicy(Policy):
    def __init__(self):
        self.history_lengths = []
        self.decision_times = []

    def act(self, state, env, history):
        self.history_lengths.append(len(history))
        self.decision_times.append(float(env.now))
        return 0.0


class DemandAccumulationModel(SDAModel):
    def build(self, env, scenario, recorder):
        state = {"inventory": float(scenario.initial_state or 0.0)}
        env.process(self._run(env, scenario, recorder, state))
        return state

    def _run(self, env, scenario, recorder, state):
        for demand in np.asarray(scenario.data["demand"], dtype=float):
            decision = float(self.policy.act(state, env, recorder.history))
            state["inventory"] += decision + float(demand)
            recorder.cost(demand)
            recorder.log("inventory", state["inventory"])
            yield env.timeout(1.0)

    def finalize(self, state, scenario, recorder):
        del scenario
        recorder.trajectory("ending_inventory", state["inventory"])


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


def test_simulator_runs_simpy_model_and_logs_event_times():
    demand = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ],
        dtype=float,
    )
    policy = ZeroPolicy()
    result = Simulator().evaluate(DemandAccumulationModel(policy), make_data(demand))

    assert result.metric("total_cost").values().tolist() == [6, 15, 24]
    assert len(result.metric("cost").values()) == 9
    assert result.metric("cost").at_time(1).values().tolist() == [2, 5, 8]
    assert result.metric("ending_inventory").values().tolist() == [6, 15, 24]
    assert policy.decision_times[:3] == [0.0, 1.0, 2.0]
    assert policy.history_lengths[:3] == [0, 2, 4]


def test_evaluate_delegates_to_configured_simulator(monkeypatch):
    calls = {}
    model = DemandAccumulationModel(ZeroPolicy())
    data = make_data(np.ones((1, 2)))
    tracker = object()
    sentinel = object()

    class FakeSimulator:
        def __init__(self, tracking=None):
            calls["tracking"] = tracking

        def evaluate(self, model, data, *, stage="evaluate"):
            calls["model"] = model
            calls["data"] = data
            calls["stage"] = stage
            return sentinel

    monkeypatch.setattr(simulation, "Simulator", FakeSimulator)

    result = simulation.evaluate(model, data, stage="stress", tracking=tracker)

    assert result is sentinel
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

    data = DemandDataModule(np.array([[1, 2], [3, 4]], dtype=float))
    result = Simulator().evaluate(DemandAccumulationModel(ZeroPolicy()), data, stage="stress")

    assert data.events == ["prepare", "setup:stress", "batches:stress"]
    assert result["total_cost"].values().tolist() == [3, 7]


def test_simulator_tracking_runs_once_after_successful_evaluation():
    tracker = RecordingTracker()

    result = Simulator(tracking=tracker).evaluate(
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
    }
    assert tracker.calls[0]["tags"] == {}


def test_simulator_does_not_track_when_rollout_fails():
    class FailingModel(DemandAccumulationModel):
        def build(self, env, scenario, recorder):
            raise RuntimeError("boom")

    tracker = RecordingTracker()

    with pytest.raises(RuntimeError, match="boom"):
        Simulator(tracking=tracker).evaluate(
            FailingModel(ZeroPolicy()),
            make_data(np.ones((1, 2), dtype=float)),
        )

    assert tracker.calls == []


def test_evaluate_accepts_generator_data_module():
    def constant_demand(*, shape):
        return {"demand": np.full(shape, 2.0)}

    data = GeneratorDataModule(
        constant_demand,
        horizon=3,
        n_scenarios=4,
        initial_state=0,
        batch_size=2,
    )

    result = evaluate(DemandAccumulationModel(ZeroPolicy()), data)

    assert result["total_cost"].values().tolist() == [6, 6, 6, 6]


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


def test_simulation_result_supports_metric_collection_access_and_export():
    demand = np.array([[1, 2], [3, 4]], dtype=float)
    result = evaluate(DemandAccumulationModel(ZeroPolicy()), make_data(demand))

    assert result.names() == ["cost", "inventory", "ending_inventory", "total_cost"]
    assert "total_cost" in result
    assert "missing" not in result
    assert result["total_cost"].values().tolist() == [3, 7]
    assert [row.value for row in result.rows("total_cost")] == [3, 7]
    assert result.records("total_cost") == [
        {
            "name": "total_cost",
            "value": 3.0,
            "scenario_id": 0,
            "time": 2.0,
            "level": "trajectory",
            "tags": {},
        },
        {
            "name": "total_cost",
            "value": 7.0,
            "scenario_id": 1,
            "time": 2.0,
            "level": "trajectory",
            "tags": {},
        },
    ]
    assert len(result.records()) == 12
    assert result.summary()["total_cost"]["mean"] == pytest.approx(5)


def test_simulator_processes_generator_returned_by_build():
    class GeneratorBuildModel(SDAModel):
        def build(self, env, scenario, recorder):
            del scenario

            def process():
                yield env.timeout(0.5)
                recorder.log("ran", 1)

            return process()

    data = ArrayDataModule({"placeholder": np.zeros((1, 1))}, end_time=1)

    result = evaluate(GeneratorBuildModel(ZeroPolicy()), data)

    assert result["ran"].values().tolist() == [1]


def test_simulator_accepts_custom_scenario_specs():
    class OneScenarioData(DataModule):
        def batches(self, stage="evaluate"):
            del stage
            yield ScenarioBatch(
                [
                    ScenarioSpec(
                        scenario_id=99,
                        end_time=2,
                        initial_state=5,
                        data={"demand": np.array([1, 1])},
                    )
                ]
            )

    result = evaluate(DemandAccumulationModel(ZeroPolicy()), OneScenarioData())

    assert result["ending_inventory"].records()[0]["scenario_id"] == 99
    assert result["ending_inventory"].values().tolist() == [7]
