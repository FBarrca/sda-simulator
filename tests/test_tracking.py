import importlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from sda import ArrayDataModule, MLflowTracker, Policy, SDAModel, evaluate


class FakeMLflow:
    def __init__(self):
        self.tracking_uri = None
        self.experiment_name = None
        self.started_runs = []
        self.params = {}
        self.tags = {}
        self.metrics = {}

    def set_tracking_uri(self, tracking_uri):
        self.tracking_uri = tracking_uri

    def set_experiment(self, experiment_name):
        self.experiment_name = experiment_name

    def start_run(self, run_name=None, nested=False):
        self.started_runs.append({"run_name": run_name, "nested": nested})
        return FakeRun()

    def log_params(self, params):
        self.params.update(params)

    def set_tags(self, tags):
        self.tags.update(tags)

    def log_metrics(self, metrics):
        self.metrics.update(metrics)


class FakeRun:
    info = SimpleNamespace(run_id="run-123")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


class ZeroPolicy(Policy):
    def act(self, state, t, history):
        return np.zeros_like(state, dtype=float)


class DemandAccumulationModel(SDAModel):
    def transition(self, state, decision, exogenous, t):
        return np.asarray(state, dtype=float) + np.asarray(
            exogenous["demand"],
            dtype=float,
        )

    def cost(self, state, decision, exogenous, next_state, t):
        return np.asarray(exogenous["demand"], dtype=float)


def make_data():
    demand = np.array(
        [
            [1, 2],
            [3, 4],
        ],
        dtype=float,
    )
    return ArrayDataModule(
        {"demand": demand},
        initial_state=np.zeros(demand.shape[0]),
    )


def test_evaluate_logs_result_summary_to_mlflow(monkeypatch):
    mlflow = FakeMLflow()
    monkeypatch.setitem(sys.modules, "mlflow", mlflow)
    tracker = MLflowTracker(
        experiment_name="inventory",
        run_name="baseline",
        tracking_uri="file:///tmp/mlruns",
        params={"policy": "zero"},
        tags={"team": "analytics"},
        nested=True,
    )

    result = evaluate(
        DemandAccumulationModel(ZeroPolicy()),
        make_data(),
        tracking=tracker,
    )

    assert result["total_cost"].values().tolist() == [3, 7]
    assert mlflow.tracking_uri == "file:///tmp/mlruns"
    assert mlflow.experiment_name == "inventory"
    assert mlflow.started_runs == [{"run_name": "baseline", "nested": True}]
    assert mlflow.params == {
        "policy": "zero",
        "sda.model": "DemandAccumulationModel",
        "sda.policy": "ZeroPolicy",
        "sda.data": "ArrayDataModule",
        "sda.stage": "evaluate",
        "sda.keep_history": "True",
    }
    assert mlflow.tags == {"team": "analytics"}
    assert mlflow.metrics["total_cost.count"] == pytest.approx(2)
    assert mlflow.metrics["total_cost.mean"] == pytest.approx(5)
    assert mlflow.metrics["step_cost.count"] == pytest.approx(4)
    assert mlflow.metrics["step_cost.mean"] == pytest.approx(2.5)


def test_mlflow_tracker_logs_existing_result_and_sanitizes_metric_keys(monkeypatch):
    mlflow = FakeMLflow()
    monkeypatch.setitem(sys.modules, "mlflow", mlflow)
    tracker = MLflowTracker(metric_prefix="eval", summary_stats=["mean", "p95"])
    result = evaluate(DemandAccumulationModel(ZeroPolicy()), make_data())

    run_id = tracker.log_result(
        result,
        params={"batch size": 2},
        tags={"scenario/set": "demo"},
    )

    assert run_id == "run-123"
    assert mlflow.params == {"batch size": "2"}
    assert mlflow.tags == {"scenario/set": "demo"}
    assert sorted(mlflow.metrics) == [
        "eval.step_cost.mean",
        "eval.step_cost.p95",
        "eval.total_cost.mean",
        "eval.total_cost.p95",
    ]


def test_mlflow_tracker_raises_helpful_error_without_mlflow(monkeypatch):
    import sda.tracking as tracking

    def import_module(name):
        if name == "mlflow":
            raise ModuleNotFoundError("No module named 'mlflow'", name="mlflow")
        return importlib.import_module(name)

    monkeypatch.setattr(tracking.importlib, "import_module", import_module)
    tracker = MLflowTracker()
    result = evaluate(DemandAccumulationModel(ZeroPolicy()), make_data())

    with pytest.raises(ImportError, match="optional 'mlflow' dependency"):
        tracker.log_result(result)
