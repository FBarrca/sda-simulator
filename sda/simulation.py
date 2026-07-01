from __future__ import annotations

from collections.abc import Iterator
from types import GeneratorType
from typing import Any

import simpy

from sda.core import SDAModel, ScenarioBatch, ScenarioSpec
from sda.data.module import DataModule
from sda.metrics import MetricSeries, MetricStore, Recorder
from sda.tracking import MLflowTracker


class SimulationResult:
    """Access point for metrics produced by a SimPy-native simulation run."""

    def __init__(self, store: MetricStore) -> None:
        """Create a result around the metric observations in ``store``."""
        self.store = store

    def metric(self, name: str) -> MetricSeries:
        """Return the recorded observations for one metric name."""
        return self.store.metric(name)

    def __getitem__(self, name: str) -> MetricSeries:
        """Return a metric series using dictionary-style access."""
        return self.metric(name)

    def __contains__(self, name: object) -> bool:
        """Return whether a metric name was recorded."""
        return isinstance(name, str) and name in self.names()

    def names(self) -> list[str]:
        """Return metric names in the order they were first recorded."""
        return self.store.names()

    def rows(self, name: str | None = None):
        """Return raw event rows, optionally filtered by metric name."""
        if name is None:
            return self.store.rows()
        return self.metric(name).rows()

    def records(self, name: str | None = None) -> list[dict[str, Any]]:
        """Return event rows as plain dictionaries for lightweight export."""
        if name is None:
            return MetricSeries(self.store.rows()).records()
        return self.metric(name).records()

    def summary(self) -> dict[str, dict[str, float]]:
        """Return summary statistics for every recorded metric."""
        return {
            name: self.metric(name).summary()
            for name in self.names()
        }


def evaluate(
    model: SDAModel,
    data: DataModule,
    *,
    stage: str = "evaluate",
    tracking: MLflowTracker | None = None,
) -> SimulationResult:
    """Evaluate a SimPy-native model over every scenario from ``data``."""
    return Simulator(tracking=tracking).evaluate(model, data, stage=stage)


class Simulator:
    """Run SimPy-native SDA models over scenario batches."""

    def __init__(self, tracking: MLflowTracker | None = None) -> None:
        """Configure the simulator."""
        self.tracking = tracking

    def evaluate(
        self,
        model: SDAModel,
        data: DataModule,
        *,
        stage: str = "evaluate",
    ) -> SimulationResult:
        """Run ``model`` over every scenario produced by ``data``."""
        store = MetricStore()

        for batch in _prepared_batches(data, stage=stage):
            _run_batch(model=model, batch=batch, store=store)

        result = SimulationResult(store)
        if self.tracking is not None:
            self.tracking.log_result(
                result,
                params={
                    "sda.model": type(model).__name__,
                    "sda.policy": type(model.policy).__name__,
                    "sda.data": type(data).__name__,
                    "sda.stage": stage,
                },
            )
        return result


def _prepared_batches(
    data: DataModule,
    *,
    stage: str,
) -> Iterator[ScenarioBatch]:
    if not isinstance(data, DataModule):
        raise TypeError("evaluate expects a DataModule")

    data.prepare_data()
    data.setup(stage=stage)
    return data.batches(stage=stage)


def _run_batch(
    *,
    model: SDAModel,
    batch: ScenarioBatch,
    store: MetricStore,
) -> None:
    for scenario in batch.scenarios:
        _run_scenario(model=model, scenario=scenario, store=store)


def _run_scenario(
    *,
    model: SDAModel,
    scenario: ScenarioSpec,
    store: MetricStore,
) -> None:
    env = simpy.Environment()
    recorder = Recorder(store, scenario_id=scenario.scenario_id, env=env)
    state = model.build(env, scenario, recorder)

    if isinstance(state, GeneratorType):
        state = env.process(state)

    env.run(until=float(scenario.end_time))
    model.finalize(state, scenario, recorder)
    recorder.close()


__all__ = [
    "SimulationResult",
    "Simulator",
    "evaluate",
]
