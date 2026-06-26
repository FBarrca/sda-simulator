from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from sda.data import ScenarioLoader
from sda.metrics import Metric, MetricSeries, MetricSet, MetricStore
from sda.model import SDAModel, StepRecord, TrajectoryRecord


class SimulationResult:
    """Result wrapper exposing metric distributions and summaries."""

    def __init__(self, store: MetricStore) -> None:
        self.store = store

    def metric(self, name: str) -> MetricSeries:
        return self.store.metric(name)

    def summary(self) -> dict[str, dict[str, float]]:
        return {
            name: self.metric(name).summary()
            for name in self.store.names()
        }


class Simulator:
    """Generic sequential decision rollout engine."""

    def __init__(
        self,
        metrics: Iterable[Metric] | MetricSet | None = None,
        keep_history: bool = True,
    ) -> None:
        self.metrics = metrics if isinstance(metrics, MetricSet) else MetricSet(metrics)
        self.keep_history = keep_history

    def evaluate(self, model: SDAModel, scenarios: ScenarioLoader) -> SimulationResult:
        store = MetricStore()

        for batch in scenarios:
            state = model.initial_state(batch)
            total_cost = np.zeros(batch.batch_size, dtype=float)
            history: list[StepRecord] = []

            for t in range(batch.horizon):
                exogenous_t = _exogenous_at_time(batch.exogenous, t)
                decision = model.decide(state, t, history)
                next_state = model.transition(state, decision, exogenous_t, t)
                cost = _as_batch_vector(
                    model.cost(state, decision, exogenous_t, next_state, t),
                    batch.batch_size,
                    "cost",
                )
                info = model.info(state, decision, exogenous_t, next_state, cost, t)

                step = StepRecord(
                    scenario_ids=batch.scenario_ids,
                    t=t,
                    state=state,
                    decision=decision,
                    exogenous=exogenous_t,
                    next_state=next_state,
                    cost=cost,
                    info=info,
                )
                self.metrics.on_step(step, store)

                total_cost += cost
                if self.keep_history:
                    history.append(step)
                state = next_state

            trajectory = TrajectoryRecord(
                scenario_ids=batch.scenario_ids,
                total_cost=total_cost,
                final_state=state,
                steps=list(history) if self.keep_history else [],
            )
            self.metrics.on_trajectory(trajectory, store)

        return SimulationResult(store)


def _exogenous_at_time(exogenous: dict[str, Any], t: int) -> dict[str, Any]:
    step_values = {}
    for name, path in exogenous.items():
        array = np.asarray(path)
        if array.ndim < 2 or t >= array.shape[1]:
            raise ValueError(f"exogenous[{name!r}] does not contain time {t}")
        step_values[name] = array[:, t, ...]
    return step_values


def _as_batch_vector(values: Any, batch_size: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        return np.full(batch_size, float(array))
    if array.ndim != 1:
        raise ValueError(f"{name} must be scalar or one-dimensional")
    if array.shape[0] != batch_size:
        raise ValueError(
            f"{name} has length {array.shape[0]}, expected batch size {batch_size}"
        )
    return array
