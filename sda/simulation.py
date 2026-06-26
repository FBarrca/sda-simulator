from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from sda.data import ScenarioLoader
from sda.metrics import Metric, MetricSeries, MetricSet, MetricStore
from sda.model import SDAModel, StepRecord, TrajectoryRecord


class SimulationResult:
    """Access point for metrics produced by a simulation run.

    ``Simulator.evaluate`` returns this wrapper after all scenario batches have
    been rolled out. Use :meth:`metric` to inspect one metric distribution or
    :meth:`summary` to get summary statistics for every recorded metric.
    """

    def __init__(self, store: MetricStore) -> None:
        """Create a result around the metric observations in ``store``."""
        self.store = store

    def metric(self, name: str) -> MetricSeries:
        """Return the recorded observations for one metric name.

        Parameters
        ----------
        name
            Metric name, for example ``"step_cost"`` or ``"total_cost"``.

        Returns
        -------
        MetricSeries
            A queryable series. If no observations were recorded for ``name``,
            the series is empty and summary methods return ``nan`` values where
            appropriate.
        """
        return self.store.metric(name)

    def summary(self) -> dict[str, dict[str, float]]:
        """Return summary statistics for every metric in the result.

        Each key is a metric name and each value is the dictionary returned by
        ``MetricSeries.summary()``, including count, mean, standard deviation,
        percentiles, and min/max.
        """
        return {
            name: self.metric(name).summary()
            for name in self.store.names()
        }


class Simulator:
    """Roll out sequential decision models over scenario batches.

    The simulator coordinates the standard loop: initial state, policy
    decision, transition, cost calculation, optional info capture, and metric
    logging. It is model-agnostic; domain behavior lives in ``SDAModel`` and
    ``Policy`` subclasses.
    """

    def __init__(
        self,
        metrics: Iterable[Metric] | MetricSet | None = None,
        keep_history: bool = True,
    ) -> None:
        """Configure a simulator.

        Parameters
        ----------
        metrics
            Metrics to update during each rollout. Pass ``None`` for no metrics,
            an iterable of ``Metric`` instances, or a prebuilt ``MetricSet``.
        keep_history
            When ``True``, previous ``StepRecord`` objects are passed to the
            policy and stored on each ``TrajectoryRecord``. When ``False``, the
            policy receives an empty history list and trajectories store no
            step records, which can reduce memory use for large simulations.
        """
        self.metrics = metrics if isinstance(metrics, MetricSet) else MetricSet(metrics)
        self.keep_history = keep_history

    def evaluate(self, model: SDAModel, scenarios: ScenarioLoader) -> SimulationResult:
        """Run ``model`` over every batch produced by ``scenarios``.

        For each scenario batch, the simulator asks the model for an initial
        state and then iterates from ``t = 0`` to ``batch.horizon - 1``. At each
        period it slices exogenous values for that time, asks the model/policy
        for a decision, applies the transition, computes a cost vector, records
        step metrics, and accumulates total cost. After the batch finishes,
        trajectory metrics are recorded.

        Parameters
        ----------
        model
            Sequential decision model to evaluate.
        scenarios
            Loader that yields ``ScenarioBatch`` objects. Exogenous arrays must
            be shaped ``[batch_size, horizon, ...]`` inside each batch.

        Returns
        -------
        SimulationResult
            Queryable metrics produced during the rollout.
        """
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
    """Return the current-period slice for each exogenous path.

    Each path must have at least scenario and time dimensions. The returned
    values preserve the batch dimension and drop only the time dimension.
    """
    step_values = {}
    for name, path in exogenous.items():
        array = np.asarray(path)
        if array.ndim < 2 or t >= array.shape[1]:
            raise ValueError(f"exogenous[{name!r}] does not contain time {t}")
        step_values[name] = array[:, t, ...]
    return step_values


def _as_batch_vector(values: Any, batch_size: int, name: str) -> np.ndarray:
    """Convert scalar or per-scenario values into a float vector.

    Scalars are broadcast to ``batch_size``. One-dimensional values must already
    have length ``batch_size``; higher-dimensional values are rejected so metric
    and total-cost calculations stay aligned by scenario.
    """
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
