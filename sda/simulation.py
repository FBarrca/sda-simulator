from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np

from sda.core import (
    SDAModel,
    ScenarioBatch,
    StepRecord,
    TrajectoryRecord,
)
from sda.data.module import DataModule
from sda.metrics import (
    Metric,
    MetricRow,
    MetricSeries,
    MetricSet,
    MetricStore,
    StepCostMetric,
    TotalCostMetric,
)
from sda.tracking import MLflowTracker


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

    def __getitem__(self, name: str) -> MetricSeries:
        """Return a metric series using dictionary-style access."""
        return self.metric(name)

    def __contains__(self, name: object) -> bool:
        """Return whether a metric name was recorded."""
        return isinstance(name, str) and name in self.names()

    def names(self) -> list[str]:
        """Return metric names in the order they were first recorded."""
        return self.store.names()

    def rows(self, name: str | None = None) -> list[MetricRow]:
        """Return raw metric rows, optionally filtered by metric name."""
        if name is None:
            return self.store.rows()
        return self.metric(name).rows()

    def records(
        self,
        name: str | None = None,
    ) -> list[dict[str, float | int | str | None]]:
        """Return metric rows as plain dictionaries for lightweight export."""
        return [
            {
                "name": row.name,
                "value": row.value,
                "scenario_id": row.scenario_id,
                "t": row.t,
                "level": row.level,
            }
            for row in self.rows(name)
        ]

    def summary(self) -> dict[str, dict[str, float]]:
        """Return summary statistics for every metric in the result.

        Each key is a metric name and each value is the dictionary returned by
        ``MetricSeries.summary()``, including count, mean, standard deviation,
        percentiles, and min/max.
        """
        return {
            name: self.metric(name).summary()
            for name in self.names()
        }


def evaluate(
    model: SDAModel,
    data: DataModule,
    *,
    metrics: Iterable[Metric] | MetricSet | None = None,
    extra_metrics: Iterable[Metric] | None = None,
    keep_history: bool = True,
    stage: str = "evaluate",
    tracking: MLflowTracker | None = None,
) -> SimulationResult:
    """Evaluate a model with sensible default cost metrics.

    This is the quickest public entrypoint for a simulation run. When
    ``metrics`` is omitted, the result records ``step_cost`` and
    ``total_cost``. Pass ``extra_metrics`` to add domain metrics while keeping
    the defaults. Pass an explicit ``metrics`` iterable to control exactly what
    is logged, or pass an empty iterable to record no metrics. Pass
    ``tracking=MLflowTracker(...)`` to log the result summary to MLflow after
    the rollout finishes.
    """
    selected_metrics = _selected_metrics(metrics, extra_metrics)
    return Simulator(
        metrics=selected_metrics,
        keep_history=keep_history,
        tracking=tracking,
    ).evaluate(
        model,
        data,
        stage=stage,
    )


class Simulator:
    """Roll out sequential decision models over scenario batches.

    The simulator coordinates the standard loop: initial state, policy decision
    from the observed state, exogenous reveal, transition, cost calculation,
    optional info capture, and metric logging. It is model-agnostic; domain
    behavior lives in ``SDAModel`` and ``Policy`` subclasses.
    """

    def __init__(
        self,
        metrics: Iterable[Metric] | MetricSet | None = None,
        keep_history: bool = True,
        tracking: MLflowTracker | None = None,
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
        tracking
            Optional MLflow tracker that logs aggregate result summaries after
            evaluation completes.
        """
        self.metrics = metrics if isinstance(metrics, MetricSet) else MetricSet(metrics)
        self.keep_history = keep_history
        self.tracking = tracking

    def evaluate(
        self,
        model: SDAModel,
        data: DataModule,
        *,
        stage: str = "evaluate",
    ) -> SimulationResult:
        """Run ``model`` over every batch produced by ``data``.

        For each scenario batch, the simulator asks the model for an initial
        state and then iterates from ``t = 0`` to ``batch.horizon - 1``. At each
        period it asks the model/policy for a decision using only the current
        state, time, and completed history. It then reveals the current
        exogenous time slice to the model transition and cost hooks, records
        step metrics, and accumulates total cost. After the batch finishes,
        trajectory metrics are recorded.

        Parameters
        ----------
        model
            Sequential decision model to evaluate.
        data
            Data module that yields ``ScenarioBatch`` objects. Exogenous arrays
            must be shaped ``[batch_size, horizon, ...]`` inside each batch.

        Returns
        -------
        SimulationResult
            Queryable metrics produced during the rollout.
        """
        store = MetricStore()

        for batch in _prepared_batches(data, stage=stage):
            _rollout_batch(
                model=model,
                batch=batch,
                metrics=self.metrics,
                keep_history=self.keep_history,
                store=store,
            )

        result = SimulationResult(store)
        if self.tracking is not None:
            self.tracking.log_result(
                result,
                params={
                    "sda.model": type(model).__name__,
                    "sda.policy": type(model.policy).__name__,
                    "sda.data": type(data).__name__,
                    "sda.stage": stage,
                    "sda.keep_history": self.keep_history,
                },
            )
        return result


def _default_metrics() -> list[Metric]:
    return [StepCostMetric(), TotalCostMetric()]


def _selected_metrics(
    metrics: Iterable[Metric] | MetricSet | None,
    extra_metrics: Iterable[Metric] | None,
) -> Iterable[Metric] | MetricSet:
    if extra_metrics is None:
        return _default_metrics() if metrics is None else metrics

    selected = _default_metrics() if metrics is None else _metric_list(metrics)
    return [*selected, *extra_metrics]


def _metric_list(metrics: Iterable[Metric] | MetricSet) -> list[Metric]:
    if isinstance(metrics, MetricSet):
        return list(metrics.metrics)
    return list(metrics)


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


def _rollout_batch(
    *,
    model: SDAModel,
    batch: ScenarioBatch,
    metrics: MetricSet,
    keep_history: bool,
    store: MetricStore,
) -> None:
    state = model.initial_state(batch)
    total_cost = np.zeros(batch.batch_size, dtype=float)
    history: list[StepRecord] = []

    for t in range(batch.horizon):
        step = _rollout_step(
            model=model,
            batch=batch,
            state=state,
            t=t,
            history=history,
        )
        metrics.on_step(step, store)

        total_cost += step.cost
        if keep_history:
            history.append(step)
        state = step.next_state

    trajectory = TrajectoryRecord(
        scenario_ids=batch.scenario_ids,
        total_cost=total_cost,
        final_state=state,
        steps=list(history) if keep_history else [],
    )
    metrics.on_trajectory(trajectory, store)


def _rollout_step(
    *,
    model: SDAModel,
    batch: ScenarioBatch,
    state: Any,
    t: int,
    history: list[StepRecord],
) -> StepRecord:
    decision = model.decide(state, t, history)
    exogenous_t = _exogenous_at_time(batch.exogenous, t)
    next_state = model.transition(state, decision, exogenous_t, t)
    cost = _as_batch_vector(
        model.cost(state, decision, exogenous_t, next_state, t),
        batch.batch_size,
        "cost",
    )
    info = model.info(state, decision, exogenous_t, next_state, cost, t)

    return StepRecord(
        scenario_ids=batch.scenario_ids,
        t=t,
        state=state,
        decision=decision,
        exogenous=exogenous_t,
        next_state=next_state,
        cost=cost,
        info=info,
    )


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
