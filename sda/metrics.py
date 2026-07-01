from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from sda.core import StepRecord, TrajectoryRecord

MetricLevel = Literal["step", "trajectory"]


@dataclass(frozen=True)
class MetricRow:
    """One stored metric observation.

    ``scenario_id`` and ``t`` are present for normal per-scenario step metrics.
    Trajectory metrics usually have a ``scenario_id`` and no time index.
    Aggregate metrics can leave ``scenario_id`` as ``None``.
    """

    name: str
    value: float
    scenario_id: int | None
    t: int | None
    level: MetricLevel


class MetricStore:
    """In-memory store for raw metric observations."""

    def __init__(self) -> None:
        """Create an empty metric store."""
        self._rows: list[MetricRow] = []

    def log(
        self,
        name: str,
        values,
        scenario_ids: Sequence[int] | None = None,
        t: int | None = None,
        level: MetricLevel = "step",
    ) -> None:
        """Append one metric observation or one value per scenario.

        Parameters
        ----------
        name
            Metric name used later with ``SimulationResult.metric(name)``.
        values
            Scalar value or one-dimensional array-like values. Scalars are
            broadcast when ``scenario_ids`` are provided.
        scenario_ids
            Optional scenario ids aligned with ``values``.
        t
            Optional time index for step-level values.
        level
            ``"step"`` for per-period values or ``"trajectory"`` for
            whole-scenario values.
        """
        if level not in {"step", "trajectory"}:
            raise ValueError("level must be 'step' or 'trajectory'")

        ids = None if scenario_ids is None else list(scenario_ids)
        values_array = np.asarray(values, dtype=float)

        if values_array.ndim == 0:
            values_list = [float(values_array)]
            if ids is not None:
                values_list = values_list * len(ids)
        elif values_array.ndim == 1:
            values_list = [float(value) for value in values_array]
        else:
            raise ValueError("metric values must be scalar or one-dimensional")

        if ids is None:
            ids_list: list[int | None] = [None] * len(values_list)
        else:
            if len(values_list) != len(ids):
                raise ValueError(
                    f"metric {name!r} received {len(values_list)} values "
                    f"for {len(ids)} scenario_ids"
                )
            ids_list = [int(scenario_id) for scenario_id in ids]

        self._rows.extend(
            MetricRow(
                name=name,
                value=value,
                scenario_id=scenario_id,
                t=t,
                level=level,
            )
            for value, scenario_id in zip(values_list, ids_list, strict=True)
        )

    def metric(self, name: str) -> "MetricSeries":
        """Return a queryable series containing rows with ``name``."""
        return MetricSeries(row for row in self._rows if row.name == name)

    def names(self) -> list[str]:
        """Return metric names in the order they were first logged."""
        return list(dict.fromkeys(row.name for row in self._rows))

    def rows(self) -> list[MetricRow]:
        """Return a copy of all raw metric rows."""
        return list(self._rows)


class MetricSeries:
    """Queryable distribution over stored metric observations."""

    def __init__(self, rows: Iterable[MetricRow]) -> None:
        """Create a series from metric rows."""
        self._rows = tuple(rows)

    def __iter__(self) -> Iterator[MetricRow]:
        """Iterate over raw metric rows."""
        return iter(self._rows)

    def __len__(self) -> int:
        """Return the number of observations in this series."""
        return len(self._rows)

    def rows(self) -> list[MetricRow]:
        """Return a copy of the raw rows in this series."""
        return list(self._rows)

    def records(self) -> list[dict[str, float | int | str | None]]:
        """Return rows as plain dictionaries for lightweight export."""
        return [
            {
                "name": row.name,
                "value": row.value,
                "scenario_id": row.scenario_id,
                "t": row.t,
                "level": row.level,
            }
            for row in self._rows
        ]

    def values(self) -> np.ndarray:
        """Return metric values as a one-dimensional float array."""
        return np.asarray([row.value for row in self._rows], dtype=float)

    def count(self) -> int:
        """Return the number of observations in this series."""
        return len(self._rows)

    def at_time(self, t: int) -> "MetricSeries":
        """Return only observations recorded at time ``t``."""
        return MetricSeries(row for row in self._rows if row.t == t)

    def step_level(self) -> "MetricSeries":
        """Return only step-level observations."""
        return MetricSeries(row for row in self._rows if row.level == "step")

    def trajectory_level(self) -> "MetricSeries":
        """Return only trajectory-level observations."""
        return MetricSeries(row for row in self._rows if row.level == "trajectory")

    def to_trajectory_matrix(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return step observations as scenario ids, times, and a value matrix.

        The returned tuple is ``(scenario_ids, times, values)`` where
        ``values[i, j]`` is the metric value for ``scenario_ids[i]`` at
        ``times[j]``. Missing scenario-time combinations are filled with
        ``nan``.
        """
        step_rows = tuple(row for row in self._rows if row.level == "step")
        if not step_rows:
            return (
                np.asarray([], dtype=int),
                np.asarray([], dtype=int),
                np.empty((0, 0), dtype=float),
            )

        if any(row.scenario_id is None or row.t is None for row in step_rows):
            raise ValueError(
                "trajectory matrix requires scenario_id and t on every step row"
            )

        scenario_ids = np.asarray(
            sorted({int(row.scenario_id) for row in step_rows}),
            dtype=int,
        )
        times = np.asarray(
            sorted({int(row.t) for row in step_rows}),
            dtype=int,
        )
        scenario_index = {
            scenario_id: index for index, scenario_id in enumerate(scenario_ids)
        }
        time_index = {t: index for index, t in enumerate(times)}
        values = np.full((len(scenario_ids), len(times)), np.nan, dtype=float)

        seen: set[tuple[int, int]] = set()
        for row in step_rows:
            scenario_id = int(row.scenario_id)
            t = int(row.t)
            key = (scenario_id, t)
            if key in seen:
                raise ValueError(
                    "trajectory matrix received duplicate values for "
                    f"scenario_id={scenario_id}, t={t}"
                )
            seen.add(key)
            values[scenario_index[scenario_id], time_index[t]] = row.value

        return scenario_ids, times, values

    def mean(self) -> float:
        """Return the arithmetic mean, or ``nan`` for an empty series."""
        return float(np.mean(self.values())) if self._rows else float("nan")

    def std(self) -> float:
        """Return the population standard deviation, or ``nan`` if empty."""
        return float(np.std(self.values())) if self._rows else float("nan")

    def min(self) -> float:
        """Return the minimum value, or ``nan`` for an empty series."""
        return float(np.min(self.values())) if self._rows else float("nan")

    def max(self) -> float:
        """Return the maximum value, or ``nan`` for an empty series."""
        return float(np.max(self.values())) if self._rows else float("nan")

    def quantile(self, q: float) -> float:
        """Return the ``q`` quantile where ``q`` is between 0 and 1."""
        if not 0 <= q <= 1:
            raise ValueError("q must be between 0 and 1")
        return float(np.quantile(self.values(), q)) if self._rows else float("nan")

    def percentile(self, p: float) -> float:
        """Return the ``p`` percentile where ``p`` is between 0 and 100."""
        if not 0 <= p <= 100:
            raise ValueError("p must be between 0 and 100")
        return self.quantile(p / 100)

    def cvar(self, alpha: float = 0.95) -> float:
        """Return conditional value at risk for the upper tail.

        ``alpha`` is the quantile threshold. For cost metrics, this is the mean
        of the worst observations at or above that threshold.
        """
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be between 0 and 1")
        if not self._rows:
            return float("nan")

        values = self.values()
        threshold = np.quantile(values, alpha)
        tail = values[values >= threshold]
        return float(np.mean(tail))

    def summary(self) -> dict[str, float]:
        """Return common distribution summary statistics."""
        values = self.values()
        if len(values) == 0:
            return {
                "count": 0,
                "mean": float("nan"),
                "std": float("nan"),
                "min": float("nan"),
                "p50": float("nan"),
                "p90": float("nan"),
                "p95": float("nan"),
                "p99": float("nan"),
                "max": float("nan"),
            }

        return {
            "count": len(values),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "p50": float(np.percentile(values, 50)),
            "p90": float(np.percentile(values, 90)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)),
            "max": float(np.max(values)),
        }


class Metric:
    """Base class for custom simulation metrics.

    Subclasses set ``name`` and override one or both hooks. Step metrics read
    from ``StepRecord`` after each period. Trajectory metrics read from
    ``TrajectoryRecord`` after a scenario batch finishes.
    """

    name: str

    def on_step(self, step: StepRecord, store: MetricStore) -> None:
        """Record observations from one simulated step."""
        pass

    def on_trajectory(self, trajectory: TrajectoryRecord, store: MetricStore) -> None:
        """Record observations from one completed trajectory batch."""
        pass


class StepMetric(Metric):
    """Metric defined by a function over ``StepRecord`` objects."""

    def __init__(
        self,
        name: str,
        value: Callable[[StepRecord], Any],
    ) -> None:
        """Create a step metric.

        ``value`` should return a scalar or one value per scenario in
        ``step.scenario_ids``.
        """
        self.name = name
        self.value = value

    def on_step(self, step: StepRecord, store: MetricStore) -> None:
        """Record the value returned by the metric function."""
        store.log(
            name=self.name,
            values=self.value(step),
            scenario_ids=step.scenario_ids,
            t=step.t,
            level="step",
        )


class TrajectoryMetric(Metric):
    """Metric defined by a function over ``TrajectoryRecord`` objects."""

    def __init__(
        self,
        name: str,
        value: Callable[[TrajectoryRecord], Any],
    ) -> None:
        """Create a trajectory metric.

        ``value`` should return a scalar or one value per scenario in
        ``trajectory.scenario_ids``.
        """
        self.name = name
        self.value = value

    def on_trajectory(self, trajectory: TrajectoryRecord, store: MetricStore) -> None:
        """Record the value returned by the metric function."""
        store.log(
            name=self.name,
            values=self.value(trajectory),
            scenario_ids=trajectory.scenario_ids,
            level="trajectory",
        )


class InfoMetric(StepMetric):
    """Step metric that logs a value from ``StepRecord.info``."""

    def __init__(self, name: str, key: str | None = None) -> None:
        """Create an info metric.

        When ``key`` is omitted, the metric reads ``step.info[name]``.
        """
        self.key = name if key is None else key
        super().__init__(name, lambda step: step.info[self.key])


def step_metric(
    name: str,
    value: Callable[[StepRecord], Any],
) -> StepMetric:
    """Create a metric from a function called after each simulated step."""
    return StepMetric(name, value)


def trajectory_metric(
    name: str,
    value: Callable[[TrajectoryRecord], Any],
) -> TrajectoryMetric:
    """Create a metric from a function called after each trajectory batch."""
    return TrajectoryMetric(name, value)


def info_metric(name: str, key: str | None = None) -> InfoMetric:
    """Create a step metric that logs ``step.info[key or name]``."""
    return InfoMetric(name, key=key)


class MetricSet:
    """Dispatches records to a group of metrics."""

    def __init__(self, metrics: Iterable[Metric] | None = None) -> None:
        """Create a metric set and reject duplicate metric names."""
        self.metrics = list(metrics or [])
        names = [metric.name for metric in self.metrics]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate metric names: {duplicates}")

    def on_step(self, step: StepRecord, store: MetricStore) -> None:
        """Send a step record to every metric in the set."""
        for metric in self.metrics:
            metric.on_step(step, store)

    def on_trajectory(self, trajectory: TrajectoryRecord, store: MetricStore) -> None:
        """Send a trajectory record to every metric in the set."""
        for metric in self.metrics:
            metric.on_trajectory(trajectory, store)


class StepCostMetric(Metric):
    """Built-in step metric that logs the model's per-period cost vector."""

    name = "step_cost"

    def on_step(self, step: StepRecord, store: MetricStore) -> None:
        """Record ``step.cost`` for each scenario at ``step.t``."""
        store.log(
            name=self.name,
            values=step.cost,
            scenario_ids=step.scenario_ids,
            t=step.t,
            level="step",
        )


class TotalCostMetric(Metric):
    """Built-in trajectory metric that logs total cost per scenario."""

    name = "total_cost"

    def on_trajectory(self, trajectory: TrajectoryRecord, store: MetricStore) -> None:
        """Record ``trajectory.total_cost`` for each scenario."""
        store.log(
            name=self.name,
            values=trajectory.total_cost,
            scenario_ids=trajectory.scenario_ids,
            level="trajectory",
        )


__all__ = [
    "InfoMetric",
    "Metric",
    "MetricLevel",
    "MetricRow",
    "MetricSeries",
    "MetricSet",
    "MetricStore",
    "StepCostMetric",
    "StepMetric",
    "TotalCostMetric",
    "TrajectoryMetric",
    "info_metric",
    "step_metric",
    "trajectory_metric",
]
