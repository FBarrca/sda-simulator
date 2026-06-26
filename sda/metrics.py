from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from sda.model import StepRecord, TrajectoryRecord

MetricLevel = Literal["step", "trajectory"]


@dataclass(frozen=True)
class MetricRow:
    name: str
    value: float
    scenario_id: int | None
    t: int | None
    level: MetricLevel


class MetricStore:
    """In-memory raw metric observation store."""

    def __init__(self) -> None:
        self._rows: list[MetricRow] = []

    def log(
        self,
        name: str,
        values,
        scenario_ids: Sequence[int] | None = None,
        t: int | None = None,
        level: MetricLevel = "step",
    ) -> None:
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
        return MetricSeries(row for row in self._rows if row.name == name)

    def names(self) -> list[str]:
        return list(dict.fromkeys(row.name for row in self._rows))

    def rows(self) -> list[MetricRow]:
        return list(self._rows)


class MetricSeries:
    """Queryable distribution over stored metric observations."""

    def __init__(self, rows: Iterable[MetricRow]) -> None:
        self._rows = tuple(rows)

    def values(self) -> np.ndarray:
        return np.asarray([row.value for row in self._rows], dtype=float)

    def at_time(self, t: int) -> "MetricSeries":
        return MetricSeries(row for row in self._rows if row.t == t)

    def step_level(self) -> "MetricSeries":
        return MetricSeries(row for row in self._rows if row.level == "step")

    def trajectory_level(self) -> "MetricSeries":
        return MetricSeries(row for row in self._rows if row.level == "trajectory")

    def mean(self) -> float:
        return float(np.mean(self.values())) if self._rows else float("nan")

    def std(self) -> float:
        return float(np.std(self.values())) if self._rows else float("nan")

    def min(self) -> float:
        return float(np.min(self.values())) if self._rows else float("nan")

    def max(self) -> float:
        return float(np.max(self.values())) if self._rows else float("nan")

    def quantile(self, q: float) -> float:
        if not 0 <= q <= 1:
            raise ValueError("q must be between 0 and 1")
        return float(np.quantile(self.values(), q)) if self._rows else float("nan")

    def percentile(self, p: float) -> float:
        if not 0 <= p <= 100:
            raise ValueError("p must be between 0 and 100")
        return self.quantile(p / 100)

    def cvar(self, alpha: float = 0.95) -> float:
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be between 0 and 1")
        if not self._rows:
            return float("nan")

        values = self.values()
        threshold = np.quantile(values, alpha)
        tail = values[values >= threshold]
        return float(np.mean(tail))

    def summary(self) -> dict[str, float]:
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
    name: str

    def on_step(self, step: StepRecord, store: MetricStore) -> None:
        pass

    def on_trajectory(self, trajectory: TrajectoryRecord, store: MetricStore) -> None:
        pass


class MetricSet:
    """Dispatches records to a group of metrics."""

    def __init__(self, metrics: Iterable[Metric] | None = None) -> None:
        self.metrics = list(metrics or [])
        names = [metric.name for metric in self.metrics]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate metric names: {duplicates}")

    def on_step(self, step: StepRecord, store: MetricStore) -> None:
        for metric in self.metrics:
            metric.on_step(step, store)

    def on_trajectory(self, trajectory: TrajectoryRecord, store: MetricStore) -> None:
        for metric in self.metrics:
            metric.on_trajectory(trajectory, store)


class StepCostMetric(Metric):
    name = "step_cost"

    def on_step(self, step: StepRecord, store: MetricStore) -> None:
        store.log(
            name=self.name,
            values=step.cost,
            scenario_ids=step.scenario_ids,
            t=step.t,
            level="step",
        )


class TotalCostMetric(Metric):
    name = "total_cost"

    def on_trajectory(self, trajectory: TrajectoryRecord, store: MetricStore) -> None:
        store.log(
            name=self.name,
            values=trajectory.total_cost,
            scenario_ids=trajectory.scenario_ids,
            level="trajectory",
        )
