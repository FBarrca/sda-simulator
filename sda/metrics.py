from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import simpy

from sda.core import EventLevel, EventRecord


class MetricStore:
    """In-memory store for SimPy event-time metric observations."""

    def __init__(self) -> None:
        """Create an empty metric store."""
        self._rows: list[EventRecord] = []

    def log(
        self,
        name: str,
        values: Any,
        *,
        scenario_id: int | None = None,
        scenario_ids: Sequence[int] | None = None,
        time: float = 0.0,
        level: EventLevel = "event",
        tags: Mapping[str, str] | None = None,
    ) -> None:
        """Append one metric observation or one value per scenario."""
        if level not in {"event", "trajectory"}:
            raise ValueError("level must be 'event' or 'trajectory'")

        values_array = np.asarray(values, dtype=float)
        if values_array.ndim == 0:
            values_list = [float(values_array)]
        elif values_array.ndim == 1:
            values_list = [float(value) for value in values_array]
        else:
            raise ValueError("metric values must be scalar or one-dimensional")

        if scenario_ids is not None:
            ids = [int(item) for item in scenario_ids]
            if len(values_list) == 1 and len(ids) > 1:
                values_list = values_list * len(ids)
            if len(ids) != len(values_list):
                raise ValueError(
                    f"metric {name!r} received {len(values_list)} values "
                    f"for {len(ids)} scenario_ids"
                )
        else:
            ids = [None if scenario_id is None else int(scenario_id)]
            if len(values_list) != 1:
                raise ValueError("scenario_ids are required for vector metric values")

        tag_values = {str(key): str(value) for key, value in dict(tags or {}).items()}
        for value, row_scenario_id in zip(values_list, ids, strict=True):
            self._append_record(
                EventRecord(
                    name=name,
                    value=value,
                    scenario_id=row_scenario_id,
                    time=float(time),
                    level=level,
                    tags=tag_values,
                )
            )

    def metric(self, name: str) -> "MetricSeries":
        """Return a queryable series containing rows with ``name``."""
        return MetricSeries(row for row in self._rows if row.name == name)

    def names(self) -> list[str]:
        """Return metric names in the order they were first logged."""
        return list(dict.fromkeys(row.name for row in self._rows))

    def rows(self) -> list[EventRecord]:
        """Return a copy of all raw event records."""
        return list(self._rows)

    def _append_record(self, record: EventRecord) -> EventRecord:
        self._rows.append(record)
        return record


class Recorder:
    """Scenario-local helper for logging SimPy event-time metrics."""

    def __init__(
        self,
        store: MetricStore,
        *,
        scenario_id: int,
        env: simpy.Environment,
    ) -> None:
        """Create a recorder bound to one scenario and environment."""
        self.store = store
        self.scenario_id = int(scenario_id)
        self.env = env
        self.total_cost = 0.0
        self.history: list[EventRecord] = []
        self._closed = False

    def log(
        self,
        name: str,
        value: Any,
        *,
        tags: Mapping[str, str] | None = None,
    ) -> EventRecord:
        """Log an event-level metric at ``env.now``."""
        record = self._append(
            name=name,
            value=value,
            level="event",
            tags=tags,
        )
        return record

    def cost(
        self,
        value: Any,
        *,
        name: str = "cost",
        tags: Mapping[str, str] | None = None,
    ) -> EventRecord:
        """Log an event cost and add it to this scenario's total cost."""
        amount = _scalar_float(value, "cost")
        self.total_cost += amount
        return self._append(
            name=name,
            value=amount,
            level="event",
            tags=tags,
        )

    def trajectory(
        self,
        name: str,
        value: Any,
        *,
        tags: Mapping[str, str] | None = None,
    ) -> EventRecord:
        """Log a trajectory-level metric at ``env.now``."""
        return self._append(
            name=name,
            value=value,
            level="trajectory",
            tags=tags,
        )

    def close(self) -> None:
        """Log final total cost once."""
        if self._closed:
            return
        self.trajectory("total_cost", self.total_cost)
        self._closed = True

    def _append(
        self,
        *,
        name: str,
        value: Any,
        level: EventLevel,
        tags: Mapping[str, str] | None,
    ) -> EventRecord:
        amount = _scalar_float(value, name)
        record = self.store._append_record(
            EventRecord(
                name=name,
                value=amount,
                scenario_id=self.scenario_id,
                time=float(self.env.now),
                level=level,
                tags={str(key): str(value) for key, value in dict(tags or {}).items()},
            )
        )
        self.history.append(record)
        return record


class MetricSeries:
    """Queryable distribution over stored event observations."""

    def __init__(self, rows: Iterable[EventRecord]) -> None:
        """Create a series from event records."""
        self._rows = tuple(rows)

    def __iter__(self):
        """Iterate over raw event records."""
        return iter(self._rows)

    def __len__(self) -> int:
        """Return the number of observations in this series."""
        return len(self._rows)

    def rows(self) -> list[EventRecord]:
        """Return a copy of the raw rows in this series."""
        return list(self._rows)

    def records(self) -> list[dict[str, Any]]:
        """Return rows as plain dictionaries for lightweight export."""
        return [
            {
                "name": row.name,
                "value": row.value,
                "scenario_id": row.scenario_id,
                "time": row.time,
                "level": row.level,
                "tags": dict(row.tags),
            }
            for row in self._rows
        ]

    def values(self) -> np.ndarray:
        """Return metric values as a one-dimensional float array."""
        return np.asarray([row.value for row in self._rows], dtype=float)

    def count(self) -> int:
        """Return the number of observations in this series."""
        return len(self._rows)

    def at_time(self, time: float) -> "MetricSeries":
        """Return observations recorded exactly at ``time``."""
        return MetricSeries(row for row in self._rows if row.time == float(time))

    def between(
        self,
        start: float,
        stop: float,
        *,
        inclusive: bool = True,
    ) -> "MetricSeries":
        """Return observations whose event time falls in a range."""
        if inclusive:
            return MetricSeries(
                row for row in self._rows if float(start) <= row.time <= float(stop)
            )
        return MetricSeries(
            row for row in self._rows if float(start) < row.time < float(stop)
        )

    def event_level(self) -> "MetricSeries":
        """Return only event-level observations."""
        return MetricSeries(row for row in self._rows if row.level == "event")

    def trajectory_level(self) -> "MetricSeries":
        """Return only trajectory-level observations."""
        return MetricSeries(row for row in self._rows if row.level == "trajectory")

    def with_tag(self, key: str, value: str) -> "MetricSeries":
        """Return observations with a matching tag value."""
        return MetricSeries(
            row for row in self._rows if row.tags.get(str(key)) == str(value)
        )

    def to_trajectory_matrix(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return event observations as scenario ids, event times, and values."""
        event_rows = tuple(row for row in self._rows if row.level == "event")
        if not event_rows:
            return (
                np.asarray([], dtype=int),
                np.asarray([], dtype=float),
                np.empty((0, 0), dtype=float),
            )
        if any(row.scenario_id is None for row in event_rows):
            raise ValueError("trajectory matrix requires scenario_id on every row")

        scenario_ids = np.asarray(
            sorted({int(row.scenario_id) for row in event_rows}),
            dtype=int,
        )
        times = np.asarray(sorted({float(row.time) for row in event_rows}), dtype=float)
        scenario_index = {
            scenario_id: index for index, scenario_id in enumerate(scenario_ids)
        }
        time_index = {time: index for index, time in enumerate(times)}
        values = np.full((len(scenario_ids), len(times)), np.nan, dtype=float)

        seen: set[tuple[int, float]] = set()
        for row in event_rows:
            scenario_id = int(row.scenario_id)
            key = (scenario_id, float(row.time))
            if key in seen:
                raise ValueError(
                    "trajectory matrix received duplicate values for "
                    f"scenario_id={scenario_id}, time={row.time}"
                )
            seen.add(key)
            values[scenario_index[scenario_id], time_index[float(row.time)]] = row.value
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
        """Return conditional value at risk for the upper tail."""
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


def _scalar_float(value: Any, name: str) -> float:
    array = np.asarray(value, dtype=float)
    if array.ndim != 0:
        raise ValueError(f"{name} must be scalar")
    return float(array)


__all__ = [
    "MetricSeries",
    "MetricStore",
    "Recorder",
]
