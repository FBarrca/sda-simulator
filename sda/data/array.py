from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from numbers import Integral, Real
from typing import Any

import numpy as np

from sda.core import ScenarioBatch, ScenarioSpec
from sda.data._state import scenario_initial_state
from sda.data.module import DataModule


class ArrayDataModule(DataModule):
    """Data module backed by in-memory per-scenario time paths."""

    def __init__(
        self,
        paths: Mapping[str, Any],
        *,
        initial_state: Any = None,
        batch_size: int | None = None,
        scenario_ids: Sequence[int] | None = None,
        seeds: Sequence[int] | None = None,
        end_time: float | None = None,
    ) -> None:
        """Create an array-backed SimPy scenario data module."""
        if not paths:
            raise ValueError("paths must contain at least one path")

        self.initial_state = initial_state
        self.paths = {name: np.asarray(path) for name, path in paths.items()}
        first_path = next(iter(self.paths.values()))
        if first_path.ndim < 1:
            raise ValueError("paths must have shape [n_scenarios, ...]")

        self.n_scenarios = int(first_path.shape[0])
        self.path_length = int(first_path.shape[1]) if first_path.ndim >= 2 else 0
        self.end_time = (
            _nonnegative_float("end_time", end_time)
            if end_time is not None
            else float(self.path_length)
        )
        self.batch_size = (
            self.n_scenarios
            if batch_size is None
            else _positive_int("batch_size", batch_size)
        )

        for name, path in self.paths.items():
            if path.ndim < 1:
                raise ValueError(f"paths[{name!r}] must have shape [n_scenarios, ...]")
            if path.shape[0] != self.n_scenarios:
                raise ValueError(
                    f"paths[{name!r}] has {path.shape[0]} scenarios, "
                    f"expected {self.n_scenarios}"
                )

        self.scenario_ids = _prepare_scenario_ids(scenario_ids, self.n_scenarios)
        self.seeds = _prepare_seeds(seeds, self.n_scenarios)

    def batches(self, stage: str = "evaluate") -> Iterator[ScenarioBatch]:
        """Yield consecutive scenario batches."""
        del stage
        for start in range(0, self.n_scenarios, self.batch_size):
            stop = min(start + self.batch_size, self.n_scenarios)
            scenarios = [
                ScenarioSpec(
                    scenario_id=int(self.scenario_ids[index]),
                    end_time=self.end_time,
                    initial_state=scenario_initial_state(
                        self.initial_state,
                        index,
                        self.n_scenarios,
                    ),
                    data={
                        name: path[index]
                        for name, path in self.paths.items()
                    },
                    seed=None if self.seeds is None else int(self.seeds[index]),
                )
                for index in range(start, stop)
            ]
            yield ScenarioBatch(scenarios)


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return int(value)


def _nonnegative_float(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return float(value)


def _prepare_scenario_ids(
    scenario_ids: Sequence[int] | None,
    n_scenarios: int,
) -> np.ndarray:
    if scenario_ids is None:
        return np.arange(n_scenarios)
    if len(scenario_ids) != n_scenarios:
        raise ValueError("scenario_ids length must match scenario count")
    return np.asarray(scenario_ids)


def _prepare_seeds(
    seeds: Sequence[int] | None,
    n_scenarios: int,
) -> np.ndarray | None:
    if seeds is None:
        return None
    if len(seeds) != n_scenarios:
        raise ValueError("seeds length must match scenario count")
    return np.asarray(seeds, dtype=int)

__all__ = [
    "ArrayDataModule",
]
