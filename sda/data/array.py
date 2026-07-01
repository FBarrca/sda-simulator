from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from numbers import Integral
from typing import Any

import numpy as np

from sda.core import ScenarioBatch
from sda.data._state import slice_initial_state
from sda.data.module import DataModule


class ArrayDataModule(DataModule):
    """Data module backed by NumPy-compatible arrays.

    Use this data module when all scenario paths are already available in memory.
    Each exogenous value must be array-like with shape
    ``[n_scenarios, horizon, ...]``. The module slices those arrays into
    ``ScenarioBatch`` objects of at most ``batch_size`` scenarios.
    """

    def __init__(
        self,
        exogenous: Mapping[str, Any],
        *,
        initial_state: Any = 0,
        batch_size: int | None = None,
        scenario_ids: Sequence[int] | None = None,
    ) -> None:
        """Create an array-backed data module."""
        if not exogenous:
            raise ValueError("exogenous must contain at least one path")

        self.initial_state = initial_state
        self.exogenous = {name: np.asarray(path) for name, path in exogenous.items()}

        first_path = next(iter(self.exogenous.values()))
        if first_path.ndim < 2:
            raise ValueError(
                "exogenous paths must have shape [n_scenarios, horizon, ...]"
            )

        self.n_scenarios = int(first_path.shape[0])
        self.horizon = int(first_path.shape[1])
        self.batch_size = (
            self.n_scenarios
            if batch_size is None
            else _positive_int("batch_size", batch_size)
        )

        for name, path in self.exogenous.items():
            if path.ndim < 2:
                raise ValueError(
                    f"exogenous[{name!r}] must have shape [n_scenarios, horizon, ...]"
                )
            if path.shape[0] != self.n_scenarios:
                raise ValueError(
                    f"exogenous[{name!r}] has {path.shape[0]} scenarios, "
                    f"expected {self.n_scenarios}"
                )
            if path.shape[1] != self.horizon:
                raise ValueError(
                    f"exogenous[{name!r}] has horizon {path.shape[1]}, "
                    f"expected {self.horizon}"
                )

        if scenario_ids is None:
            self.scenario_ids = np.arange(self.n_scenarios)
        else:
            if len(scenario_ids) != self.n_scenarios:
                raise ValueError(
                    "scenario_ids length must match exogenous scenario count"
                )
            self.scenario_ids = np.asarray(scenario_ids)

    def batches(self, stage: str = "evaluate") -> Iterator[ScenarioBatch]:
        """Yield consecutive scenario batches from the stored arrays."""
        del stage
        for start in range(0, self.n_scenarios, self.batch_size):
            stop = min(start + self.batch_size, self.n_scenarios)
            yield ScenarioBatch(
                initial_state=slice_initial_state(
                    self.initial_state, start, stop, self.n_scenarios
                ),
                exogenous={
                    name: path[start:stop]
                    for name, path in self.exogenous.items()
                },
                scenario_ids=self.scenario_ids[start:stop].tolist(),
            )


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return int(value)


__all__ = [
    "ArrayDataModule",
]
