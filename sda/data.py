from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ScenarioBatch:
    """A batch of independent futures for sequential simulation."""

    initial_state: Any
    exogenous: Mapping[str, Any]
    scenario_ids: Sequence[int]

    def __post_init__(self) -> None:
        scenario_count = len(self.scenario_ids)
        if scenario_count == 0:
            raise ValueError("scenario_ids must contain at least one scenario")

        if not self.exogenous:
            raise ValueError("exogenous must contain at least one path")

        horizon: int | None = None
        for name, path in self.exogenous.items():
            array = np.asarray(path)
            if array.ndim < 2:
                raise ValueError(
                    f"exogenous[{name!r}] must have shape [batch_size, horizon, ...]"
                )
            if array.shape[0] != scenario_count:
                raise ValueError(
                    f"exogenous[{name!r}] has batch size {array.shape[0]}, "
                    f"but scenario_ids has length {scenario_count}"
                )
            if horizon is None:
                horizon = int(array.shape[1])
            elif array.shape[1] != horizon:
                raise ValueError(
                    f"exogenous[{name!r}] has horizon {array.shape[1]}, "
                    f"expected {horizon}"
                )

    @property
    def batch_size(self) -> int:
        return len(self.scenario_ids)

    @property
    def horizon(self) -> int:
        first_path = next(iter(self.exogenous.values()))
        return int(np.asarray(first_path).shape[1])


class ScenarioLoader:
    """Produces batches of exogenous futures."""

    def __iter__(self) -> Iterator[ScenarioBatch]:
        raise NotImplementedError


class ArrayScenarioLoader(ScenarioLoader):
    """Scenario loader backed by batch-first NumPy-compatible arrays."""

    def __init__(
        self,
        initial_state: Any,
        exogenous: Mapping[str, Any],
        batch_size: int,
        scenario_ids: Sequence[int] | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not exogenous:
            raise ValueError("exogenous must contain at least one path")

        self.initial_state = initial_state
        self.exogenous = {name: np.asarray(path) for name, path in exogenous.items()}
        self.batch_size = int(batch_size)

        first_path = next(iter(self.exogenous.values()))
        if first_path.ndim < 2:
            raise ValueError("exogenous paths must have shape [n_scenarios, horizon, ...]")

        self.n_scenarios = int(first_path.shape[0])
        self.horizon = int(first_path.shape[1])

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
                raise ValueError("scenario_ids length must match exogenous scenario count")
            self.scenario_ids = np.asarray(scenario_ids)

    def __iter__(self) -> Iterator[ScenarioBatch]:
        for start in range(0, self.n_scenarios, self.batch_size):
            stop = min(start + self.batch_size, self.n_scenarios)
            yield ScenarioBatch(
                initial_state=_slice_initial_state(
                    self.initial_state, start, stop, self.n_scenarios
                ),
                exogenous={
                    name: path[start:stop]
                    for name, path in self.exogenous.items()
                },
                scenario_ids=self.scenario_ids[start:stop].tolist(),
            )


def _slice_initial_state(value: Any, start: int, stop: int, n_scenarios: int) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _slice_initial_state(item, start, stop, n_scenarios)
            for key, item in value.items()
        }

    array = np.asarray(value)
    if array.ndim == 0:
        return np.full(stop - start, array.item())
    if array.shape[0] != n_scenarios:
        raise ValueError(
            "initial_state must be scalar, mapping of scalar/vector values, "
            "or have one entry per scenario"
        )
    return array[start:stop]
