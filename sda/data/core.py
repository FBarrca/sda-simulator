from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ScenarioBatch:
    """A batch of independent future paths for simulation.

    ``initial_state`` is the model's starting state for each scenario in the
    batch. ``exogenous`` maps input names to arrays shaped
    ``[batch_size, horizon, ...]``. ``scenario_ids`` identifies the scenarios
    represented by the first dimension of each exogenous array.
    """

    initial_state: Any
    exogenous: Mapping[str, Any]
    scenario_ids: Sequence[int]

    def __post_init__(self) -> None:
        """Validate batch size and horizon consistency across exogenous paths."""
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
        """Return the number of scenarios in this batch."""
        return len(self.scenario_ids)

    @property
    def horizon(self) -> int:
        """Return the number of time periods in this batch."""
        first_path = next(iter(self.exogenous.values()))
        return int(np.asarray(first_path).shape[1])


class ScenarioLoader:
    """Base interface for objects that produce scenario batches."""

    def __iter__(self) -> Iterator[ScenarioBatch]:
        """Yield ``ScenarioBatch`` objects for a simulation run."""
        raise NotImplementedError


def _slice_initial_state(value: Any, start: int, stop: int, n_scenarios: int) -> Any:
    """Slice or broadcast an initial-state value for one scenario batch.

    Scalars are broadcast to the current batch length. Mappings are processed
    recursively. Array-like values must have one entry per original scenario
    and are sliced along their first dimension.
    """
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
