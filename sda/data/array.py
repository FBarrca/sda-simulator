from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import numpy as np

from sda.data.core import ScenarioBatch, ScenarioLoader, _slice_initial_state


class ArrayScenarioLoader(ScenarioLoader):
    """Scenario loader backed by NumPy-compatible arrays.

    Use this loader when all scenario paths are already available in memory.
    Each exogenous value must be array-like with shape
    ``[n_scenarios, horizon, ...]``. The loader slices those arrays into
    ``ScenarioBatch`` objects of at most ``batch_size`` scenarios.
    """

    def __init__(
        self,
        initial_state: Any,
        exogenous: Mapping[str, Any],
        batch_size: int,
        scenario_ids: Sequence[int] | None = None,
    ) -> None:
        """Create an array-backed scenario loader.

        Parameters
        ----------
        initial_state
            Starting state for the model. It may be a scalar, a vector with one
            entry per scenario, or a mapping whose values follow the same rule.
        exogenous
            Mapping from input name to array-like future paths shaped
            ``[n_scenarios, horizon, ...]``.
        batch_size
            Maximum number of scenarios yielded in each batch.
        scenario_ids
            Optional scenario identifiers. When omitted, scenarios are numbered
            from ``0`` to ``n_scenarios - 1``.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not exogenous:
            raise ValueError("exogenous must contain at least one path")

        self.initial_state = initial_state
        self.exogenous = {name: np.asarray(path) for name, path in exogenous.items()}
        self.batch_size = int(batch_size)

        first_path = next(iter(self.exogenous.values()))
        if first_path.ndim < 2:
            raise ValueError(
                "exogenous paths must have shape [n_scenarios, horizon, ...]"
            )

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
                raise ValueError(
                    "scenario_ids length must match exogenous scenario count"
                )
            self.scenario_ids = np.asarray(scenario_ids)

    def __iter__(self) -> Iterator[ScenarioBatch]:
        """Yield consecutive scenario batches from the stored arrays."""
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
