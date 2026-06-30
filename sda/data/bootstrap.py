from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from numbers import Integral
from typing import Any, Literal

import numpy as np

from sda.data.core import ScenarioBatch, ScenarioLoader, _slice_initial_state

BootstrapMethod = Literal[
    "iid",
    "circular_block",
    "moving_block",
    "stationary_block",
]


class BootstrapScenarioLoader(ScenarioLoader):
    """Scenario loader that resamples futures from historical observations.

    Each history value must be array-like with shape ``[n_observations, ...]``.
    The loader samples observation indexes with replacement and returns futures
    shaped ``[batch_size, horizon, ...]``.

    The supported methods follow common bootstrap resampling schemes:

    ``"iid"``
        Sample individual observations independently with replacement.
    ``"circular_block"``
        Sample fixed-length contiguous blocks that wrap around the history.
    ``"moving_block"``
        Sample fixed-length contiguous blocks that do not wrap.
    ``"stationary_block"``
        Sample circular blocks with geometrically distributed lengths.
    """

    def __init__(
        self,
        initial_state: Any,
        history: Mapping[str, Any],
        horizon: int,
        n_scenarios: int,
        batch_size: int,
        *,
        method: BootstrapMethod = "iid",
        block_size: int | None = None,
        average_block_size: float | None = None,
        seed: int | None = None,
        scenario_ids: Sequence[int] | None = None,
    ) -> None:
        """Create a bootstrap scenario loader.

        Parameters
        ----------
        initial_state
            Starting state for the model. It follows the same scalar,
            per-scenario vector, and mapping rules as ``ArrayScenarioLoader``.
        history
            Mapping from input name to historical observations shaped
            ``[n_observations, ...]``.
        horizon
            Number of periods in each sampled future.
        n_scenarios
            Number of bootstrap futures to generate.
        batch_size
            Maximum number of scenarios yielded in each batch.
        method
            Bootstrap method: ``"iid"``, ``"circular_block"``,
            ``"moving_block"``, or ``"stationary_block"``.
        block_size
            Fixed block length for ``"circular_block"`` or ``"moving_block"``.
        average_block_size
            Mean block length for ``"stationary_block"``. The corresponding
            geometric restart probability is ``1 / average_block_size``.
        seed
            Optional random seed.
        scenario_ids
            Optional scenario identifiers. When omitted, scenarios are numbered
            from ``0`` to ``n_scenarios - 1``.
        """
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if n_scenarios <= 0:
            raise ValueError("n_scenarios must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not history:
            raise ValueError("history must contain at least one path")
        if method not in (
            "iid",
            "circular_block",
            "moving_block",
            "stationary_block",
        ):
            raise ValueError(
                "method must be 'iid', 'circular_block', 'moving_block', "
                "or 'stationary_block'"
            )

        self.initial_state = initial_state
        self.history = {name: np.asarray(values) for name, values in history.items()}
        self.horizon = int(horizon)
        self.n_scenarios = int(n_scenarios)
        self.batch_size = int(batch_size)
        self.method = method
        self.seed = seed

        first_path = next(iter(self.history.values()))
        if first_path.ndim < 1:
            raise ValueError(
                "history values must have shape [n_observations, ...]"
            )
        self.n_observations = int(first_path.shape[0])
        if self.n_observations == 0:
            raise ValueError("history values must contain at least one observation")

        for name, values in self.history.items():
            if values.ndim < 1:
                raise ValueError(
                    f"history[{name!r}] must have shape [n_observations, ...]"
                )
            if values.shape[0] != self.n_observations:
                raise ValueError(
                    f"history[{name!r}] has {values.shape[0]} observations, "
                    f"expected {self.n_observations}"
                )

        self.block_size = _validate_bootstrap_block_size(
            method,
            block_size,
            self.n_observations,
        )
        self.average_block_size = _validate_average_block_size(
            method,
            average_block_size,
        )

        if scenario_ids is None:
            self.scenario_ids = np.arange(self.n_scenarios)
        else:
            if len(scenario_ids) != self.n_scenarios:
                raise ValueError("scenario_ids length must match n_scenarios")
            self.scenario_ids = np.asarray(scenario_ids)

    def __iter__(self) -> Iterator[ScenarioBatch]:
        """Yield bootstrap scenario batches."""
        rng = np.random.default_rng(self.seed)
        for start in range(0, self.n_scenarios, self.batch_size):
            stop = min(start + self.batch_size, self.n_scenarios)
            size = stop - start
            sampled_indexes = self._sample_indexes(rng, size)

            yield ScenarioBatch(
                initial_state=_slice_initial_state(
                    self.initial_state, start, stop, self.n_scenarios
                ),
                exogenous={
                    name: values[sampled_indexes]
                    for name, values in self.history.items()
                },
                scenario_ids=self.scenario_ids[start:stop].tolist(),
            )

    def _sample_indexes(
        self,
        rng: np.random.Generator,
        batch_size: int,
    ) -> np.ndarray:
        if self.method == "iid":
            return _iid_bootstrap_indexes(
                rng,
                n_observations=self.n_observations,
                batch_size=batch_size,
                horizon=self.horizon,
            )
        if self.method == "circular_block":
            return _circular_block_bootstrap_indexes(
                rng,
                n_observations=self.n_observations,
                batch_size=batch_size,
                horizon=self.horizon,
                block_size=self.block_size,
            )
        if self.method == "moving_block":
            return _moving_block_bootstrap_indexes(
                rng,
                n_observations=self.n_observations,
                batch_size=batch_size,
                horizon=self.horizon,
                block_size=self.block_size,
            )
        if self.method == "stationary_block":
            return _stationary_block_bootstrap_indexes(
                rng,
                n_observations=self.n_observations,
                batch_size=batch_size,
                horizon=self.horizon,
                average_block_size=self.average_block_size,
            )
        raise ValueError(
            "method must be 'iid', 'circular_block', 'moving_block', "
            "or 'stationary_block'"
        )


class IIDBootstrapScenarioLoader(BootstrapScenarioLoader):
    """Bootstrap futures by independently sampling historical observations."""

    def __init__(
        self,
        initial_state: Any,
        history: Mapping[str, Any],
        horizon: int,
        n_scenarios: int,
        batch_size: int,
        *,
        seed: int | None = None,
        scenario_ids: Sequence[int] | None = None,
    ) -> None:
        super().__init__(
            initial_state=initial_state,
            history=history,
            horizon=horizon,
            n_scenarios=n_scenarios,
            batch_size=batch_size,
            method="iid",
            seed=seed,
            scenario_ids=scenario_ids,
        )


class CircularBlockBootstrapScenarioLoader(BootstrapScenarioLoader):
    """Bootstrap futures from fixed-length circular blocks of history."""

    def __init__(
        self,
        initial_state: Any,
        history: Mapping[str, Any],
        horizon: int,
        n_scenarios: int,
        batch_size: int,
        *,
        block_size: int,
        seed: int | None = None,
        scenario_ids: Sequence[int] | None = None,
    ) -> None:
        super().__init__(
            initial_state=initial_state,
            history=history,
            horizon=horizon,
            n_scenarios=n_scenarios,
            batch_size=batch_size,
            method="circular_block",
            block_size=block_size,
            seed=seed,
            scenario_ids=scenario_ids,
        )


class MovingBlockBootstrapScenarioLoader(BootstrapScenarioLoader):
    """Bootstrap futures from fixed-length non-wrapping blocks of history."""

    def __init__(
        self,
        initial_state: Any,
        history: Mapping[str, Any],
        horizon: int,
        n_scenarios: int,
        batch_size: int,
        *,
        block_size: int,
        seed: int | None = None,
        scenario_ids: Sequence[int] | None = None,
    ) -> None:
        super().__init__(
            initial_state=initial_state,
            history=history,
            horizon=horizon,
            n_scenarios=n_scenarios,
            batch_size=batch_size,
            method="moving_block",
            block_size=block_size,
            seed=seed,
            scenario_ids=scenario_ids,
        )


class StationaryBlockBootstrapScenarioLoader(BootstrapScenarioLoader):
    """Bootstrap futures from circular blocks with random lengths."""

    def __init__(
        self,
        initial_state: Any,
        history: Mapping[str, Any],
        horizon: int,
        n_scenarios: int,
        batch_size: int,
        *,
        average_block_size: float,
        seed: int | None = None,
        scenario_ids: Sequence[int] | None = None,
    ) -> None:
        super().__init__(
            initial_state=initial_state,
            history=history,
            horizon=horizon,
            n_scenarios=n_scenarios,
            batch_size=batch_size,
            method="stationary_block",
            average_block_size=average_block_size,
            seed=seed,
            scenario_ids=scenario_ids,
        )


class IIDBootstrap(IIDBootstrapScenarioLoader):
    """Arch-style name for IID bootstrap scenario loading."""


class CircularBlockBootstrap(CircularBlockBootstrapScenarioLoader):
    """Arch-style name for circular block bootstrap scenario loading."""


class MovingBlockBootstrap(MovingBlockBootstrapScenarioLoader):
    """Arch-style name for moving block bootstrap scenario loading."""


class StationaryBootstrap(StationaryBlockBootstrapScenarioLoader):
    """Arch-style name for stationary bootstrap scenario loading.

    ``block_size`` is the average block size, matching the name used by
    ``arch.bootstrap.StationaryBootstrap``.
    """

    def __init__(
        self,
        initial_state: Any,
        history: Mapping[str, Any],
        horizon: int,
        n_scenarios: int,
        batch_size: int,
        *,
        block_size: float,
        seed: int | None = None,
        scenario_ids: Sequence[int] | None = None,
    ) -> None:
        super().__init__(
            initial_state=initial_state,
            history=history,
            horizon=horizon,
            n_scenarios=n_scenarios,
            batch_size=batch_size,
            average_block_size=block_size,
            seed=seed,
            scenario_ids=scenario_ids,
        )


def _validate_bootstrap_block_size(
    method: BootstrapMethod,
    block_size: int | None,
    n_observations: int,
) -> int:
    if method in ("circular_block", "moving_block"):
        if block_size is None:
            raise ValueError(f"block_size is required for {method} bootstrap")
        if not isinstance(block_size, Integral):
            raise ValueError("block_size must be an integer")
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if method == "moving_block" and block_size > n_observations:
            raise ValueError(
                "block_size must be less than or equal to the number of "
                "observations for moving_block bootstrap"
            )
        return int(block_size)

    if block_size is not None:
        raise ValueError(
            "block_size is only supported for circular_block and moving_block "
            "bootstrap"
        )
    return 1


def _validate_average_block_size(
    method: BootstrapMethod,
    average_block_size: float | None,
) -> float:
    if method == "stationary_block":
        if average_block_size is None:
            raise ValueError(
                "average_block_size is required for stationary_block bootstrap"
            )
        if average_block_size < 1 or not np.isfinite(average_block_size):
            raise ValueError("average_block_size must be finite and at least 1")
        return float(average_block_size)

    if average_block_size is not None:
        raise ValueError(
            "average_block_size is only supported for stationary_block bootstrap"
        )
    return 1.0


def _iid_bootstrap_indexes(
    rng: np.random.Generator,
    *,
    n_observations: int,
    batch_size: int,
    horizon: int,
) -> np.ndarray:
    return rng.integers(
        0,
        n_observations,
        size=(batch_size, horizon),
        dtype=np.int64,
    )


def _circular_block_bootstrap_indexes(
    rng: np.random.Generator,
    *,
    n_observations: int,
    batch_size: int,
    horizon: int,
    block_size: int,
) -> np.ndarray:
    indexes = np.empty((batch_size, horizon), dtype=np.int64)
    for scenario_index in range(batch_size):
        position = 0
        while position < horizon:
            block_start = int(rng.integers(0, n_observations))
            length = min(block_size, horizon - position)
            indexes[scenario_index, position : position + length] = (
                block_start + np.arange(length, dtype=np.int64)
            ) % n_observations
            position += length
    return indexes


def _moving_block_bootstrap_indexes(
    rng: np.random.Generator,
    *,
    n_observations: int,
    batch_size: int,
    horizon: int,
    block_size: int,
) -> np.ndarray:
    indexes = np.empty((batch_size, horizon), dtype=np.int64)
    max_start = n_observations - block_size
    for scenario_index in range(batch_size):
        position = 0
        while position < horizon:
            block_start = int(rng.integers(0, max_start + 1))
            length = min(block_size, horizon - position)
            indexes[scenario_index, position : position + length] = (
                block_start + np.arange(length, dtype=np.int64)
            )
            position += length
    return indexes


def _stationary_block_bootstrap_indexes(
    rng: np.random.Generator,
    *,
    n_observations: int,
    batch_size: int,
    horizon: int,
    average_block_size: float,
) -> np.ndarray:
    restart_probability = 1.0 / average_block_size
    indexes = np.empty((batch_size, horizon), dtype=np.int64)
    for scenario_index in range(batch_size):
        current = int(rng.integers(0, n_observations))
        indexes[scenario_index, 0] = current
        for t in range(1, horizon):
            if rng.random() < restart_probability:
                current = int(rng.integers(0, n_observations))
            else:
                current = (current + 1) % n_observations
            indexes[scenario_index, t] = current
    return indexes
