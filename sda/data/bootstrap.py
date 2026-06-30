from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Literal, cast

import numpy as np

from sda.data.core import ScenarioBatch, ScenarioLoader, _slice_initial_state

BootstrapMethod = Literal[
    "iid",
    "circular_block",
    "moving_block",
    "stationary_block",
]

_VALID_METHODS: tuple[BootstrapMethod, ...] = (
    "iid",
    "circular_block",
    "moving_block",
    "stationary_block",
)
_FIXED_BLOCK_METHODS = {"circular_block", "moving_block"}
_METHOD_ERROR = (
    "method must be 'iid', 'circular_block', 'moving_block', or 'stationary_block'"
)


@dataclass(frozen=True)
class _SamplingSpec:
    n_observations: int
    horizon: int
    block_size: int = 1
    average_block_size: float = 1.0


_IndexSampler = Callable[[np.random.Generator, _SamplingSpec, int], np.ndarray]


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
        """Create a bootstrap scenario loader."""
        self.horizon = _require_positive_integer("horizon", horizon)
        self.n_scenarios = _require_positive_integer("n_scenarios", n_scenarios)
        self.batch_size = _require_positive_integer("batch_size", batch_size)
        self.method = _validate_method(method)

        self.initial_state = initial_state
        self.history, self.n_observations = _coerce_history(history)
        self.block_size = _validate_block_size(
            self.method,
            block_size,
            self.n_observations,
        )
        self.average_block_size = _validate_average_block_size(
            self.method,
            average_block_size,
        )
        self.seed = seed
        self.scenario_ids = _scenario_id_array(scenario_ids, self.n_scenarios)
        self._sampling_spec = _SamplingSpec(
            n_observations=self.n_observations,
            horizon=self.horizon,
            block_size=self.block_size,
            average_block_size=self.average_block_size,
        )
        self._index_sampler = _INDEX_SAMPLERS[self.method]

    def __iter__(self) -> Iterator[ScenarioBatch]:
        """Yield bootstrap scenario batches."""
        rng = np.random.default_rng(self.seed)
        for start in range(0, self.n_scenarios, self.batch_size):
            stop = min(start + self.batch_size, self.n_scenarios)
            sampled_indexes = self._sample_indexes(rng, stop - start)

            yield ScenarioBatch(
                initial_state=_slice_initial_state(
                    self.initial_state,
                    start,
                    stop,
                    self.n_scenarios,
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
        return self._index_sampler(rng, self._sampling_spec, batch_size)


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
            initial_state,
            history,
            horizon,
            n_scenarios,
            batch_size,
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
            initial_state,
            history,
            horizon,
            n_scenarios,
            batch_size,
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
            initial_state,
            history,
            horizon,
            n_scenarios,
            batch_size,
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
            initial_state,
            history,
            horizon,
            n_scenarios,
            batch_size,
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
            initial_state,
            history,
            horizon,
            n_scenarios,
            batch_size,
            average_block_size=block_size,
            seed=seed,
            scenario_ids=scenario_ids,
        )


def _require_positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return int(value)


def _validate_method(method: str) -> BootstrapMethod:
    if method not in _VALID_METHODS:
        raise ValueError(_METHOD_ERROR)
    return cast(BootstrapMethod, method)


def _coerce_history(history: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], int]:
    if not history:
        raise ValueError("history must contain at least one path")

    arrays = {name: np.asarray(values) for name, values in history.items()}
    history_items = iter(arrays.items())
    first_name, first_values = next(history_items)
    n_observations = _validate_history_array(first_name, first_values)

    for name, values in history_items:
        observation_count = _validate_history_array(name, values)
        if observation_count != n_observations:
            raise ValueError(
                f"history[{name!r}] has {observation_count} observations, "
                f"expected {n_observations}"
            )
    return arrays, n_observations


def _validate_history_array(name: str, values: np.ndarray) -> int:
    if values.ndim < 1:
        raise ValueError(
            f"history[{name!r}] must have shape [n_observations, ...]"
        )
    if values.shape[0] == 0:
        raise ValueError("history values must contain at least one observation")
    return int(values.shape[0])


def _scenario_id_array(
    scenario_ids: Sequence[int] | None,
    n_scenarios: int,
) -> np.ndarray:
    if scenario_ids is None:
        return np.arange(n_scenarios)
    if len(scenario_ids) != n_scenarios:
        raise ValueError("scenario_ids length must match n_scenarios")
    return np.asarray(scenario_ids)


def _validate_block_size(
    method: BootstrapMethod,
    block_size: int | None,
    n_observations: int,
) -> int:
    if method not in _FIXED_BLOCK_METHODS:
        if block_size is not None:
            raise ValueError(
                "block_size is only supported for circular_block and moving_block "
                "bootstrap"
            )
        return 1

    if block_size is None:
        raise ValueError(f"block_size is required for {method} bootstrap")
    if isinstance(block_size, bool) or not isinstance(block_size, Integral):
        raise ValueError("block_size must be an integer")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if method == "moving_block" and block_size > n_observations:
        raise ValueError(
            "block_size must be less than or equal to the number of "
            "observations for moving_block bootstrap"
        )
    return int(block_size)


def _validate_average_block_size(
    method: BootstrapMethod,
    average_block_size: float | None,
) -> float:
    if method != "stationary_block":
        if average_block_size is not None:
            raise ValueError(
                "average_block_size is only supported for stationary_block bootstrap"
            )
        return 1.0

    if average_block_size is None:
        raise ValueError("average_block_size is required for stationary_block bootstrap")
    if not isinstance(average_block_size, Real) or not np.isfinite(average_block_size):
        raise ValueError("average_block_size must be finite and at least 1")
    if average_block_size < 1:
        raise ValueError("average_block_size must be finite and at least 1")
    return float(average_block_size)


def _iid_bootstrap_indexes(
    rng: np.random.Generator,
    spec: _SamplingSpec,
    batch_size: int,
) -> np.ndarray:
    return rng.integers(
        0,
        spec.n_observations,
        size=(batch_size, spec.horizon),
        dtype=np.int64,
    )


def _circular_block_bootstrap_indexes(
    rng: np.random.Generator,
    spec: _SamplingSpec,
    batch_size: int,
) -> np.ndarray:
    return _fixed_block_bootstrap_indexes(rng, spec, batch_size, wrap=True)


def _moving_block_bootstrap_indexes(
    rng: np.random.Generator,
    spec: _SamplingSpec,
    batch_size: int,
) -> np.ndarray:
    return _fixed_block_bootstrap_indexes(rng, spec, batch_size, wrap=False)


def _fixed_block_bootstrap_indexes(
    rng: np.random.Generator,
    spec: _SamplingSpec,
    batch_size: int,
    *,
    wrap: bool,
) -> np.ndarray:
    indexes = np.empty((batch_size, spec.horizon), dtype=np.int64)
    offsets = np.arange(spec.block_size, dtype=np.int64)
    high = spec.n_observations if wrap else spec.n_observations - spec.block_size + 1

    for scenario_index in range(batch_size):
        position = 0
        while position < spec.horizon:
            block_start = int(rng.integers(0, high))
            length = min(spec.block_size, spec.horizon - position)
            block = block_start + offsets[:length]
            if wrap:
                block %= spec.n_observations
            indexes[scenario_index, position : position + length] = block
            position += length
    return indexes


def _stationary_block_bootstrap_indexes(
    rng: np.random.Generator,
    spec: _SamplingSpec,
    batch_size: int,
) -> np.ndarray:
    restart_probability = 1.0 / spec.average_block_size
    indexes = np.empty((batch_size, spec.horizon), dtype=np.int64)

    for scenario_index in range(batch_size):
        current = int(rng.integers(0, spec.n_observations))
        indexes[scenario_index, 0] = current
        for t in range(1, spec.horizon):
            if rng.random() < restart_probability:
                current = int(rng.integers(0, spec.n_observations))
            else:
                current = (current + 1) % spec.n_observations
            indexes[scenario_index, t] = current
    return indexes


_INDEX_SAMPLERS: dict[BootstrapMethod, _IndexSampler] = {
    "iid": _iid_bootstrap_indexes,
    "circular_block": _circular_block_bootstrap_indexes,
    "moving_block": _moving_block_bootstrap_indexes,
    "stationary_block": _stationary_block_bootstrap_indexes,
}
