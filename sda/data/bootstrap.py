from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from numbers import Integral, Real
from typing import Any, Literal, cast

import numpy as np

from sda.core import ScenarioBatch, ScenarioSpec
from sda.data._state import scenario_initial_state
from sda.data.module import DataModule

BootstrapMethod = Literal[
    "iid",
    "circular_block",
    "moving_block",
    "stationary_block",
]

_BOOTSTRAP_METHODS: tuple[BootstrapMethod, ...] = (
    "iid",
    "circular_block",
    "moving_block",
    "stationary_block",
)
_METHOD_ALIASES: dict[str, BootstrapMethod] = {
    "iid": "iid",
    "circular": "circular_block",
    "circular_block": "circular_block",
    "moving": "moving_block",
    "moving_block": "moving_block",
    "stationary": "stationary_block",
    "stationary_block": "stationary_block",
}
_FIXED_BLOCK_METHODS: tuple[BootstrapMethod, ...] = (
    "circular_block",
    "moving_block",
)


class BootstrapDataModule(DataModule):
    """Data module that resamples futures from historical observations."""

    def __init__(
        self,
        history: Mapping[str, Any],
        *,
        horizon: int,
        n_scenarios: int,
        initial_state: Any = 0,
        batch_size: int | None = None,
        method: str = "iid",
        block_size: int | float | None = None,
        average_block_size: float | None = None,
        seed: int | None = None,
        scenario_ids: Sequence[int] | None = None,
    ) -> None:
        """Create a bootstrap data module."""
        self.horizon = _positive_int("horizon", horizon)
        self.n_scenarios = _positive_int("n_scenarios", n_scenarios)
        self.batch_size = (
            self.n_scenarios
            if batch_size is None
            else _positive_int("batch_size", batch_size)
        )
        self.method = _normalize_method(method)
        self.initial_state = initial_state
        self.seed = seed

        self.history, self.n_observations = _prepare_history(history)
        fixed_block_size, stationary_average = _bootstrap_block_parameters(
            method=self.method,
            block_size=block_size,
            average_block_size=average_block_size,
        )
        self.block_size = _validate_block_size(
            self.method,
            fixed_block_size,
            self.n_observations,
        )
        self.average_block_size = _validate_average_block_size(
            self.method,
            stationary_average,
        )
        self.scenario_ids = _prepare_scenario_ids(scenario_ids, self.n_scenarios)

    def batches(self, stage: str = "evaluate") -> Iterator[ScenarioBatch]:
        """Yield bootstrap scenario batches."""
        del stage
        rng = np.random.default_rng(self.seed)

        for start in range(0, self.n_scenarios, self.batch_size):
            stop = min(start + self.batch_size, self.n_scenarios)
            sampled_indexes = self._sample_indexes(rng, batch_size=stop - start)

            yield ScenarioBatch(
                [
                    ScenarioSpec(
                        scenario_id=int(self.scenario_ids[index]),
                        end_time=float(self.horizon),
                        initial_state=scenario_initial_state(
                            self.initial_state,
                            index,
                            self.n_scenarios,
                        ),
                        data={
                            name: values[sampled_indexes[index - start]]
                            for name, values in self.history.items()
                        },
                    )
                    for index in range(start, stop)
                ]
            )

    def _sample_indexes(
        self,
        rng: np.random.Generator,
        *,
        batch_size: int,
    ) -> np.ndarray:
        """Sample history indexes for one batch."""
        if self.method == "iid":
            return _iid_indexes(
                rng,
                n_observations=self.n_observations,
                batch_size=batch_size,
                horizon=self.horizon,
            )

        if self.method == "circular_block":
            return _circular_block_indexes(
                rng,
                n_observations=self.n_observations,
                batch_size=batch_size,
                horizon=self.horizon,
                block_size=self.block_size,
            )

        if self.method == "moving_block":
            return _moving_block_indexes(
                rng,
                n_observations=self.n_observations,
                batch_size=batch_size,
                horizon=self.horizon,
                block_size=self.block_size,
            )

        return _stationary_block_indexes(
            rng,
            n_observations=self.n_observations,
            batch_size=batch_size,
            horizon=self.horizon,
            average_block_size=self.average_block_size,
        )


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return int(value)


def _normalize_method(method: str) -> BootstrapMethod:
    try:
        return _METHOD_ALIASES[method]
    except KeyError as exc:
        methods = "', '".join(_METHOD_ALIASES)
        raise ValueError(f"method must be one of '{methods}'") from exc


def _bootstrap_block_parameters(
    *,
    method: BootstrapMethod,
    block_size: int | float | None,
    average_block_size: float | None,
) -> tuple[int | None, float | None]:
    if method == "stationary_block":
        if block_size is not None and average_block_size is not None:
            raise ValueError(
                "provide either block_size or average_block_size for stationary bootstrap"
            )
        return None, average_block_size if average_block_size is not None else block_size

    return cast(int | None, block_size), average_block_size


def _prepare_history(history: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], int]:
    if not history:
        raise ValueError("history must contain at least one path")

    arrays = {name: np.asarray(values) for name, values in history.items()}
    first_name, first_values = next(iter(arrays.items()))
    n_observations = _history_observation_count(first_name, first_values)

    for name, values in arrays.items():
        observation_count = _history_observation_count(name, values)
        if observation_count != n_observations:
            raise ValueError(
                f"history[{name!r}] has {observation_count} observations, "
                f"expected {n_observations}"
            )

    return arrays, n_observations


def _history_observation_count(name: str, values: np.ndarray) -> int:
    if values.ndim < 1:
        raise ValueError(
            f"history[{name!r}] must have shape [n_observations, ...]"
        )
    if values.shape[0] == 0:
        raise ValueError("history values must contain at least one observation")
    return int(values.shape[0])


def _prepare_scenario_ids(
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
    if (
        isinstance(average_block_size, bool)
        or not isinstance(average_block_size, Real)
        or not np.isfinite(average_block_size)
        or average_block_size < 1
    ):
        raise ValueError("average_block_size must be finite and at least 1")
    return float(average_block_size)


def _iid_indexes(
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


def _circular_block_indexes(
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
            block = block_start + np.arange(length, dtype=np.int64)
            indexes[scenario_index, position : position + length] = (
                block % n_observations
            )
            position += length

    return indexes


def _moving_block_indexes(
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


def _stationary_block_indexes(
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


__all__ = [
    "BootstrapDataModule",
    "BootstrapMethod",
]
