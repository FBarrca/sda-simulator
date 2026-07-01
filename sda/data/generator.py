from __future__ import annotations

import inspect
from collections.abc import Callable, Iterator, Mapping, Sequence
from numbers import Integral, Real
from typing import Any

import numpy as np

from sda.core import ScenarioBatch, ScenarioSpec
from sda.data._state import scenario_initial_state
from sda.data.module import DataModule

GeneratedScenarios = Mapping[str, Any] | ScenarioBatch | Sequence[ScenarioSpec]
ScenarioGenerator = Callable[..., GeneratedScenarios]
_GENERATOR_CONTEXT_NAMES = (
    "rng",
    "scenario_ids",
    "end_time",
    "horizon",
    "batch_size",
    "shape",
    "start",
    "stop",
    "n_scenarios",
)


class GeneratorDataModule(DataModule):
    """Data module backed by a user-supplied scenario generator."""

    def __init__(
        self,
        generator: ScenarioGenerator,
        *,
        end_time: float | None = None,
        horizon: int | None = None,
        n_scenarios: int,
        initial_state: Any = None,
        batch_size: int | None = None,
        seed: int | None = None,
        scenario_ids: Sequence[int] | None = None,
        scenario_seeds: Sequence[int] | None = None,
    ) -> None:
        """Create a generator-backed SimPy data module."""
        if not callable(generator):
            raise TypeError("generator must be callable")
        if end_time is None and horizon is None:
            raise ValueError("provide end_time or horizon")

        self.generator = generator
        self.end_time = (
            _nonnegative_float("end_time", end_time)
            if end_time is not None
            else float(_positive_int("horizon", horizon))
        )
        self.horizon = int(self.end_time) if horizon is None else _positive_int(
            "horizon",
            horizon,
        )
        self.n_scenarios = _positive_int("n_scenarios", n_scenarios)
        self.batch_size = (
            self.n_scenarios
            if batch_size is None
            else _positive_int("batch_size", batch_size)
        )
        self.initial_state = initial_state
        self.seed = seed
        self.scenario_ids = _prepare_scenario_ids(scenario_ids, self.n_scenarios)
        self.scenario_seeds = _prepare_scenario_seeds(
            scenario_seeds,
            self.n_scenarios,
            seed,
        )

    def batches(self, stage: str = "evaluate") -> Iterator[ScenarioBatch]:
        """Yield generated scenario batches."""
        del stage
        rng = np.random.default_rng(self.seed)

        for start in range(0, self.n_scenarios, self.batch_size):
            stop = min(start + self.batch_size, self.n_scenarios)
            batch_ids = self.scenario_ids[start:stop].tolist()
            generated = _call_generator(
                self.generator,
                {
                    "rng": rng,
                    "scenario_ids": batch_ids,
                    "end_time": self.end_time,
                    "horizon": self.horizon,
                    "batch_size": stop - start,
                    "shape": (stop - start, self.horizon),
                    "start": start,
                    "stop": stop,
                    "n_scenarios": self.n_scenarios,
                },
            )

            if isinstance(generated, ScenarioBatch):
                yield generated
                continue

            scenario_sequence = _as_scenario_sequence(generated)
            if scenario_sequence is not None:
                yield ScenarioBatch(scenario_sequence)
                continue

            if not isinstance(generated, Mapping):
                raise TypeError(
                    "generator must return a mapping, ScenarioBatch, or ScenarioSpec sequence"
                )

            yield ScenarioBatch(
                [
                    ScenarioSpec(
                        scenario_id=int(self.scenario_ids[index]),
                        end_time=self.end_time,
                        initial_state=scenario_initial_state(
                            self.initial_state,
                            index,
                            self.n_scenarios,
                        ),
                        data=_scenario_data(generated, index=index, start=start, stop=stop),
                        seed=None
                        if self.scenario_seeds is None
                        else int(self.scenario_seeds[index]),
                    )
                    for index in range(start, stop)
                ]
            )


def _positive_int(name: str, value: int | None) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return int(value)


def _nonnegative_float(name: str, value: float | None) -> float:
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
        raise ValueError("scenario_ids length must match n_scenarios")
    return np.asarray(scenario_ids)


def _prepare_scenario_seeds(
    scenario_seeds: Sequence[int] | None,
    n_scenarios: int,
    seed: int | None,
) -> np.ndarray | None:
    if scenario_seeds is not None:
        if len(scenario_seeds) != n_scenarios:
            raise ValueError("scenario_seeds length must match n_scenarios")
        return np.asarray(scenario_seeds, dtype=int)
    if seed is None:
        return None
    return np.random.default_rng(seed).integers(
        0,
        np.iinfo(np.int32).max,
        size=n_scenarios,
        dtype=np.int64,
    )


def _call_generator(
    generator: ScenarioGenerator,
    context: dict[str, Any],
) -> GeneratedScenarios:
    try:
        signature = inspect.signature(generator)
    except (TypeError, ValueError):
        return generator(**context)

    parameters = signature.parameters
    if not parameters:
        return generator()
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    ):
        return generator(**context)

    kwargs = {
        name: context[name]
        for name, parameter in parameters.items()
        if name in context
        and parameter.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }
    missing = [
        parameter.name
        for parameter in parameters.values()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
        and parameter.name not in kwargs
    ]
    if missing:
        missing_names = "', '".join(missing)
        context_names = "', '".join(_GENERATOR_CONTEXT_NAMES)
        raise TypeError(
            f"generator has unsupported required parameter(s) '{missing_names}'; "
            f"choose from '{context_names}' or accept **kwargs"
        )
    return generator(**kwargs)


def _as_scenario_sequence(value: Any) -> tuple[ScenarioSpec, ...] | None:
    if isinstance(value, (str, bytes, Mapping)):
        return None
    try:
        items = tuple(value)
    except TypeError:
        return None
    if not items:
        return None
    if all(isinstance(item, ScenarioSpec) for item in items):
        return items
    return None


def _scenario_data(
    generated: Mapping[str, Any],
    *,
    index: int,
    start: int,
    stop: int,
) -> dict[str, Any]:
    data = {}
    batch_size = stop - start
    for name, values in generated.items():
        array = np.asarray(values)
        if array.ndim < 1 or array.shape[0] != batch_size:
            raise ValueError(
                f"generated path {name!r} must have one entry per scenario "
                f"in the current batch"
            )
        data[name] = array[index - start]
    return data


__all__ = [
    "GeneratedScenarios",
    "GeneratorDataModule",
    "ScenarioGenerator",
]
