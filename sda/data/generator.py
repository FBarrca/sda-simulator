from __future__ import annotations

import inspect
from collections.abc import Callable, Iterator, Mapping, Sequence
from numbers import Integral
from typing import Any

import numpy as np

from sda.core import ScenarioBatch
from sda.data._state import slice_initial_state
from sda.data.module import DataModule

GeneratedScenarios = Mapping[str, Any] | ScenarioBatch
ScenarioGenerator = Callable[..., GeneratedScenarios]
_GENERATOR_CONTEXT_NAMES = (
    "rng",
    "scenario_ids",
    "horizon",
    "batch_size",
    "shape",
    "start",
    "stop",
    "n_scenarios",
)


class GeneratorDataModule(DataModule):
    """Data module backed by a user-supplied generator function.

    Use this module when futures come from a statistical sampler, forecasting
    model, domain simulator, service call, or any source that can produce one
    batch at a time.
    """

    def __init__(
        self,
        generator: ScenarioGenerator,
        *,
        horizon: int,
        n_scenarios: int,
        initial_state: Any = 0,
        batch_size: int | None = None,
        seed: int | None = None,
        scenario_ids: Sequence[int] | None = None,
    ) -> None:
        """Create a generator-backed data module."""
        if not callable(generator):
            raise TypeError("generator must be callable")

        self.generator = generator
        self.horizon = _positive_int("horizon", horizon)
        self.n_scenarios = _positive_int("n_scenarios", n_scenarios)
        self.batch_size = (
            self.n_scenarios
            if batch_size is None
            else _positive_int("batch_size", batch_size)
        )
        self.initial_state = initial_state
        self.seed = seed
        self.scenario_ids = _prepare_scenario_ids(scenario_ids, self.n_scenarios)

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

            if not isinstance(generated, Mapping):
                raise TypeError(
                    "generator must return an exogenous mapping or ScenarioBatch"
                )

            yield ScenarioBatch(
                initial_state=slice_initial_state(
                    self.initial_state,
                    start,
                    stop,
                    self.n_scenarios,
                ),
                exogenous=generated,
                scenario_ids=batch_ids,
            )


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return int(value)


def _prepare_scenario_ids(
    scenario_ids: Sequence[int] | None,
    n_scenarios: int,
) -> np.ndarray:
    if scenario_ids is None:
        return np.arange(n_scenarios)
    if len(scenario_ids) != n_scenarios:
        raise ValueError("scenario_ids length must match n_scenarios")
    return np.asarray(scenario_ids)


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


__all__ = [
    "GeneratedScenarios",
    "GeneratorDataModule",
    "ScenarioGenerator",
]
