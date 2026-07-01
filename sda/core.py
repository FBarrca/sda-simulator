from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Literal

import numpy as np
import simpy

EventLevel = Literal["event", "trajectory"]


@dataclass(frozen=True)
class ScenarioSpec:
    """One independent SimPy simulation scenario."""

    scenario_id: int
    end_time: float
    initial_state: Any = None
    data: Mapping[str, Any] = field(default_factory=dict)
    seed: int | None = None

    def __post_init__(self) -> None:
        """Validate scenario identity and end time."""
        if isinstance(self.scenario_id, bool) or not isinstance(self.scenario_id, int):
            raise ValueError("scenario_id must be an integer")
        if (
            isinstance(self.end_time, bool)
            or not isinstance(self.end_time, Real)
            or not np.isfinite(self.end_time)
            or self.end_time < 0
        ):
            raise ValueError("end_time must be finite and non-negative")
        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, int)
        ):
            raise ValueError("seed must be an integer or None")


@dataclass(frozen=True)
class ScenarioBatch:
    """A batch of independent SimPy scenario specifications."""

    scenarios: Sequence[ScenarioSpec]

    def __post_init__(self) -> None:
        """Validate that the batch contains scenarios."""
        if len(self.scenarios) == 0:
            raise ValueError("scenarios must contain at least one ScenarioSpec")
        for scenario in self.scenarios:
            if not isinstance(scenario, ScenarioSpec):
                raise TypeError("scenarios must contain ScenarioSpec objects")

    @property
    def batch_size(self) -> int:
        """Return the number of scenarios in this batch."""
        return len(self.scenarios)

    @property
    def scenario_ids(self) -> list[int]:
        """Return scenario ids in batch order."""
        return [scenario.scenario_id for scenario in self.scenarios]

    @property
    def end_time(self) -> float:
        """Return the maximum end time across scenarios in this batch."""
        return max(float(scenario.end_time) for scenario in self.scenarios)


@dataclass(frozen=True)
class EventRecord:
    """One metric observation emitted during or after a SimPy scenario."""

    name: str
    value: float
    scenario_id: int | None
    time: float
    level: EventLevel = "event"
    tags: Mapping[str, str] = field(default_factory=dict)


class Policy:
    """Base interface for decision rules used by SimPy processes."""

    def act(
        self,
        state: Any,
        env: simpy.Environment,
        history: list[EventRecord],
    ) -> Any:
        """Return a decision from the current state and SimPy environment."""
        raise NotImplementedError


class SDAModel:
    """Base interface for SimPy-native sequential decision models."""

    def __init__(self, policy: Policy) -> None:
        """Create a model using ``policy`` as the default decision rule."""
        self.policy = policy

    def build(
        self,
        env: simpy.Environment,
        scenario: ScenarioSpec,
        recorder: Any,
    ) -> Any:
        """Register SimPy processes for one scenario and return model state."""
        raise NotImplementedError

    def finalize(
        self,
        state: Any,
        scenario: ScenarioSpec,
        recorder: Any,
    ) -> None:
        """Record optional end-of-scenario diagnostics."""


__all__ = [
    "EventLevel",
    "EventRecord",
    "Policy",
    "SDAModel",
    "ScenarioBatch",
    "ScenarioSpec",
]
