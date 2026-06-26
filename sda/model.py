from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from sda.data import ScenarioBatch


@dataclass(frozen=True)
class StepRecord:
    """One simulated period for a batch of scenarios."""

    scenario_ids: Sequence[int]
    t: int
    state: Any
    decision: Any
    exogenous: dict[str, Any]
    next_state: Any
    cost: Any
    info: dict[str, Any]


@dataclass(frozen=True)
class TrajectoryRecord:
    """A complete simulated trajectory for a batch of scenarios."""

    scenario_ids: Sequence[int]
    total_cost: Any
    final_state: Any
    steps: list[StepRecord]


class Policy:
    """Decision rule interface."""

    def act(self, state: Any, t: int, history: list[StepRecord]) -> Any:
        raise NotImplementedError


class SDAModel:
    """Sequential decision model interface."""

    def __init__(self, policy: Policy) -> None:
        self.policy = policy

    def initial_state(self, batch: ScenarioBatch) -> Any:
        return batch.initial_state

    def decide(self, state: Any, t: int, history: list[StepRecord]) -> Any:
        return self.policy.act(state, t, history)

    def transition(
        self,
        state: Any,
        decision: Any,
        exogenous: dict[str, Any],
        t: int,
    ) -> Any:
        raise NotImplementedError

    def cost(
        self,
        state: Any,
        decision: Any,
        exogenous: dict[str, Any],
        next_state: Any,
        t: int,
    ) -> Any:
        raise NotImplementedError

    def info(
        self,
        state: Any,
        decision: Any,
        exogenous: dict[str, Any],
        next_state: Any,
        cost: Any,
        t: int,
    ) -> dict[str, Any]:
        return {}
