from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from sda.data import ScenarioBatch


@dataclass(frozen=True)
class StepRecord:
    """Data captured for one simulated period of a scenario batch.

    The simulator creates one record after each transition and passes it to
    step-level metrics. Fields such as ``state``, ``decision``, and
    ``next_state`` use the representation chosen by the model. Values that vary
    by scenario should keep the batch dimension first and align with
    ``scenario_ids``.
    """

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
    """Data captured after a full rollout of a scenario batch.

    The simulator creates one record per batch after all time periods have run
    and passes it to trajectory-level metrics. ``steps`` contains the collected
    :class:`StepRecord` objects when history is enabled on the simulator.
    """

    scenario_ids: Sequence[int]
    total_cost: Any
    final_state: Any
    steps: list[StepRecord]


class Policy:
    """Base interface for decision rules.

    Subclass ``Policy`` and implement :meth:`act` with the logic that maps the
    current state and time to a decision. The decision can be any Python object
    that the corresponding :class:`SDAModel` understands.
    """

    def act(self, state: Any, t: int, history: list[StepRecord]) -> Any:
        """Return the decision for the current batch and time period.

        Parameters
        ----------
        state
            Current state returned by ``SDAModel.initial_state`` or the previous
            ``SDAModel.transition`` call.
        t
            Zero-based time index within the scenario horizon.
        history
            Completed step records for the current batch. This list is empty at
            ``t == 0`` and remains empty when ``Simulator(keep_history=False)``
            is used.

        Returns
        -------
        Any
            A decision object accepted by the model's ``transition`` and
            ``cost`` methods.
        """
        raise NotImplementedError


class SDAModel:
    """Base interface for a sequential decision model.

    A model defines how a batch of scenarios starts, evolves, and accumulates
    cost. Subclasses usually override :meth:`transition` and :meth:`cost`; they
    can also override :meth:`initial_state`, :meth:`decide`, and :meth:`info` to
    customize setup, policy dispatch, and metric diagnostics.
    """

    def __init__(self, policy: Policy) -> None:
        """Create a model using ``policy`` as the default decision rule."""
        self.policy = policy

    def initial_state(self, batch: ScenarioBatch) -> Any:
        """Return the starting state for one scenario batch.

        The default implementation uses ``batch.initial_state`` exactly as
        provided by the scenario loader. Override this method when the model
        needs to build a richer state object, validate inputs, or copy mutable
        data before the rollout begins.
        """
        return batch.initial_state

    def decide(self, state: Any, t: int, history: list[StepRecord]) -> Any:
        """Return the decision to apply at the current state and time.

        The default implementation delegates to ``self.policy.act``. Override
        this method when decision logic needs direct access to model internals
        or when combining several policies.
        """
        return self.policy.act(state, t, history)

    def transition(
        self,
        state: Any,
        decision: Any,
        exogenous: dict[str, Any],
        t: int,
    ) -> Any:
        """Advance the model by one period.

        Parameters
        ----------
        state
            Current batch state.
        decision
            Decision returned by :meth:`decide` for the same state and time.
        exogenous
            Current-period exogenous values. Each entry is sliced from the
            scenario batch at time ``t`` and is batch-first, so
            ``exogenous[name][i]`` belongs to ``scenario_ids[i]``.
        t
            Zero-based time index.

        Returns
        -------
        Any
            The next state for the batch. This value becomes the ``state``
            argument at the next time step and is stored on ``StepRecord``.
        """
        raise NotImplementedError

    def cost(
        self,
        state: Any,
        decision: Any,
        exogenous: dict[str, Any],
        next_state: Any,
        t: int,
    ) -> Any:
        """Compute the cost incurred by one transition.

        Return either a scalar, which is broadcast to every scenario in the
        batch, or a one-dimensional array-like value with one cost per scenario.
        The simulator converts the result to a float NumPy vector before adding
        it to total cost and sending it to metrics.
        """
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
        """Return optional diagnostic values for metrics and analysis.

        The returned dictionary is stored on ``StepRecord.info``. Custom metrics
        can read it to log model-specific values such as service level,
        utilization, inventory position, or constraint violations. The default
        implementation records no extra information.
        """
        return {}
