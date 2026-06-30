from __future__ import annotations

from typing import Any

import numpy as np

from examples.logistics.assignment import (
    assignment_score,
    build_assignments as _build_assignments,
    conflict_free_subset,
    feasible_assignments,
    index_orders_by_id as _orders_by_id,
    lane_distance_km,
    risk_score as _risk_score,
)
from examples.logistics.domain import Assignment, LogisticsState, Order, clone_state
from examples.logistics.network import SKUS, WAREHOUSES
from examples.logistics.rollout import SyntheticRolloutSampler
from sda.model import Policy, StepRecord


class DispatchPolicy(Policy):
    name = "dispatch"

    def act(self, state, t: int, history: list[StepRecord]):
        single_state = isinstance(state, LogisticsState)
        states = [state] if single_state else list(state)
        decisions = [self._act_one(logistics_state) for logistics_state in states]
        return decisions[0] if single_state else decisions

    def decide(self, state: LogisticsState) -> tuple[Assignment, ...]:
        return self._act_one(state)

    def _act_one(self, state: LogisticsState) -> tuple[Assignment, ...]:
        raise NotImplementedError


class RandomPolicy(DispatchPolicy):
    """Randomly shuffle feasible assignments and greedily keep a conflict-free set."""

    name = "random"

    def __init__(self, seed: int | None = None) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def _act_one(self, state: LogisticsState) -> tuple[Assignment, ...]:
        assignments = feasible_assignments(state)
        self.rng.shuffle(assignments)
        return conflict_free_subset(assignments, state)


class GreedyPolicy(DispatchPolicy):
    """Prefer the shortest warehouse-to-customer lane first."""

    name = "greedy"

    def _act_one(self, state: LogisticsState) -> tuple[Assignment, ...]:
        orders_by_id = _orders_by_id(state)
        assignments = sorted(
            feasible_assignments(state),
            key=lambda assignment: (
                lane_distance_km(
                    assignment.warehouse,
                    orders_by_id[assignment.order_id].destination,
                ),
                orders_by_id[assignment.order_id].deadline,
                assignment.order_id,
                assignment.warehouse,
                assignment.vehicle_id,
            ),
        )
        return conflict_free_subset(assignments, state, orders_by_id=orders_by_id)


class PriorityPolicy(DispatchPolicy):
    """Score priority, quantity, deadline pressure, duration, and lane distance."""

    name = "priority"

    def _act_one(self, state: LogisticsState) -> tuple[Assignment, ...]:
        orders_by_id = _orders_by_id(state)
        candidates = [
            (assignment_score(state, assignment, orders_by_id=orders_by_id), assignment)
            for assignment in feasible_assignments(state)
        ]
        candidates.sort(
            key=lambda item: (
                -item[0],
                orders_by_id[item[1].order_id].deadline,
                item[1].order_id,
                item[1].warehouse,
                item[1].vehicle_id,
            )
        )
        return conflict_free_subset(
            [assignment for _, assignment in candidates],
            state,
            orders_by_id=orders_by_id,
        )


class MilpPolicy(DispatchPolicy):
    """Globally choose assignments that maximize the priority-distance score."""

    name = "milp_distance_priority"

    def __init__(
        self,
        *,
        time_limit: float = 0.25,
        mip_rel_gap: float = 0.01,
        fallback_policy: DispatchPolicy | None = None,
    ) -> None:
        self.time_limit = float(time_limit)
        self.mip_rel_gap = float(mip_rel_gap)
        self.fallback_policy = fallback_policy or PriorityPolicy()

    def _act_one(self, state: LogisticsState) -> tuple[Assignment, ...]:
        orders_by_id = _orders_by_id(state)
        candidates = feasible_assignments(state)
        if not candidates:
            return ()

        scores = np.asarray(
            [
                assignment_score(state, assignment, orders_by_id=orders_by_id)
                for assignment in candidates
            ],
            dtype=float,
        )
        selected = self._solve_with_scipy(state, candidates, scores, orders_by_id)
        if selected is None:
            return self.fallback_policy._act_one(state)

        chosen = [assignment for keep, assignment in zip(selected, candidates, strict=True) if keep]
        return conflict_free_subset(chosen, state, orders_by_id=orders_by_id)

    def _solve_with_scipy(
        self,
        state: LogisticsState,
        candidates: list[Assignment],
        scores: np.ndarray,
        orders_by_id: dict[int, Order],
    ) -> np.ndarray | None:
        try:
            from scipy.optimize import Bounds, LinearConstraint, milp
        except ImportError:
            return None

        rows: list[list[float]] = []
        upper_bounds: list[float] = []

        for order_id in sorted({assignment.order_id for assignment in candidates}):
            rows.append(
                [
                    1.0 if assignment.order_id == order_id else 0.0
                    for assignment in candidates
                ]
            )
            upper_bounds.append(1.0)

        for vehicle_id in sorted({assignment.vehicle_id for assignment in candidates}):
            rows.append(
                [
                    1.0 if assignment.vehicle_id == vehicle_id else 0.0
                    for assignment in candidates
                ]
            )
            upper_bounds.append(1.0)

        for warehouse in WAREHOUSES:
            for sku in SKUS:
                row = [
                    float(orders_by_id[assignment.order_id].quantity)
                    if assignment.warehouse == warehouse
                    and orders_by_id[assignment.order_id].sku == sku
                    else 0.0
                    for assignment in candidates
                ]
                if any(row):
                    rows.append(row)
                    upper_bounds.append(float(state.inventory[warehouse].get(sku, 0)))

        constraint_matrix = np.asarray(rows, dtype=float)
        constraints = LinearConstraint(
            constraint_matrix,
            lb=np.full(len(rows), -np.inf, dtype=float),
            ub=np.asarray(upper_bounds, dtype=float),
        )
        result = milp(
            c=-scores,
            integrality=np.ones(len(candidates), dtype=int),
            bounds=Bounds(0, 1),
            constraints=constraints,
            options={"time_limit": self.time_limit, "mip_rel_gap": self.mip_rel_gap},
        )
        if result.x is None or not result.success:
            return None
        return np.asarray(result.x >= 0.5, dtype=bool)


class LookaheadRolloutPolicy(DispatchPolicy):
    """Evaluate first-day candidates by rolling out sampled futures."""

    name = "lookahead_rollout"

    def __init__(
        self,
        *,
        model: Any | None = None,
        sampler: SyntheticRolloutSampler | None = None,
        horizon: int = 3,
        scenarios: int = 2,
        seed: int = 1729,
        base_policy: DispatchPolicy | None = None,
    ) -> None:
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if scenarios <= 0:
            raise ValueError("scenarios must be positive")

        self.model = model
        self.sampler = sampler
        self.horizon = int(horizon)
        self.scenarios = int(scenarios)
        self.seed = int(seed)
        self.base = base_policy or PriorityPolicy()
        if self.model is not None and self.sampler is None:
            self.sampler = SyntheticRolloutSampler(seed=self.seed)

    def bind_rollout_model(self, model: Any) -> None:
        self.model = model
        if self.sampler is None:
            self.sampler = SyntheticRolloutSampler(seed=self.seed)

    def _act_one(self, state: LogisticsState) -> tuple[Assignment, ...]:
        if self.model is None or self.sampler is None:
            return self.base._act_one(state)

        candidates = self._candidate_decisions(state)
        return max(candidates, key=lambda decision: self._rollout_value(state, decision))

    def _candidate_decisions(self, state: LogisticsState) -> tuple[tuple[Assignment, ...], ...]:
        candidates = (
            self.base._act_one(state),
            GreedyPolicy()._act_one(state),
            (),
        )
        unique: list[tuple[Assignment, ...]] = []
        seen: set[tuple[Assignment, ...]] = set()
        for decision in candidates:
            key = tuple(decision)
            if key not in seen:
                seen.add(key)
                unique.append(tuple(decision))
        return tuple(unique)

    def _rollout_value(
        self,
        state: LogisticsState,
        first_decision: tuple[Assignment, ...],
    ) -> float:
        total_reward = 0.0
        for scenario in range(self.scenarios):
            self.sampler.reset(self.seed + scenario)
            sim_state = clone_state(state)
            decision = first_decision

            for step in range(self.horizon):
                exogenous = self.sampler.sample(sim_state, step)
                next_state = self.model.transition(
                    [sim_state],
                    [decision],
                    exogenous,
                    sim_state.time,
                )[0]
                cost = self.model.cost(
                    [sim_state],
                    [decision],
                    exogenous,
                    [next_state],
                    sim_state.time,
                )
                total_reward -= float(np.asarray(cost, dtype=float)[0])
                sim_state = next_state
                decision = self.base._act_one(sim_state)

        return total_reward / self.scenarios


class NearestFeasiblePolicy(DispatchPolicy):
    """Dispatch pending orders FIFO from the nearest warehouse that can serve them."""

    name = "nearest_feasible"

    def _act_one(self, state: LogisticsState) -> tuple[Assignment, ...]:
        return _build_assignments(
            state,
            orders=sorted(state.pending_orders, key=lambda order: (order.day, order.order_id)),
            warehouse_key=lambda order, warehouse: lane_distance_km(
                warehouse,
                order.destination,
            ),
        )


class PriorityDeadlinePolicy(DispatchPolicy):
    """Dispatch high-priority and tight-deadline orders first."""

    name = "priority_deadline"

    def _act_one(self, state: LogisticsState) -> tuple[Assignment, ...]:
        return _build_assignments(
            state,
            orders=sorted(
                state.pending_orders,
                key=lambda order: (-order.priority, order.deadline, order.day, order.order_id),
            ),
            warehouse_key=lambda order, warehouse: lane_distance_km(
                warehouse,
                order.destination,
            ),
        )


class RiskAwareDispatchPolicy(DispatchPolicy):
    """Score priority, deadline slack, lane distance, stock scarcity, and fit."""

    name = "risk_aware"

    def _act_one(self, state: LogisticsState) -> tuple[Assignment, ...]:
        orders = sorted(
            state.pending_orders,
            key=lambda order: (
                -order.priority,
                order.deadline - state.time,
                order.day,
                order.order_id,
            ),
        )
        return _build_assignments(
            state,
            orders=orders,
            assignment_score=lambda order, warehouse, vehicle: _risk_score(
                state,
                order,
                warehouse,
                vehicle,
            ),
            reverse=True,
        )
