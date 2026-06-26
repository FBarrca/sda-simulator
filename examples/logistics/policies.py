from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import replace
from math import ceil
from typing import Any

import numpy as np

from examples.logistics.domain import Assignment, LogisticsState, Order, VehicleState
from examples.logistics.network import SKUS, WAREHOUSES, distance_km
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
            sim_state = _clone_logistics_state(state)
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


class SyntheticRolloutSampler:
    """Small one-scenario sampler used by the lookahead policy."""

    def __init__(self, *, history_days: int = 365, seed: int | None = None) -> None:
        from examples.logistics.data import synthetic_history

        self.history = synthetic_history(history_days, seed=seed)
        self.rng = np.random.default_rng(seed)
        self.start_day = 0
        self.scenario_key = 0

    def reset(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.start_day = int(self.rng.integers(0, self.history.days))
        self.scenario_key = int(seed)

    def sample(self, state: LogisticsState, step: int) -> dict[str, np.ndarray]:
        source_day = (self.start_day + step) % self.history.days
        raw_orders = self.history.orders[source_day]
        orders = np.empty(1, dtype=object)
        orders[0] = tuple(
            _rollout_order(
                order,
                day=state.time,
                scenario_key=self.scenario_key,
                step=step,
                index=index,
            )
            for index, order in enumerate(raw_orders)
        )
        return {
            "orders": orders,
            "traffic_multiplier": self.history.traffic_multiplier[source_day][None, ...],
            "vehicle_outages": self.history.vehicle_outages[source_day][None, ...],
        }


class NearestFeasiblePolicy(DispatchPolicy):
    """Dispatch pending orders FIFO from the nearest warehouse that can serve them."""

    name = "nearest_feasible"

    def _act_one(self, state: LogisticsState) -> tuple[Assignment, ...]:
        return _build_assignments(
            state,
            orders=sorted(state.pending_orders, key=lambda order: (order.day, order.order_id)),
            warehouse_key=lambda order, warehouse: distance_km(warehouse, order.destination),
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
            warehouse_key=lambda order, warehouse: distance_km(warehouse, order.destination),
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


def feasible_assignments(state: LogisticsState) -> list[Assignment]:
    assignments: list[Assignment] = []
    for order in sorted(
        state.pending_orders,
        key=lambda item: (item.day, item.deadline, -item.priority, item.order_id),
    ):
        for warehouse in WAREHOUSES:
            if state.inventory[warehouse].get(order.sku, 0) < order.quantity:
                continue
            for vehicle in _available_vehicles(state, warehouse, used_vehicles=set()):
                if vehicle.capacity < order.quantity:
                    continue
                assignments.append(
                    Assignment(
                        order_id=order.order_id,
                        warehouse=warehouse,
                        vehicle_id=vehicle.vehicle_id,
                    )
                )
    return assignments


def conflict_free_subset(
    assignments: Iterable[Assignment],
    state: LogisticsState,
    *,
    orders_by_id: dict[int, Order] | None = None,
) -> tuple[Assignment, ...]:
    orders = orders_by_id or _orders_by_id(state)
    chosen: list[Assignment] = []
    used_orders: set[int] = set()
    used_vehicles: set[str] = set()
    reserved_inventory: dict[tuple[str, str], int] = defaultdict(int)

    for assignment in assignments:
        order = orders.get(assignment.order_id)
        if order is None:
            continue
        if assignment.order_id in used_orders or assignment.vehicle_id in used_vehicles:
            continue
        available = state.inventory[assignment.warehouse].get(order.sku, 0)
        key = (assignment.warehouse, order.sku)
        if reserved_inventory[key] + order.quantity > available:
            continue

        chosen.append(assignment)
        used_orders.add(assignment.order_id)
        used_vehicles.add(assignment.vehicle_id)
        reserved_inventory[key] += order.quantity

    return tuple(chosen)


def assignment_score(
    state: LogisticsState,
    assignment: Assignment,
    *,
    orders_by_id: dict[int, Order] | None = None,
) -> float:
    orders = orders_by_id or _orders_by_id(state)
    order = orders[assignment.order_id]
    distance = lane_distance_km(assignment.warehouse, order.destination)
    duration = _travel_duration_days(distance)
    delivery_slack = order.deadline - (state.time + duration)
    urgency_pressure = max(0, 4 - delivery_slack) * 55.0
    rescue_pressure = max(0, -delivery_slack) * 180.0
    return (
        order.priority * 200.0
        + min(order.quantity, 32) * 6.0
        + urgency_pressure
        + rescue_pressure
        - duration * 35.0
        - distance * 0.45
    )


def lane_distance_km(warehouse: str, customer: str) -> float:
    return distance_km(warehouse, customer)


def _build_assignments(
    state: LogisticsState,
    *,
    orders: Iterable[Order],
    warehouse_key=None,
    assignment_score=None,
    reverse: bool = False,
) -> tuple[Assignment, ...]:
    assignments: list[Assignment] = []
    used_vehicles: set[str] = set()
    reserved_inventory: dict[tuple[str, str], int] = defaultdict(int)

    for order in orders:
        candidates: list[tuple[float, str, VehicleState]] = []
        warehouses = WAREHOUSES
        if warehouse_key is not None:
            warehouses = tuple(sorted(WAREHOUSES, key=lambda warehouse: warehouse_key(order, warehouse)))

        for warehouse in warehouses:
            available = state.inventory[warehouse].get(order.sku, 0)
            if reserved_inventory[(warehouse, order.sku)] + order.quantity > available:
                continue
            for vehicle in _available_vehicles(state, warehouse, used_vehicles):
                if vehicle.capacity < order.quantity:
                    continue
                if assignment_score is None:
                    score = distance_km(warehouse, order.destination)
                else:
                    score = float(assignment_score(order, warehouse, vehicle))
                candidates.append((score, warehouse, vehicle))

        if not candidates:
            continue

        candidates.sort(key=lambda item: item[0], reverse=reverse)
        _, warehouse, vehicle = candidates[0]
        assignments.append(
            Assignment(
                order_id=order.order_id,
                warehouse=warehouse,
                vehicle_id=vehicle.vehicle_id,
            )
        )
        used_vehicles.add(vehicle.vehicle_id)
        reserved_inventory[(warehouse, order.sku)] += order.quantity

    return tuple(assignments)


def _available_vehicles(
    state: LogisticsState,
    warehouse: str,
    used_vehicles: set[str],
) -> list[VehicleState]:
    vehicles = [
        vehicle
        for vehicle in state.vehicles.values()
        if vehicle.location == warehouse
        and vehicle.status == "available"
        and vehicle.vehicle_id not in used_vehicles
    ]
    return sorted(vehicles, key=lambda vehicle: vehicle.capacity)


def _risk_score(
    state: LogisticsState,
    order: Order,
    warehouse: str,
    vehicle: VehicleState,
) -> float:
    distance = distance_km(warehouse, order.destination)
    stock_after = state.inventory[warehouse][order.sku] - order.quantity
    stock_scarcity = order.quantity / max(state.inventory[warehouse][order.sku], 1)
    slack = order.deadline - state.time
    urgency = max(0, 5 - slack)
    spare_capacity = max(vehicle.capacity - order.quantity, 0)
    return (
        order.priority * 1000.0
        + urgency * 130.0
        - distance * 0.42
        - stock_scarcity * 95.0
        - max(0, 40 - stock_after) * 1.5
        - spare_capacity * 0.6
    )


def _orders_by_id(state: LogisticsState) -> dict[int, Order]:
    return {order.order_id: order for order in state.pending_orders}


def _travel_duration_days(distance: float) -> int:
    return max(1, int(ceil(distance / 520.0)))


def _clone_logistics_state(state: LogisticsState) -> LogisticsState:
    return LogisticsState(
        inventory={
            warehouse: dict(sku_inventory)
            for warehouse, sku_inventory in state.inventory.items()
        },
        vehicles=dict(state.vehicles),
        pending_orders=tuple(state.pending_orders),
        completed_orders=tuple(state.completed_orders),
        time=int(state.time),
        day_of_week=int(state.day_of_week),
    )


def _rollout_order(
    order: Order,
    *,
    day: int,
    scenario_key: int,
    step: int,
    index: int,
) -> Order:
    lead_time = max(1, order.deadline - order.day)
    return replace(
        order,
        order_id=900_000_000_000 + scenario_key * 100_000 + step * 1000 + index,
        day=day,
        deadline=day + lead_time,
        origin=None,
    )
