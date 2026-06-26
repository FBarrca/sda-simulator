from __future__ import annotations

from collections.abc import Iterable

from examples.logistics.domain import Assignment, LogisticsState, Order, VehicleState
from examples.logistics.network import WAREHOUSES, distance_km
from sda.model import Policy, StepRecord


class DispatchPolicy(Policy):
    name = "dispatch"

    def act(self, state, t: int, history: list[StepRecord]):
        single_state = isinstance(state, LogisticsState)
        states = [state] if single_state else list(state)
        decisions = [self._act_one(logistics_state) for logistics_state in states]
        return decisions[0] if single_state else decisions

    def _act_one(self, state: LogisticsState) -> tuple[Assignment, ...]:
        raise NotImplementedError


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

    for order in orders:
        candidates: list[tuple[float, str, VehicleState]] = []
        warehouses = WAREHOUSES
        if warehouse_key is not None:
            warehouses = tuple(sorted(WAREHOUSES, key=lambda warehouse: warehouse_key(order, warehouse)))

        for warehouse in warehouses:
            if state.inventory[warehouse].get(order.sku, 0) < order.quantity:
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
