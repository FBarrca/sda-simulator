from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable
from math import ceil

from examples.logistics.domain import Assignment, LogisticsState, Order, VehicleState
from examples.logistics.network import SKUS, WAREHOUSES, distance_km

WarehouseKey = Callable[[Order, str], float]
AssignmentScore = Callable[[Order, str, VehicleState], float]


def feasible_assignments(state: LogisticsState) -> list[Assignment]:
    assignments: list[Assignment] = []
    for order in sorted(
        state.pending_orders,
        key=lambda item: (item.day, item.deadline, -item.priority, item.order_id),
    ):
        for warehouse in WAREHOUSES:
            if state.inventory[warehouse].get(order.sku, 0) < order.quantity:
                continue
            for vehicle in available_vehicles(state, warehouse, used_vehicles=set()):
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
    orders = orders_by_id or index_orders_by_id(state)
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
    orders = orders_by_id or index_orders_by_id(state)
    order = orders[assignment.order_id]
    distance = lane_distance_km(assignment.warehouse, order.destination)
    duration = travel_duration_days(distance)
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


def build_assignments(
    state: LogisticsState,
    *,
    orders: Iterable[Order],
    warehouse_key: WarehouseKey | None = None,
    assignment_score: AssignmentScore | None = None,
    reverse: bool = False,
) -> tuple[Assignment, ...]:
    assignments: list[Assignment] = []
    used_vehicles: set[str] = set()
    reserved_inventory: dict[tuple[str, str], int] = defaultdict(int)

    for order in orders:
        candidates: list[tuple[float, str, VehicleState]] = []
        warehouses = WAREHOUSES
        if warehouse_key is not None:
            warehouses = tuple(
                sorted(WAREHOUSES, key=lambda warehouse: warehouse_key(order, warehouse))
            )

        for warehouse in warehouses:
            available = state.inventory[warehouse].get(order.sku, 0)
            if reserved_inventory[(warehouse, order.sku)] + order.quantity > available:
                continue
            for vehicle in available_vehicles(state, warehouse, used_vehicles):
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


def available_vehicles(
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


def risk_score(
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


def index_orders_by_id(state: LogisticsState) -> dict[int, Order]:
    return {order.order_id: order for order in state.pending_orders}


def travel_duration_days(distance: float) -> int:
    return max(1, int(ceil(distance / 520.0)))
