from __future__ import annotations

from dataclasses import dataclass, replace
from math import ceil

import numpy as np
import simpy

from examples.logistics.domain import (
    Assignment,
    LogisticsState,
    Order,
    OrderOutcome,
    VehicleState,
    clone_state,
)
from examples.logistics.network import (
    CUSTOMER_INDEX,
    VEHICLE_INDEX,
    VEHICLES,
    WAREHOUSE_INDEX,
    distance_km,
)
from sda import Policy, Recorder, SDAModel, ScenarioSpec


@dataclass
class LogisticsSimulationState:
    """Mutable holder for one SimPy logistics scenario."""

    current: LogisticsState


class LogisticsModel(SDAModel):
    """Road-freight dispatch model for batched logistics scenarios."""

    def __init__(
        self,
        policy: Policy,
        *,
        km_cost: float = 0.55,
        unit_handling_cost: float = 1.2,
        late_cost_per_priority_unit_day: float = 32.0,
        backlog_cost_per_priority_day: float = 2.0,
        waiting_cost_per_priority_day: float = 0.0,
        invalid_assignment_cost: float = 20.0,
    ) -> None:
        super().__init__(policy)
        self.km_cost = float(km_cost)
        self.unit_handling_cost = float(unit_handling_cost)
        self.late_cost_per_priority_unit_day = float(late_cost_per_priority_unit_day)
        self.backlog_cost_per_priority_day = float(backlog_cost_per_priority_day)
        # Per-day charge on *every* waiting order, not only overdue ones. Default
        # 0.0 keeps the reference objective; raising it prices backlog properly.
        self.waiting_cost_per_priority_day = float(waiting_cost_per_priority_day)
        self.invalid_assignment_cost = float(invalid_assignment_cost)
        bind_rollout_model = getattr(policy, "bind_rollout_model", None)
        if callable(bind_rollout_model):
            bind_rollout_model(self)

    def build(
        self,
        env: simpy.Environment,
        scenario: ScenarioSpec,
        recorder: Recorder,
    ) -> LogisticsSimulationState:
        """Register daily logistics dispatch processes for one scenario."""
        state = LogisticsSimulationState(current=clone_state(scenario.initial_state))
        env.process(self._run(env, scenario, recorder, state))
        return state

    def _run(
        self,
        env: simpy.Environment,
        scenario: ScenarioSpec,
        recorder: Recorder,
        state: LogisticsSimulationState,
    ):
        orders_path = np.asarray(scenario.data["orders"], dtype=object)
        traffic_path = np.asarray(scenario.data["traffic_multiplier"], dtype=float)
        outages_path = np.asarray(scenario.data["vehicle_outages"], dtype=bool)

        for index, new_orders in enumerate(orders_path):
            assignments = self.policy.act(state.current, env, recorder.history)
            next_state, info, cost = self._transition_one(
                state=state.current,
                assignments=tuple(assignments),
                new_orders=tuple(new_orders),
                traffic=traffic_path[index],
                outages=outages_path[index],
                t=state.current.time,
            )
            state.current = next_state
            recorder.cost(cost)
            for name, value in info.items():
                recorder.log(name, value)
            yield env.timeout(1.0)

    def _transition_one(
        self,
        *,
        state: LogisticsState,
        assignments: tuple[Assignment, ...],
        new_orders: tuple[Order, ...],
        traffic: np.ndarray,
        outages: np.ndarray,
        t: int,
    ) -> tuple[LogisticsState, dict[str, float], float]:
        inventory = {
            warehouse: dict(sku_inventory)
            for warehouse, sku_inventory in state.inventory.items()
        }
        vehicles_next = {
            vehicle_id: _advance_vehicle(vehicle)
            for vehicle_id, vehicle in state.vehicles.items()
        }
        pending_by_id = {order.order_id: order for order in state.pending_orders}
        assigned_orders: set[int] = set()
        assigned_vehicles: set[str] = set()
        completed_step: list[OrderOutcome] = []
        invalid_assignments = 0
        dispatch_cost = 0.0
        late_cost = 0.0
        on_time_count = 0
        priority_weight = 0.0
        on_time_priority_weight = 0.0

        for assignment in assignments:
            if assignment.order_id in assigned_orders or assignment.vehicle_id in assigned_vehicles:
                invalid_assignments += 1
                continue

            order = pending_by_id.get(assignment.order_id)
            vehicle = state.vehicles.get(assignment.vehicle_id)
            if order is None or vehicle is None:
                invalid_assignments += 1
                continue
            if assignment.warehouse not in inventory or assignment.warehouse not in WAREHOUSE_INDEX:
                invalid_assignments += 1
                continue
            if vehicle.location != assignment.warehouse or vehicle.status != "available":
                invalid_assignments += 1
                continue
            if _is_outaged(assignment.vehicle_id, outages):
                invalid_assignments += 1
                continue
            if vehicle.capacity < order.quantity:
                invalid_assignments += 1
                continue
            if inventory[assignment.warehouse].get(order.sku, 0) < order.quantity:
                invalid_assignments += 1
                continue

            inventory[assignment.warehouse][order.sku] -= order.quantity
            assigned_orders.add(order.order_id)
            assigned_vehicles.add(vehicle.vehicle_id)

            distance = distance_km(assignment.warehouse, order.destination)
            lane_multiplier = traffic[
                WAREHOUSE_INDEX[assignment.warehouse],
                CUSTOMER_INDEX[order.destination],
            ]
            travel_days = max(1, int(ceil(distance * float(lane_multiplier) / 520.0)))
            delivered_day = state.time + travel_days
            late_days = max(0, delivered_day - order.deadline)
            outcome = OrderOutcome(
                order_id=order.order_id,
                warehouse=assignment.warehouse,
                vehicle_id=vehicle.vehicle_id,
                destination=order.destination,
                sku=order.sku,
                quantity=order.quantity,
                priority=order.priority,
                order_day=order.day,
                deadline=order.deadline,
                dispatched_day=state.time,
                delivered_day=delivered_day,
                late_days=late_days,
                distance_km=distance,
            )
            completed_step.append(outcome)

            dispatch_cost += self.km_cost * distance + self.unit_handling_cost * order.quantity
            late_cost += (
                late_days
                * order.quantity
                * order.priority
                * self.late_cost_per_priority_unit_day
            )
            priority_units = order.priority * order.quantity
            priority_weight += priority_units
            if late_days == 0:
                on_time_count += 1
                on_time_priority_weight += priority_units

            route_remaining = max(0, travel_days - 1)
            vehicles_next[vehicle.vehicle_id] = VehicleState(
                vehicle_id=vehicle.vehicle_id,
                location=assignment.warehouse,
                capacity=vehicle.capacity,
                status="en_route" if route_remaining > 0 else "available",
                remaining_time=route_remaining,
            )

        pending_after = [
            order for order in state.pending_orders if order.order_id not in assigned_orders
        ]
        pending_after.extend(new_orders)

        backlog_cost = sum(
            max(0, state.time + 1 - order.deadline)
            * order.priority
            * self.backlog_cost_per_priority_day
            + order.priority * self.waiting_cost_per_priority_day
            for order in pending_after
        )
        invalid_assignment_cost = invalid_assignments * self.invalid_assignment_cost
        total_cost = dispatch_cost + late_cost + backlog_cost + invalid_assignment_cost

        dispatched_count = len(completed_step)
        info = {
            "dispatch_cost": dispatch_cost,
            "late_cost": late_cost,
            "pending_backlog": float(len(pending_after)),
            "dispatched_order_count": float(dispatched_count),
            "on_time_rate": (
                1.0 if dispatched_count == 0 else on_time_count / dispatched_count
            ),
            "priority_weighted_on_time_rate": (
                1.0 if priority_weight == 0 else on_time_priority_weight / priority_weight
            ),
            "vehicle_utilization": dispatched_count / len(VEHICLES),
            "invalid_assignment_count": float(invalid_assignments),
            "new_order_count": float(len(new_orders)),
        }
        next_state = LogisticsState(
            inventory=inventory,
            vehicles=vehicles_next,
            pending_orders=tuple(pending_after),
            completed_orders=state.completed_orders + tuple(completed_step),
            time=state.time + 1,
            day_of_week=(state.day_of_week + 1) % 7,
        )
        return next_state, info, total_cost


def _advance_vehicle(vehicle: VehicleState) -> VehicleState:
    if vehicle.status != "en_route":
        return replace(vehicle, status="available", remaining_time=0)

    remaining_time = max(0, vehicle.remaining_time - 1)
    return replace(
        vehicle,
        status="available" if remaining_time == 0 else "en_route",
        remaining_time=remaining_time,
    )


def _is_outaged(vehicle_id: str, outages: np.ndarray) -> bool:
    return bool(outages[VEHICLE_INDEX[vehicle_id]])
