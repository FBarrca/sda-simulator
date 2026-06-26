from __future__ import annotations

from dataclasses import dataclass

from examples.logistics.network import SKUS, VEHICLE_SPECS, WAREHOUSES


@dataclass(frozen=True)
class Order:
    order_id: int
    day: int
    destination: str
    sku: str
    quantity: int
    priority: int
    deadline: int
    origin: str | None = None


@dataclass(frozen=True)
class Assignment:
    order_id: int
    warehouse: str
    vehicle_id: str


@dataclass(frozen=True)
class VehicleState:
    vehicle_id: str
    location: str
    capacity: int
    status: str = "available"
    remaining_time: int = 0


@dataclass(frozen=True)
class OrderOutcome:
    order_id: int
    warehouse: str
    vehicle_id: str
    destination: str
    sku: str
    quantity: int
    priority: int
    order_day: int
    deadline: int
    dispatched_day: int
    delivered_day: int
    late_days: int
    distance_km: float


@dataclass
class LogisticsState:
    inventory: dict[str, dict[str, int]]
    vehicles: dict[str, VehicleState]
    pending_orders: tuple[Order, ...]
    completed_orders: tuple[OrderOutcome, ...]
    time: int
    day_of_week: int


DEFAULT_INVENTORY = {
    "W_MADRID": {
        "AMBIENT_FOOD": 560,
        "COLD_CHAIN": 190,
        "ELECTRONICS": 145,
        "PHARMA": 90,
    },
    "W_BARCELONA": {
        "AMBIENT_FOOD": 520,
        "COLD_CHAIN": 205,
        "ELECTRONICS": 165,
        "PHARMA": 85,
    },
    "W_VALENCIA": {
        "AMBIENT_FOOD": 500,
        "COLD_CHAIN": 180,
        "ELECTRONICS": 130,
        "PHARMA": 80,
    },
}


def initial_logistics_state() -> LogisticsState:
    return LogisticsState(
        inventory={
            warehouse: {sku: int(DEFAULT_INVENTORY[warehouse][sku]) for sku in SKUS}
            for warehouse in WAREHOUSES
        },
        vehicles={
            vehicle_id: VehicleState(
                vehicle_id=vehicle_id,
                location=home_warehouse,
                capacity=capacity,
            )
            for vehicle_id, (home_warehouse, capacity) in VEHICLE_SPECS.items()
        },
        pending_orders=(),
        completed_orders=(),
        time=0,
        day_of_week=0,
    )


def clone_state(state: LogisticsState) -> LogisticsState:
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
