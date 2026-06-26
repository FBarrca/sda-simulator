from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace

import numpy as np

from examples.logistics.domain import Order, initial_logistics_state
from examples.logistics.network import (
    CUSTOMER_INDEX,
    CUSTOMERS,
    SKUS,
    VEHICLE_INDEX,
    VEHICLES,
    WAREHOUSE_INDEX,
    WAREHOUSES,
)
from sda.data import ScenarioBatch, ScenarioLoader

DEMAND_BY_DAY = np.asarray([1.0, 1.05, 1.2, 1.1, 1.0, 0.55, 0.4], dtype=float)
MONTHLY_SEASONALITY = np.asarray(
    [0.90, 0.94, 0.98, 1.02, 1.06, 1.00, 0.96, 0.98, 1.05, 1.14, 1.26, 1.35],
    dtype=float,
)

SKU_PROFILES = {
    "AMBIENT_FOOD": {"weight": 0.48, "mean_quantity": 10.0, "priorities": (1, 1, 2)},
    "COLD_CHAIN": {"weight": 0.22, "mean_quantity": 6.0, "priorities": (2, 2, 3)},
    "ELECTRONICS": {"weight": 0.20, "mean_quantity": 4.0, "priorities": (2, 3, 3)},
    "PHARMA": {"weight": 0.10, "mean_quantity": 3.0, "priorities": (3, 3, 3)},
}

REGIONAL_WEIGHTS = np.asarray(
    [0.13, 0.12, 0.10, 0.06, 0.08, 0.09, 0.08, 0.07, 0.07, 0.06, 0.05, 0.09],
    dtype=float,
)
REGIONAL_WEIGHTS = REGIONAL_WEIGHTS / REGIONAL_WEIGHTS.sum()


@dataclass(frozen=True)
class SyntheticHistory:
    orders: np.ndarray
    traffic_multiplier: np.ndarray
    vehicle_outages: np.ndarray
    event_labels: np.ndarray

    @property
    def days(self) -> int:
        return int(self.orders.shape[0])


def synthetic_history(days: int, seed: int | None = None) -> SyntheticHistory:
    if days <= 0:
        raise ValueError("days must be positive")

    rng = np.random.default_rng(seed)
    orders = np.empty(days, dtype=object)
    traffic = np.empty((days, len(WAREHOUSES), len(CUSTOMERS)), dtype=float)
    outages = np.empty((days, len(VEHICLES)), dtype=bool)
    event_labels = np.empty(days, dtype=object)

    for day in range(days):
        day_of_week = day % 7
        month_index = min(11, int(day * 12 / max(days, 1)))
        labels: list[str] = []
        demand_multiplier = DEMAND_BY_DAY[day_of_week] * MONTHLY_SEASONALITY[month_index]
        outage_probability = 0.025

        holiday_peak = month_index in {10, 11}
        promotion = rng.random() < 0.045
        severe_weather = rng.random() < 0.025
        port_congestion = month_index in {9, 10, 11} and rng.random() < 0.06

        if holiday_peak:
            labels.append("holiday_peak")
            demand_multiplier *= 1.16
        if promotion:
            labels.append("promotion")
            demand_multiplier *= 1.35
        if severe_weather:
            labels.append("severe_weather")
            demand_multiplier *= 1.08
            outage_probability += 0.08
        if port_congestion:
            labels.append("port_congestion")
            demand_multiplier *= 1.06
            outage_probability += 0.025

        order_count = int(rng.poisson(9.5 * demand_multiplier))
        sku_weights = _sku_weights(promotion=promotion, holiday_peak=holiday_peak)
        day_orders = [
            _sample_order(
                rng=rng,
                order_id=day * 1000 + index,
                day=day,
                sku_weights=sku_weights,
            )
            for index in range(order_count)
        ]

        day_traffic = rng.normal(loc=1.0, scale=0.08, size=(len(WAREHOUSES), len(CUSTOMERS)))
        day_traffic = np.clip(day_traffic, 0.78, 1.35)
        if day_of_week >= 5:
            day_traffic *= 0.94
        if severe_weather:
            day_traffic *= 1.45
        if port_congestion:
            barcelona = WAREHOUSE_INDEX["W_BARCELONA"]
            port = CUSTOMER_INDEX["C_BARCELONA_PORT"]
            day_traffic[barcelona, :] *= 1.12
            day_traffic[:, port] *= 1.35

        day_outages = rng.random(len(VEHICLES)) < outage_probability
        if port_congestion:
            for vehicle_id in ("V_BAR_1", "V_BAR_2"):
                day_outages[VEHICLE_INDEX[vehicle_id]] |= rng.random() < 0.08

        orders[day] = tuple(day_orders)
        traffic[day] = day_traffic
        outages[day] = day_outages
        event_labels[day] = tuple(labels)

    return SyntheticHistory(
        orders=orders,
        traffic_multiplier=traffic,
        vehicle_outages=outages,
        event_labels=event_labels,
    )


class LogisticsScenarioLoader(ScenarioLoader):
    """Bootstrap logistics futures from synthetic history using 7-day blocks."""

    def __init__(
        self,
        horizon: int,
        n_scenarios: int,
        batch_size: int,
        history_days: int = 365,
        seed: int | None = None,
        block_size: int = 7,
    ) -> None:
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if n_scenarios <= 0:
            raise ValueError("n_scenarios must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if history_days < block_size:
            raise ValueError("history_days must be at least block_size")

        self.horizon = int(horizon)
        self.n_scenarios = int(n_scenarios)
        self.batch_size = int(batch_size)
        self.history_days = int(history_days)
        self.block_size = int(block_size)
        self.seed = seed
        self.history = synthetic_history(history_days, seed=seed)

    def __iter__(self) -> Iterator[ScenarioBatch]:
        rng = np.random.default_rng(self.seed)
        for start in range(0, self.n_scenarios, self.batch_size):
            stop = min(start + self.batch_size, self.n_scenarios)
            size = stop - start
            orders = np.empty((size, self.horizon), dtype=object)
            traffic = np.empty(
                (size, self.horizon, len(WAREHOUSES), len(CUSTOMERS)),
                dtype=float,
            )
            outages = np.empty((size, self.horizon, len(VEHICLES)), dtype=bool)
            event_labels = np.empty((size, self.horizon), dtype=object)
            history_day_index = np.empty((size, self.horizon), dtype=int)

            for batch_index, scenario_id in enumerate(range(start, stop)):
                sampled_days = self._sample_days(rng)
                for t, source_day in enumerate(sampled_days):
                    raw_orders = self.history.orders[source_day]
                    orders[batch_index, t] = tuple(
                        _shift_order(order, scenario_id=scenario_id, t=t, index=index)
                        for index, order in enumerate(raw_orders)
                    )
                    traffic[batch_index, t] = self.history.traffic_multiplier[source_day]
                    outages[batch_index, t] = self.history.vehicle_outages[source_day]
                    event_labels[batch_index, t] = self.history.event_labels[source_day]
                    history_day_index[batch_index, t] = source_day

            yield ScenarioBatch(
                initial_state=[initial_logistics_state() for _ in range(size)],
                exogenous={
                    "orders": orders,
                    "traffic_multiplier": traffic,
                    "vehicle_outages": outages,
                    "event_labels": event_labels,
                    "history_day_index": history_day_index,
                },
                scenario_ids=list(range(start, stop)),
            )

    def _sample_days(self, rng: np.random.Generator) -> list[int]:
        sampled: list[int] = []
        max_start = self.history_days - self.block_size
        while len(sampled) < self.horizon:
            block_start = int(rng.integers(0, max_start + 1))
            sampled.extend(range(block_start, block_start + self.block_size))
        return sampled[: self.horizon]


def _sample_order(
    *,
    rng: np.random.Generator,
    order_id: int,
    day: int,
    sku_weights: np.ndarray,
) -> Order:
    sku = str(rng.choice(SKUS, p=sku_weights))
    profile = SKU_PROFILES[sku]
    quantity = max(1, int(rng.poisson(float(profile["mean_quantity"]))))
    priority = int(rng.choice(profile["priorities"]))
    destination = str(rng.choice(CUSTOMERS, p=REGIONAL_WEIGHTS))

    if priority == 3:
        lead_time = int(rng.integers(1, 3))
    elif priority == 2:
        lead_time = int(rng.integers(2, 4))
    else:
        lead_time = int(rng.integers(3, 6))

    return Order(
        order_id=order_id,
        day=day,
        destination=destination,
        sku=sku,
        quantity=quantity,
        priority=priority,
        deadline=day + lead_time,
    )


def _sku_weights(*, promotion: bool, holiday_peak: bool) -> np.ndarray:
    weights = np.asarray([float(SKU_PROFILES[sku]["weight"]) for sku in SKUS], dtype=float)
    if promotion:
        weights[SKUS.index("ELECTRONICS")] *= 1.85
    if holiday_peak:
        weights[SKUS.index("AMBIENT_FOOD")] *= 1.18
        weights[SKUS.index("ELECTRONICS")] *= 1.45
    return weights / weights.sum()


def _shift_order(order: Order, *, scenario_id: int, t: int, index: int) -> Order:
    lead_time = max(1, order.deadline - order.day)
    return replace(
        order,
        order_id=scenario_id * 1_000_000 + t * 1000 + index,
        day=t,
        deadline=t + lead_time,
        origin=None,
    )
