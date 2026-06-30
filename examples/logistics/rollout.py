from __future__ import annotations

from dataclasses import replace

import numpy as np

from examples.logistics.domain import LogisticsState, Order


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
