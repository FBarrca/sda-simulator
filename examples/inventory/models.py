from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import simpy

from sda import Policy, Recorder, SDAModel, ScenarioSpec


@dataclass
class InventoryState:
    """Mutable state for one inventory scenario."""

    inventory: float


class InventoryModel(SDAModel):
    """Lost-sales single-item inventory model implemented as a SimPy process."""

    def __init__(
        self,
        policy: Policy,
        order_cost: float,
        holding_cost: float,
        stockout_cost: float,
    ) -> None:
        super().__init__(policy)
        self.order_cost = float(order_cost)
        self.holding_cost = float(holding_cost)
        self.stockout_cost = float(stockout_cost)

    def build(
        self,
        env: simpy.Environment,
        scenario: ScenarioSpec,
        recorder: Recorder,
    ) -> InventoryState:
        """Register the daily inventory process for one scenario."""
        state = InventoryState(inventory=float(scenario.initial_state))
        env.process(self._run(env, scenario, recorder, state))
        return state

    def _run(
        self,
        env: simpy.Environment,
        scenario: ScenarioSpec,
        recorder: Recorder,
        state: InventoryState,
    ):
        demand_path = np.asarray(scenario.data["demand"], dtype=float)

        for demand in demand_path:
            order = float(self.policy.act(state, env, recorder.history))
            available = state.inventory + order
            sales = min(available, float(demand))
            lost_sales = max(float(demand) - available, 0.0)
            state.inventory = available - sales
            fill_rate = 1.0 if demand <= 0 else sales / float(demand)
            cost = (
                self.order_cost * order
                + self.holding_cost * state.inventory
                + self.stockout_cost * lost_sales
            )

            recorder.cost(cost)
            recorder.log("inventory", state.inventory)
            recorder.log("stockout", float(lost_sales > 0))
            recorder.log("fill_rate", fill_rate)
            recorder.log("order_quantity", order)
            recorder.log("lost_sales", lost_sales)
            recorder.log("sales", sales)
            yield env.timeout(1.0)
