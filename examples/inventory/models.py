from __future__ import annotations

import numpy as np

from sda.model import Policy, SDAModel


class InventoryModel(SDAModel):
    """Lost-sales single-item inventory model."""

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

    def transition(self, state, decision, exogenous, t: int):
        inventory = np.asarray(state, dtype=float)
        order = np.asarray(decision, dtype=float)
        demand = np.asarray(exogenous["demand"], dtype=float)
        available = inventory + order
        sales = np.minimum(available, demand)
        return available - sales

    def cost(self, state, decision, exogenous, next_state, t: int):
        order = np.asarray(decision, dtype=float)
        demand = np.asarray(exogenous["demand"], dtype=float)
        inventory = np.asarray(state, dtype=float)
        available = inventory + order
        lost_sales = np.maximum(demand - available, 0.0)
        ending_inventory = np.asarray(next_state, dtype=float)
        return (
            self.order_cost * order
            + self.holding_cost * ending_inventory
            + self.stockout_cost * lost_sales
        )

    def info(self, state, decision, exogenous, next_state, cost, t: int):
        order = np.asarray(decision, dtype=float)
        demand = np.asarray(exogenous["demand"], dtype=float)
        inventory = np.asarray(state, dtype=float)
        available = inventory + order
        sales = np.minimum(available, demand)
        lost_sales = np.maximum(demand - available, 0.0)
        fill_rate = np.divide(
            sales,
            demand,
            out=np.ones_like(demand, dtype=float),
            where=demand > 0,
        )
        return {
            "available_inventory": available,
            "demand": demand,
            "fill_rate": fill_rate,
            "lost_sales": lost_sales,
            "order_quantity": order,
            "sales": sales,
        }
