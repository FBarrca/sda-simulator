from __future__ import annotations

from math import sqrt
from statistics import NormalDist

from sda import EventRecord, Policy


class ChaseDemandPolicy(Policy):
    """Orders exactly the previous day's realized demand.

    A naive one-for-one replacement rule. It reconstructs yesterday's demand
    from the per-scenario history as ``sales + lost_sales`` (the model logs
    those but not raw demand). With nothing yet logged on the first day it
    orders nothing and coasts on the initial inventory.
    """

    def act(self, state, env, history: list[EventRecord]):
        """Return the most recent day's realized demand."""
        del state, env
        last_sales = _last_value(history, "sales")
        last_lost = _last_value(history, "lost_sales")
        if last_sales is None:
            return 0.0
        return max(last_sales + (last_lost or 0.0), 0.0)


class FixedOrderQuantityPolicy(Policy):
    """Orders a fixed quantity ``Q`` whenever inventory falls below ``s``.

    The classic ``(s, Q)`` rule: unlike order-up-to, it always buys the same
    lot size rather than topping up to a moving target.
    """

    def __init__(self, reorder_point: float, order_quantity: float) -> None:
        if order_quantity <= 0:
            raise ValueError("order_quantity must be positive")
        self.reorder_point = float(reorder_point)
        self.order_quantity = float(order_quantity)

    def act(self, state, env, history: list[EventRecord]):
        """Return the fixed lot size while inventory is below the reorder point."""
        del env, history
        if state.inventory < self.reorder_point:
            return self.order_quantity
        return 0.0


class OrderUpToPolicy(Policy):
    """Orders up to a target level once inventory falls below a reorder point."""

    def __init__(self, reorder_point: float, order_up_to: float) -> None:
        if order_up_to < reorder_point:
            raise ValueError("order_up_to must be greater than or equal to reorder_point")
        self.reorder_point = float(reorder_point)
        self.order_up_to = float(order_up_to)

    def act(self, state, env, history: list[EventRecord]):
        """Return the replenishment quantity for the current inventory."""
        del env, history
        if state.inventory < self.reorder_point:
            return max(self.order_up_to - state.inventory, 0.0)
        return 0.0


class BaseStockPolicy(Policy):
    """Tops inventory back up to a base-stock level every single day.

    Equivalent to an order-up-to rule with the reorder point set at the target,
    so it reorders every period. This gives the highest service at the cost of
    the most holding -- the top of the policy ladder.
    """

    def __init__(self, base_stock: float) -> None:
        if base_stock <= 0:
            raise ValueError("base_stock must be positive")
        self.base_stock = float(base_stock)

    def act(self, state, env, history: list[EventRecord]):
        """Return the quantity needed to reach the base-stock level."""
        del env, history
        return max(self.base_stock - state.inventory, 0.0)


class DemandScaledOrderUpToPolicy(Policy):
    """The *fair* (s, S) baseline: levels sized from demand and the objective.

    Rather than hand-picked numbers, the reorder point and order-up-to level are
    derived from the demand distribution and the newsvendor **critical ratio**
    implied by the cost weights, ``Cu / (Cu + Co)`` (stockout over stockout plus
    holding). That calibrates the buffer to the very objective the optimiser
    targets -- the single-item analog of the reference's demand-scaled policy.
    """

    def __init__(
        self,
        demand_mean: float,
        demand_std: float,
        stockout_cost: float,
        holding_cost: float,
        lead_time: float = 0.0,
        review_period: float = 1.0,
    ) -> None:
        if stockout_cost + holding_cost <= 0:
            raise ValueError("stockout_cost + holding_cost must be positive")
        critical_ratio = stockout_cost / (stockout_cost + holding_cost)
        z = NormalDist().inv_cdf(critical_ratio)
        protection = float(lead_time) + float(review_period)
        mu = float(demand_mean) * protection
        sigma = float(demand_std) * sqrt(protection)
        self.critical_ratio = critical_ratio
        self.safety_factor = z
        self.order_up_to = mu + z * sigma
        # With zero lead time the reorder point collapses to the target, so the
        # rule tops up every review period (base-stock behaviour).
        if lead_time > 0:
            self.reorder_point = float(demand_mean) * lead_time + z * float(demand_std) * sqrt(lead_time)
        else:
            self.reorder_point = self.order_up_to

    def act(self, state, env, history: list[EventRecord]):
        """Return the replenishment quantity for the current inventory."""
        del env, history
        if state.inventory < self.reorder_point:
            return max(self.order_up_to - state.inventory, 0.0)
        return 0.0


class OptimizedBaseStockPolicy(Policy):
    """Base-stock level chosen by numerically minimising expected daily cost.

    The ladder's optimiser rung -- the single-item analog of the multi-echelon
    example's MILP. It uses ``scipy.optimize`` over ``scipy.stats.poisson`` to
    minimise expected per-period holding-plus-stockout cost over the base-stock
    level ``S``. If SciPy is unavailable it falls back to the analytic
    newsvendor optimum (the critical-ratio quantile), so it always returns a
    sensible level -- mirroring how the reference MILP falls back when SciPy is
    missing.
    """

    def __init__(
        self,
        demand_mean: float,
        holding_cost: float,
        stockout_cost: float,
    ) -> None:
        self.base_stock = _optimize_base_stock(
            demand_mean=float(demand_mean),
            holding_cost=float(holding_cost),
            stockout_cost=float(stockout_cost),
        )

    def act(self, state, env, history: list[EventRecord]):
        """Return the quantity needed to reach the optimised base-stock level."""
        del env, history
        return max(self.base_stock - state.inventory, 0.0)


def _optimize_base_stock(*, demand_mean: float, holding_cost: float, stockout_cost: float) -> float:
    """Minimise expected daily holding + stockout cost over the base-stock level.

    Order cost is proportional to expected demand and independent of the level in
    steady state, so it does not shift the minimiser and is omitted here.
    """
    critical_ratio = stockout_cost / (stockout_cost + holding_cost)
    try:
        from scipy.optimize import minimize_scalar
        from scipy.stats import poisson
    except ImportError:
        # Analytic newsvendor optimum: the critical-ratio demand quantile.
        return float(NormalDist(demand_mean, sqrt(demand_mean)).inv_cdf(critical_ratio))

    support = range(0, int(demand_mean * 5) + 10)
    pmf = poisson.pmf(list(support), demand_mean)

    def expected_cost(level: float) -> float:
        shortfall = sum(max(d - level, 0.0) * p for d, p in zip(support, pmf))
        excess = sum(max(level - d, 0.0) * p for d, p in zip(support, pmf))
        return holding_cost * excess + stockout_cost * shortfall

    result = minimize_scalar(expected_cost, bounds=(0.0, demand_mean * 4.0), method="bounded")
    return float(result.x)


def _last_value(history: list[EventRecord], name: str) -> float | None:
    """Return the value of the most recently logged record with ``name``."""
    for record in reversed(history):
        if record.name == name:
            return float(record.value)
    return None
