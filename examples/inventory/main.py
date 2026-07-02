from __future__ import annotations

from examples.inventory.data import InventoryDataModule
from examples.inventory.models import InventoryModel
from examples.inventory.policies import OrderUpToPolicy
from sda import Policy, evaluate

DEFAULT_HORIZON = 12
DEFAULT_N_SCENARIOS = 1000
DEFAULT_BATCH_SIZE = 128
DEFAULT_INITIAL_INVENTORY = 50.0
DEFAULT_DEMAND_LAMBDA = 20.0
DEFAULT_SEED = 42

DEFAULT_ORDER_COST = 1.0
DEFAULT_HOLDING_COST = 0.1
DEFAULT_STOCKOUT_COST = 8.0


def build_result(
    policy: Policy | None = None,
    *,
    horizon: int = DEFAULT_HORIZON,
    n_scenarios: int = DEFAULT_N_SCENARIOS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    initial_inventory: float = DEFAULT_INITIAL_INVENTORY,
    demand_lambda: float = DEFAULT_DEMAND_LAMBDA,
    order_cost: float = DEFAULT_ORDER_COST,
    holding_cost: float = DEFAULT_HOLDING_COST,
    stockout_cost: float = DEFAULT_STOCKOUT_COST,
    seed: int | None = DEFAULT_SEED,
):
    """Evaluate one policy against Poisson demand futures.

    With no arguments this reproduces the walkthrough's default order-up-to run.
    Pass ``policy`` (and optionally the sizing/cost/seed knobs) to compare other
    rules against the same seeded scenarios.
    """
    data = InventoryDataModule(
        horizon=horizon,
        n_scenarios=n_scenarios,
        batch_size=batch_size,
        initial_inventory=initial_inventory,
        demand_lambda=demand_lambda,
        seed=seed,
    )
    if policy is None:
        policy = OrderUpToPolicy(reorder_point=30, order_up_to=80)
    model = InventoryModel(
        policy=policy,
        order_cost=order_cost,
        holding_cost=holding_cost,
        stockout_cost=stockout_cost,
    )
    return evaluate(model, data)


def main() -> None:
    result = build_result()
    print(f"Total cost mean: {result['total_cost'].mean():.2f}")
    print(f"Total cost p95: {result['total_cost'].percentile(95):.2f}")
    print(f"Total cost CVaR 95: {result['total_cost'].cvar(0.95):.2f}")
    print(f"Inventory t=5 mean: {result['inventory'].at_time(5).mean():.2f}")
    print(f"Fill rate mean: {result['fill_rate'].mean():.3f}")
    print(f"Stockout rate: {result['stockout'].mean():.3f}")


if __name__ == "__main__":
    main()
