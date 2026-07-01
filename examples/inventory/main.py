from __future__ import annotations

from examples.inventory.data import InventoryDataModule
from examples.inventory.models import InventoryModel
from examples.inventory.policies import OrderUpToPolicy
from sda import evaluate


def build_result():
    data = InventoryDataModule(
        horizon=12,
        n_scenarios=1000,
        batch_size=128,
        initial_inventory=50,
        demand_lambda=20,
        seed=42,
    )
    policy = OrderUpToPolicy(reorder_point=30, order_up_to=80)
    model = InventoryModel(
        policy=policy,
        order_cost=1.0,
        holding_cost=0.1,
        stockout_cost=8.0,
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
