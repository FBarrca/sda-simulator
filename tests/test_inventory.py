from examples.inventory import (
    INVENTORY_METRICS,
    InventoryDataModule,
    InventoryModel,
    OrderUpToPolicy,
)
from sda import evaluate


def test_inventory_example_metrics_are_well_formed():
    data = InventoryDataModule(
        horizon=12,
        n_scenarios=100,
        batch_size=16,
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
    result = evaluate(model, data)

    assert len(result.metric("total_cost").values()) == 100
    assert result.metric("total_cost").percentile(95) > 0
    assert result.metric("inventory").min() >= 0
    assert 0 <= result.metric("stockout").mean() <= 1
    assert 0 <= result.metric("fill_rate").mean() <= 1
    assert result.metric("inventory").at_time(5).mean() >= 0
    assert set(INVENTORY_METRICS).issubset(result.names())
    assert "total_cost" in result.summary()
