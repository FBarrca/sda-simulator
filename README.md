## SDA Simulator V2

A minimal class-based framework for Sequential Decision Analytics simulation.

The installed library package is `sda`. The `examples/` directory contains source-tree demonstrations and is not part of the installed library API.

Core imports:

```python
from sda import (
    ArrayScenarioLoader,
    Policy,
    SDAModel,
    Simulator,
    StepCostMetric,
    TotalCostMetric,
)
```

Source-tree inventory example:

```python
from examples.inventory import (
    FillRateMetric,
    InventoryMetric,
    InventoryModel,
    InventoryScenarioLoader,
    OrderUpToPolicy,
    StockoutMetric,
)
from sda import Simulator, StepCostMetric, TotalCostMetric

scenarios = InventoryScenarioLoader(
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

simulator = Simulator(
    metrics=[
        StepCostMetric(),
        TotalCostMetric(),
        InventoryMetric(),
        StockoutMetric(),
        FillRateMetric(),
    ]
)

result = simulator.evaluate(model, scenarios)

print(result.metric("total_cost").mean())
print(result.metric("total_cost").percentile(95))
print(result.metric("total_cost").cvar(0.95))
print(result.metric("inventory").at_time(5).mean())
print(result.summary())
```

Core modules:

- `sda.data`: scenario batches and array-backed loaders.
- `sda.model`: policy/model interfaces and simulation records.
- `sda.simulation`: simulator loop and result wrapper.
- `sda.metrics`: raw metric store, queryable metric series, and built-in cost metrics.

Run tests with:

```bash
uv run pytest
```

Build docs with:

```bash
uv run --group docs sphinx-build -b html docs docs/_build/html
```

Run the inventory example from the source tree with:

```bash
uv run -m examples.inventory
```

You can also run the inventory module explicitly:

```bash
uv run -m examples.inventory.main
```

When using `-m`, use dotted module names like `examples.inventory.main`, not file paths like `examples/inventory/main`.
