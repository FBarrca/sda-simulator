## SDA Simulator V2

`sda` is a small Python package for Sequential Decision Analytics simulation.
Use it when you want to test a decision rule over many possible futures before
you trust it in the real system.

The package is intentionally simple. You provide four pieces:

- `Policy`: chooses the next action from the information available now.
- `SDAModel`: applies the action, updates state, and computes cost.
- `DataModule`: creates scenario batches, meaning sampled futures such as
  demand, prices, failures, traffic, orders, or service times.
- `evaluate`: runs the rollout and returns queryable metric logs.

The standard flow is:

```python
policy = MyPolicy(...)
model = MyModel(policy)
data = MyDataModule(...)
result = evaluate(model, data)
```

This is useful for questions like:

- How should a warehouse reorder inventory under uncertain demand?
- Which dispatch policy keeps freight deliveries on time when orders, traffic,
  and vehicle outages vary?
- How much downside risk does a pricing, staffing, maintenance, or allocation
  policy create?
- Which policy is better on the same sampled futures, not just on an average
  forecast?

The installed library package is `sda`. The `examples/` directory contains
source-tree demonstrations and is not part of the installed library API.

Common imports:

```python
from sda import (
    ArrayDataModule,
    BootstrapDataModule,
    DataModule,
    GeneratorDataModule,
    Policy,
    SDAModel,
    evaluate,
    step_metric,
)
```

Information timing: a `Policy` sees the current state, time, and completed
history. It does not see the current or future exogenous sample path. Put
pre-decision information in the state, and put post-decision uncertainty in the
data module's exogenous paths.

Use `Simulator` directly as the trainer-like configured runner when you want
to reuse metric, history, or tracking settings. The library intentionally uses
simulation vocabulary rather than adding a `Trainer` alias.

Source-tree inventory example:

```python
from examples.inventory import (
    FillRateMetric,
    InventoryDataModule,
    InventoryMetric,
    InventoryModel,
    OrderUpToPolicy,
    StockoutMetric,
)
from sda import evaluate

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

result = evaluate(
    model,
    data,
    extra_metrics=[
        InventoryMetric(),
        StockoutMetric(),
        FillRateMetric(),
    ],
)

print(result.names())
print(result["total_cost"].mean())
print(result["total_cost"].percentile(95))
print(result["total_cost"].cvar(0.95))
print(result["inventory"].at_time(5).mean())
print(result.summary())
```

What `evaluate` does:

1. Calls `data.prepare_data()` and `data.setup(stage)`.
2. Iterates over `data.batches(stage)`.
3. For each time step, asks the policy for a decision before revealing that
   period's uncertainty.
4. Calls the model's `transition`, `cost`, and optional `info` hooks.
5. Logs `step_cost`, `total_cost`, and any extra metrics into a
   `SimulationResult`.

Metric logs are stored as rows, not printed text. Query them after the run:

```python
result["total_cost"].mean()
result["total_cost"].percentile(95)
result["fill_rate"].step_level().mean()
result.records("total_cost")
```

Optional MLflow tracking logs aggregate summaries from the same
`SimulationResult`:

```python
from sda import MLflowTracker

result = evaluate(
    model,
    data,
    tracking=MLflowTracker(
        experiment_name="inventory",
        run_name="order-up-to",
        params={"policy": "order_up_to"},
    ),
)
```

Source-tree logistics dispatch example:

```bash
uv run -m examples.logistics
uv run --with matplotlib -m examples.logistics.policy_comparison --output examples/logistics/logistics_policy_comparison.png
uv run --with matplotlib -m examples.logistics.visualize_network --output examples/logistics/logistics_network.png
uv run --with matplotlib -m examples.logistics.visualize_demand --seed 7 --days 365 --output examples/logistics/logistics_synthetic_demand.png
```

Package architecture:

- `sda`: preferred public import surface for users.
- `sda.core`: canonical contracts and records for scenario batches, policies, models, and rollouts.
- `sda.data`: `DataModule` plus array, generator, and bootstrap data modules.
- `sda.simulation`: simulator loop and result wrapper.
- `sda.metrics`: raw metric store, queryable metric series, and built-in cost metrics.
- `sda.tracking`: optional MLflow result-summary logging.

Docs map:

- `docs/quickstart.rst`: a complete first simulation.
- `docs/architecture.rst`: package, data source, lifecycle, and rollout maps.
- `docs/workflow.rst`: standard way to structure policies, models, data, evaluation, and metrics.
- `docs/use_cases.rst`: where SDA fits and how to map real problems.
- `docs/data.rst`: array, generator, historical, and custom data module details.
- `docs/metrics.rst`: default metrics, custom metrics, and result queries.
- `docs/api.rst`: API reference.

Run tests with:

```bash
python3 -m pytest
```

Build docs with:

```bash
python3 -m sphinx -b html docs docs/_build/html
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
