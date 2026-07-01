## SDA Simulator V2

`sda` is a small SimPy-native package for Sequential Decision Analytics
simulation. Use it when you want to test a decision rule over many possible
futures before trusting it in a real system.

The standard flow is:

```python
policy = MyPolicy(...)
model = MyModel(policy)
data = MyDataModule(...)
result = evaluate(model, data)
```

The package keeps four responsibilities separate:

- `Policy`: chooses actions from the current state, `simpy.Environment`, and
  completed recorder history.
- `SDAModel`: registers SimPy processes in `build(env, scenario, recorder)` and
  optionally logs final diagnostics in `finalize(...)`.
- `DataModule`: creates `ScenarioBatch` objects made of independent
  `ScenarioSpec`s.
- `evaluate`: runs the data lifecycle, each SimPy environment, and returns
  queryable metric logs.

Common imports:

```python
from sda import (
    ArrayDataModule,
    BootstrapDataModule,
    DataModule,
    GeneratorDataModule,
    Policy,
    SDAModel,
    ScenarioSpec,
    evaluate,
)
```

Tiny SimPy model:

```python
import numpy as np

from sda import ArrayDataModule, Policy, SDAModel, evaluate


class OrderUpToPolicy(Policy):
    def __init__(self, reorder_point, order_up_to):
        self.reorder_point = reorder_point
        self.order_up_to = order_up_to

    def act(self, state, env, history):
        del env, history
        if state["inventory"] < self.reorder_point:
            return self.order_up_to - state["inventory"]
        return 0.0


class InventoryModel(SDAModel):
    def build(self, env, scenario, recorder):
        state = {"inventory": float(scenario.initial_state)}
        env.process(self._run(env, scenario, recorder, state))
        return state

    def _run(self, env, scenario, recorder, state):
        for demand in np.asarray(scenario.data["demand"], dtype=float):
            order = self.policy.act(state, env, recorder.history)
            available = state["inventory"] + order
            sales = min(available, float(demand))
            lost_sales = max(float(demand) - available, 0.0)
            state["inventory"] = available - sales
            recorder.cost(order + 0.1 * state["inventory"] + 5.0 * lost_sales)
            recorder.log("inventory", state["inventory"])
            recorder.log("fill_rate", 1.0 if demand == 0 else sales / float(demand))
            yield env.timeout(1.0)

    def finalize(self, state, scenario, recorder):
        del scenario
        recorder.trajectory("ending_inventory", state["inventory"])


demand = np.array([[18, 20, 21, 17], [25, 30, 14, 16]], dtype=float)
data = ArrayDataModule({"demand": demand}, initial_state=50, batch_size=2)
model = InventoryModel(OrderUpToPolicy(reorder_point=30, order_up_to=80))
result = evaluate(model, data)

print(result["total_cost"].mean())
print(result["inventory"].at_time(2).mean())
print(result.summary())
```

## Examples

Source-tree examples:

```bash
python3 -m examples.inventory
python3 -m examples.logistics
python3 -m examples.multi_echelon_inventory
python3 -m examples.multi_echelon_inventory --mode backorder
```

### Multi-Echelon Inventory Optimization

The multi-echelon example reproduces the reference
`multi-echelon-inventory-optimization` SimPy model in the SDA flow. It models a
six-node supply chain with a source node, downstream stocking nodes, empirical
bootstrap demand, empirical lead-time delays, and a base-stock reorder policy.

![Multi-echelon supply network](examples/multi_echelon_inventory/multi_echelon_network.svg)

The business question is how low we can drive inventory while still protecting
target service levels. The objective matches the reference project:

```text
average on-hand inventory
+ 1.0e6 * sum(max(0, service target - average service level))
```

The published reference vectors reduce inventory while meeting the service
targets, so the penalty term stays at zero:

| Mode | Initial objective | Published objective | Change |
| --- | ---: | ---: | ---: |
| Lost sales | `2783.462` | `2445.776` | `-12.1%` |
| Backorder | `2767.635` | `2515.907` | `-9.1%` |

![Multi-echelon objective scorecard](examples/multi_echelon_inventory/multi_echelon_objective.svg)

The example also exposes the daily inventory dynamics when diagnostics are
enabled. The sawtooth trace shows how the base-stock policy replenishes after
inventory position crosses a reorder point.

![Multi-echelon daily inventory trace](examples/multi_echelon_inventory/multi_echelon_inventory_trace.svg)

It supports the normal programmatic flow:

```python
from examples.multi_echelon_inventory import build_data, build_model
from sda import evaluate

data = build_data(n_scenarios=20, batch_size=1)
model = build_model()
result = evaluate(model, data)
```

Multi-echelon objective runs are fast by default. Use
`build_model(record_daily_metrics=True)` or the `--daily-metrics` CLI flag when
you need dense per-day diagnostic traces. Regenerate the SVGs with:

```bash
python3 -m examples.multi_echelon_inventory.visualize
```

Run tests:

```bash
python3 -m pytest
```

Build docs:

```bash
python3 -m sphinx -b html docs docs/_build/html
```
