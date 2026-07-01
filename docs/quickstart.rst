Quickstart
==========

This page walks through a complete first simulation. The example is a small
inventory problem:

* each scenario is one possible demand future,
* the policy decides how many units to order,
* the model updates inventory after demand arrives,
* the result stores cost, inventory, lost-sales, and fill-rate logs.

Import the public API from ``sda``:

.. code-block:: python

   from sda import ArrayDataModule, Policy, SDAModel, evaluate, step_metric

The Complete Program
--------------------

.. code-block:: python

   import numpy as np

   from sda import ArrayDataModule, Policy, SDAModel, evaluate, step_metric


   class OrderUpToPolicy(Policy):
       def __init__(self, reorder_point, order_up_to):
           self.reorder_point = reorder_point
           self.order_up_to = order_up_to

       def act(self, state, t, history):
           inventory = np.asarray(state, dtype=float)
           return np.where(
               inventory < self.reorder_point,
               self.order_up_to - inventory,
               0.0,
           )


   class InventoryModel(SDAModel):
       def transition(self, state, decision, exogenous, t):
           demand = np.asarray(exogenous["demand"], dtype=float)
           available = np.asarray(state, dtype=float) + np.asarray(decision, dtype=float)
           return np.maximum(available - demand, 0.0)

       def cost(self, state, decision, exogenous, next_state, t):
           demand = np.asarray(exogenous["demand"], dtype=float)
           available = np.asarray(state, dtype=float) + np.asarray(decision, dtype=float)
           lost_sales = np.maximum(demand - available, 0.0)
           return (
               1.0 * np.asarray(decision, dtype=float)
               + 0.1 * np.asarray(next_state, dtype=float)
               + 5.0 * lost_sales
           )

       def info(self, state, decision, exogenous, next_state, cost, t):
           demand = np.asarray(exogenous["demand"], dtype=float)
           available = np.asarray(state, dtype=float) + np.asarray(decision, dtype=float)
           sold = np.minimum(available, demand)
           lost_sales = np.maximum(demand - available, 0.0)
           fill_rate = np.divide(
               sold,
               demand,
               out=np.ones_like(sold, dtype=float),
               where=demand > 0,
           )
           return {"lost_sales": lost_sales, "fill_rate": fill_rate}


   demand_paths = np.array(
       [
           [18, 20, 21, 17],
           [25, 30, 14, 16],
           [10, 12, 35, 20],
       ],
       dtype=float,
   )

   data = ArrayDataModule(
       {"demand": demand_paths},
       initial_state=np.full(demand_paths.shape[0], 50.0),
       batch_size=2,
   )

   policy = OrderUpToPolicy(reorder_point=30, order_up_to=80)
   model = InventoryModel(policy)

   result = evaluate(
       model,
       data,
       extra_metrics=[
           step_metric("ending_inventory", lambda step: step.next_state),
           step_metric("lost_sales", lambda step: step.info["lost_sales"]),
           step_metric("fill_rate", lambda step: step.info["fill_rate"]),
       ],
   )

   print(result.names())
   print(result["total_cost"].mean())
   print(result["total_cost"].percentile(95))
   print(result["total_cost"].cvar(0.95))
   print(result["ending_inventory"].at_time(2).mean())
   print(result["fill_rate"].step_level().mean())

What Each Piece Means
---------------------

``OrderUpToPolicy``
   The decision rule. It sees the current inventory level and decides whether
   to order. It does not see today's demand before deciding.

``InventoryModel``
   The domain rules. It defines how inventory changes after demand is revealed,
   how cost is computed, and which extra diagnostics are useful for metrics.

``ArrayDataModule``
   The scenario source. Here the demand futures are already known arrays with
   shape ``[n_scenarios, horizon]``. The simulator slices them into batches.

``evaluate(model, data)``
   The standard entrypoint. It runs the data lifecycle, rolls every scenario
   forward, logs default costs, and returns a ``SimulationResult``.

``result``
   The queryable metric log. ``result["total_cost"]`` is the distribution of
   total cost over scenarios. ``result["fill_rate"]`` is the collection of
   fill-rate observations over scenario-time pairs.

What Happens During Rollout
---------------------------

For each batch and each time step, ``sda`` follows this order:

.. code-block:: text

   1. start from current state
   2. call policy.act(state, t, history)
   3. reveal exogenous values for this time step
   4. call model.transition(...)
   5. call model.cost(...)
   6. call model.info(...)
   7. send records to metrics

The policy decides before current demand is revealed. That timing is central
to SDA. If a value is known before the decision, put it in ``state``. If it is
uncertain until after the decision, put it in the data module's ``exogenous``
paths.

Reading The Logs
----------------

``evaluate`` records two cost metrics by default:

``step_cost``
   One observation per scenario per time step.

``total_cost``
   One observation per scenario after the full trajectory.

The quickstart adds three domain metrics with ``step_metric``. Query them after
the run:

.. code-block:: python

   result.names()
   result["total_cost"].values()
   result["total_cost"].mean()
   result["total_cost"].percentile(95)
   result["total_cost"].cvar(0.95)
   result["ending_inventory"].at_time(2).mean()
   result.records("lost_sales")

Use ``mean`` for average performance, percentiles for distribution shape, and
``cvar(0.95)`` for the average value in the worst 5% of observations. For cost
metrics, lower is usually better.

Choosing Scenario Data
----------------------

All scenario sources are ``DataModule`` objects:

``ArrayDataModule``
   Use when full sample paths are already in memory. This is best for tests,
   examples, cached forecasts, deterministic scenarios, and fair comparisons
   where several policies should face the same futures.

``GeneratorDataModule``
   Use when futures should be generated lazily from a distribution, forecast
   model, service call, or domain simulator.

``BootstrapDataModule``
   Use when futures should be resampled from historical observations.

Custom ``DataModule``
   Use when scenario construction has setup, fitted state, stages, or
   source-specific batching logic.

Next Steps
----------

Read :doc:`workflow` for the standard project shape, :doc:`concepts` for the
plain-language SDA vocabulary, :doc:`use_cases` for common applications, and
:doc:`metrics` when you are ready to add richer metric logs or optional MLflow
tracking.
