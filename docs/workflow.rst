Workflow
========

``sda`` is built around one public path:

.. code-block:: python

   policy = MyPolicy(...)
   model = MyModel(policy)
   data = MyDataModule(...)
   result = evaluate(model, data)

The point of the package is to keep that path obvious. Put decision logic in a
``Policy``, domain dynamics in an ``SDAModel``, scenario construction in a
``DataModule``, and outcome reading in metrics.

The Five-Step Pattern
---------------------

1. Define a ``Policy``.
2. Define an ``SDAModel``.
3. Create a ``DataModule``.
4. Call ``evaluate(model, data)``.
5. Query metric distributions.

A typical source-tree application is split like this:

.. code-block:: text

   my_app/
       data.py       # DataModule and scenario setup
       policies.py   # Policy implementations
       models.py     # SDAModel implementation
       metrics.py    # Optional domain metrics
       main.py       # Wires policy, model, data, evaluate

This structure is not required, but it keeps responsibilities visible.

1. Policy: Choose Actions
-------------------------

The policy receives the current state, the zero-based time index, and completed
history. It returns a decision object that the model understands.

.. code-block:: python

   import numpy as np

   from sda import Policy


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

Do not load scenarios, compute transitions, or log metrics inside the policy.
Its job is only to decide.

2. Model: Define Dynamics
-------------------------

The model turns a decision and one period of exogenous information into the
next state and cost.

.. code-block:: python

   import numpy as np

   from sda import SDAModel


   class InventoryModel(SDAModel):
       def transition(self, state, decision, exogenous, t):
           demand = np.asarray(exogenous["demand"], dtype=float)
           available = np.asarray(state, dtype=float) + np.asarray(decision, dtype=float)
           return np.maximum(available - demand, 0.0)

       def cost(self, state, decision, exogenous, next_state, t):
           demand = np.asarray(exogenous["demand"], dtype=float)
           available = np.asarray(state, dtype=float) + np.asarray(decision, dtype=float)
           lost_sales = np.maximum(demand - available, 0.0)
           return decision + 0.1 * next_state + 5.0 * lost_sales

       def info(self, state, decision, exogenous, next_state, cost, t):
           demand = np.asarray(exogenous["demand"], dtype=float)
           available = np.asarray(state, dtype=float) + np.asarray(decision, dtype=float)
           lost_sales = np.maximum(demand - available, 0.0)
           return {"lost_sales": lost_sales}

Use ``info`` for values that metrics need but that are already natural byproducts
of the domain calculation.

3. Data: Supply Futures
-----------------------

The data step answers one question: what full sample paths of exogenous
information should the model use to test the policy?

For already-built futures, use ``ArrayDataModule``:

.. code-block:: python

   import numpy as np

   from sda import ArrayDataModule


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

For generated futures, use ``GeneratorDataModule``:

.. code-block:: python

   from sda import GeneratorDataModule


   def poisson_demand(*, rng, shape):
       return {"demand": rng.poisson(20, size=shape)}


   data = GeneratorDataModule(
       poisson_demand,
       initial_state=50,
       horizon=12,
       n_scenarios=1000,
       batch_size=128,
       seed=42,
   )

Use ``BootstrapDataModule`` for futures resampled from historical observations.
Use a custom ``DataModule`` when setup has state, stages, source-specific
batching, or several coordinated data sources.

The policy's information set is ``state``, ``t``, and completed ``history``.
It does not receive the exogenous sample paths. If a signal is known before the
decision, make it part of the state. If it is uncertainty realized after the
decision, make it part of ``exogenous``.

4. Evaluation: Run The Rollout
------------------------------

.. code-block:: python

   from sda import evaluate


   policy = OrderUpToPolicy(reorder_point=30, order_up_to=80)
   model = InventoryModel(policy)
   result = evaluate(model, data)

Add domain metrics without losing default cost metrics:

.. code-block:: python

   from sda import info_metric, step_metric


   result = evaluate(
       model,
       data,
       extra_metrics=[
           step_metric("ending_inventory", lambda step: step.next_state),
           info_metric("lost_sales"),
       ],
   )

Pass ``metrics=[...]`` only when you want to replace the default ``step_cost``
and ``total_cost`` metrics completely.

5. Results: Read The Logs
-------------------------

Metrics are stored as observations, so one simulation run can answer several
questions:

.. code-block:: python

   result.names()
   result["total_cost"].mean()
   result["total_cost"].percentile(95)
   result["total_cost"].cvar(0.95)
   result["ending_inventory"].at_time(5).mean()
   result.records("total_cost")
   result.summary()

For cost metrics, lower is usually better. Use percentiles and CVaR when the
worst scenarios matter more than the average.

Reusable Simulator
------------------

Use ``Simulator`` directly when you want a reusable configured runner. It owns
the data lifecycle, rollout order, metric dispatch, history handling, and
optional tracking. It does not own policy choices, scenario generation, model
dynamics, or metric definitions.

.. code-block:: python

   from sda import Simulator, TotalCostMetric


   simulator = Simulator(metrics=[TotalCostMetric()], keep_history=False)
   result = simulator.evaluate(model, data)

Lightning-Inspired Map
----------------------

``sda`` borrows the idea of a small configured runner from PyTorch Lightning,
but keeps simulation vocabulary:

===============================  ======================================
PyTorch Lightning                ``sda``
===============================  ======================================
``Trainer``                      ``Simulator``
``LightningModule``              ``SDAModel`` plus ``Policy``
``LightningDataModule``          ``DataModule``
``Logger`` / ``MLFlowLogger``    ``MLflowTracker``
``self.log(...)``                ``Metric`` objects and ``SimulationResult``
===============================  ======================================

There is intentionally no ``Trainer`` alias. In sequential decision analytics,
``Simulator`` is the clearer name because this runner evaluates policies over
sampled futures rather than training model weights.

MLflow Tracking
---------------

Install the optional MLflow dependency when you want experiment runs:

.. code-block:: bash

   pip install "sda-simulator-v2[mlflow]"

Then pass an ``MLflowTracker`` to ``evaluate``. The simulator still stores
metrics in ``SimulationResult``; the tracker logs aggregate summaries such as
``total_cost.mean`` and ``total_cost.p95`` to MLflow after the rollout
finishes.

.. code-block:: python

   from sda import MLflowTracker


   tracker = MLflowTracker(
       experiment_name="inventory",
       run_name="order-up-to",
       params={"reorder_point": 30, "order_up_to": 80},
       tags={"stage": "baseline"},
   )

   result = evaluate(model, data, tracking=tracker)
