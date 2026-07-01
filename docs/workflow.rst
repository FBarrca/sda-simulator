Workflow
========

``sda`` is built around one public path:

.. code-block:: python

   policy = MyPolicy(...)
   model = MyModel(policy)
   data = MyDataModule(...)
   result = evaluate(model, data)

The Five-Step Pattern
---------------------

1. Define a ``Policy``.
2. Define an ``SDAModel`` that registers SimPy processes.
3. Create a ``DataModule`` that yields scenarios.
4. Call ``evaluate(model, data)``.
5. Query metric distributions.

A source-tree application often looks like:

.. code-block:: text

   my_app/
       data.py       # DataModule and scenario setup
       policies.py   # Policy implementations
       models.py     # SDAModel implementation
       metrics.py    # Optional metric-name helpers or summaries
       main.py       # Wires policy, model, data, evaluate

1. Policy: Choose Actions
-------------------------

.. code-block:: python

   from sda import Policy


   class OrderUpToPolicy(Policy):
       def __init__(self, reorder_point, order_up_to):
           self.reorder_point = reorder_point
           self.order_up_to = order_up_to

       def act(self, state, env, history):
           del history
           if state.inventory < self.reorder_point:
               return self.order_up_to - state.inventory
           return 0.0

Do not load scenarios, advance SimPy processes, or log metrics inside the
policy. Its job is only to decide.

2. Model: Register SimPy Processes
----------------------------------

.. code-block:: python

   import numpy as np
   from dataclasses import dataclass

   from sda import SDAModel


   @dataclass
   class InventoryState:
       inventory: float


   class InventoryModel(SDAModel):
       def build(self, env, scenario, recorder):
           state = InventoryState(float(scenario.initial_state))
           env.process(self._run(env, scenario, recorder, state))
           return state

       def _run(self, env, scenario, recorder, state):
           for demand in np.asarray(scenario.data["demand"], dtype=float):
               order = self.policy.act(state, env, recorder.history)
               available = state.inventory + order
               sales = min(available, float(demand))
               lost_sales = max(float(demand) - available, 0.0)
               state.inventory = available - sales
               recorder.cost(order + 0.1 * state.inventory + 5.0 * lost_sales)
               recorder.log("inventory", state.inventory)
               recorder.log("lost_sales", lost_sales)
               yield env.timeout(1.0)

       def finalize(self, state, scenario, recorder):
           del scenario
           recorder.trajectory("ending_inventory", state.inventory)

3. Data: Supply Scenarios
-------------------------

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
       initial_state=50,
       batch_size=2,
   )

Use ``GeneratorDataModule`` for lazy generated futures and
``BootstrapDataModule`` for historical resampling. Use a custom
``DataModule`` when setup has state, stages, or source-specific batching.

4. Evaluation: Run The Simulation
---------------------------------

.. code-block:: python

   from sda import evaluate


   policy = OrderUpToPolicy(reorder_point=30, order_up_to=80)
   model = InventoryModel(policy)
   result = evaluate(model, data)

Use ``Simulator`` directly only when you want a reusable configured runner,
for example with an ``MLflowTracker``.

5. Results: Read The Logs
-------------------------

.. code-block:: python

   result.names()
   result["total_cost"].mean()
   result["total_cost"].percentile(95)
   result["total_cost"].cvar(0.95)
   result["inventory"].at_time(5).mean()
   result.records("total_cost")
   result.summary()

For cost metrics, lower is usually better. Use percentiles and CVaR when the
worst scenarios matter more than the average.
