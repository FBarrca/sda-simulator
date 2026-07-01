Quickstart
==========

This page builds a complete SimPy-native inventory simulation.

The Complete Program
--------------------

.. code-block:: python

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
               fill_rate = 1.0 if demand == 0 else sales / float(demand)
               cost = order + 0.1 * state["inventory"] + 5.0 * lost_sales

               recorder.cost(cost)
               recorder.log("inventory", state["inventory"])
               recorder.log("lost_sales", lost_sales)
               recorder.log("fill_rate", fill_rate)
               yield env.timeout(1.0)

       def finalize(self, state, scenario, recorder):
           del scenario
           recorder.trajectory("ending_inventory", state["inventory"])


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

   policy = OrderUpToPolicy(reorder_point=30, order_up_to=80)
   model = InventoryModel(policy)
   result = evaluate(model, data)

   print(result.names())
   print(result["total_cost"].mean())
   print(result["total_cost"].percentile(95))
   print(result["inventory"].at_time(2).mean())
   print(result["fill_rate"].event_level().mean())

What Each Piece Means
---------------------

``OrderUpToPolicy``
   The decision rule. It sees the current state, the current SimPy
   environment, and completed recorder history.

``InventoryModel``
   The domain rules. ``build`` registers SimPy processes for one scenario.
   Processes call ``recorder.cost`` and ``recorder.log`` as events happen.

``ArrayDataModule``
   The scenario source. It turns batch-first arrays into independent
   ``ScenarioSpec`` objects.

``evaluate(model, data)``
   The standard entrypoint. It calls the data lifecycle, creates one SimPy
   environment per scenario, runs each environment until ``scenario.end_time``,
   and returns a ``SimulationResult``.

Rollout Order
-------------

For each scenario:

.. code-block:: text

   env = simpy.Environment()
   recorder = Recorder(...)
   state = model.build(env, scenario, recorder)
   env.run(until=scenario.end_time)
   model.finalize(state, scenario, recorder)
   recorder.close()

The SimPy event clock, ``env.now``, is the canonical time. Metric rows store
that event time.

Reading The Logs
----------------

``recorder.cost(value)`` logs an event-level ``cost`` row and adds it to the
scenario total. ``recorder.close()`` logs a trajectory-level ``total_cost``
row. Domain metrics come from ``recorder.log`` and ``recorder.trajectory``.

.. code-block:: python

   result.names()
   result["total_cost"].values()
   result["total_cost"].mean()
   result["total_cost"].percentile(95)
   result["inventory"].at_time(2).mean()
   result.records("lost_sales")

Choosing Scenario Data
----------------------

``ArrayDataModule``
   Use when full sample paths are already in memory.

``GeneratorDataModule``
   Use when futures should be generated lazily from a distribution, forecast
   model, service call, or domain simulator.

``BootstrapDataModule``
   Use when futures should be resampled from historical observations.

Custom ``DataModule``
   Use when scenario construction has setup, fitted state, stages, or
   source-specific batching logic.
