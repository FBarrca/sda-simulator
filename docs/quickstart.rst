Quickstart
==========

Import the public API from the installed package, ``sda``. From a source
checkout, run commands from the repository root so the package and examples are
importable.

.. code-block:: python

   from sda import (
       ArrayScenarioLoader,
       Policy,
       SDAModel,
       Simulator,
       StepCostMetric,
       TotalCostMetric,
   )

Minimal Simulation
------------------

This example evaluates a deterministic demand-accumulation model. It is small,
but it has the same pieces as a larger sequential decision problem:

* a policy that maps state to decisions,
* a model that applies transitions and costs,
* exogenous futures shaped by scenario and time,
* metrics queried after simulation.

.. code-block:: python

   import numpy as np

   class ZeroPolicy(Policy):
       def act(self, state, t, history):
           return np.zeros_like(state, dtype=float)

   class DemandAccumulationModel(SDAModel):
       def transition(self, state, decision, exogenous, t):
           return np.asarray(state, dtype=float) + np.asarray(
               exogenous["demand"],
               dtype=float,
           )

       def cost(self, state, decision, exogenous, next_state, t):
           return np.asarray(exogenous["demand"], dtype=float)

   scenarios = ArrayScenarioLoader(
       initial_state=np.zeros(3),
       exogenous={
           "demand": np.array(
               [
                   [1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9],
               ],
               dtype=float,
           )
       },
       batch_size=2,
   )

   model = DemandAccumulationModel(ZeroPolicy())
   simulator = Simulator(metrics=[StepCostMetric(), TotalCostMetric()])
   result = simulator.evaluate(model, scenarios)

   result.metric("total_cost").values()
   result.metric("total_cost").mean()
   result.metric("total_cost").percentile(95)
   result.metric("total_cost").cvar(0.95)

The three scenarios have total costs ``[6, 15, 24]`` because each total is the
sum of one demand path. The ``step_cost`` metric has one observation per
scenario per period.

Scenario Data
-------------

``ArrayScenarioLoader`` expects each exogenous path to be batch-first:

.. code-block:: text

   [n_scenarios, horizon, ...]

In the quickstart, demand has shape ``[3, 3]``: three scenarios and three
periods. With ``batch_size=2``, the loader yields one batch with scenarios
``[0, 1]`` and one batch with scenario ``[2]``.

``initial_state`` can be:

* a scalar, which is broadcast to each scenario in a batch,
* a vector with one entry per scenario,
* a mapping whose values are scalars or per-scenario vectors.

Model Hooks
-----------

An ``SDAModel`` must define:

``transition(state, decision, exogenous, t)``
   Returns the next state for the current batch.

``cost(state, decision, exogenous, next_state, t)``
   Returns a scalar cost or one-dimensional vector with one value per scenario.

Models can also override ``info(...)`` to return extra values for custom
metrics. The inventory example uses this to expose fill rate, lost sales, and
order quantity.

Distribution-First Metrics
--------------------------

Metric values are stored as raw observations. Summary statistics are computed
later from the stored distribution, so you can query the same run in several
ways:

.. code-block:: python

   result.metric("total_cost").summary()
   result.metric("step_cost").at_time(1).values()
   result.metric("step_cost").step_level().mean()
   result.metric("total_cost").trajectory_level().percentile(95)

Use step-level metrics for per-period values such as inventory, service level,
or instantaneous cost. Use trajectory-level metrics for whole-scenario values
such as total cost.
