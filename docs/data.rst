Data Modules
============

In Sequential Decision Analytics, a policy is evaluated by rolling it forward
through many possible realizations of the exogenous information process.
``sda`` uses ``DataModule`` as the single abstraction for supplying those
realizations to the simulator.

The distinction is important:

* a ``DataModule`` supplies full sample paths of exogenous information,
* an ``SDAModel`` defines how state evolves and how cost is measured,
* a ``Policy`` chooses decisions from the current state,
* the simulator combines all three to produce state, decision, cost, and metric
  trajectories.

A data module should not contain policy logic or transition logic. Its job is
to answer: what values of uncertainty should the model use to test the
policy's decisions?

SDA Framing
-----------

At each time step, the simulator follows the standard SDA loop:

.. code-block:: text

   state S_t
       -> policy chooses decision x_t
       -> exogenous information W_{t+1} is revealed
       -> model computes S_{t+1} and cost

In code, ``W`` is stored in the ``exogenous`` mapping of a ``ScenarioBatch``.
One scenario is one full sampled path of future exogenous information over the
horizon. A batch is several independent sample paths handled together for
vectorized simulation.

The policy does not receive ``exogenous``. It sees ``state``, ``t``, and the
completed ``history``. If information is known before the decision, put it in
the state. If information is uncertain until after the decision, put it in
``exogenous`` so the simulator reveals it to ``transition`` and ``cost``.

For an inventory model, one scenario might be:

.. code-block:: python

   demand = [18, 21, 16, 20]

For a richer model, the same scenario might include several exogenous
processes:

.. code-block:: python

   exogenous = {
       "demand": [18, 21, 16, 20],
       "unit_cost": [4.2, 4.1, 4.4, 4.3],
       "supplier_delay": [1, 0, 2, 1],
   }

The ``DataModule`` creates the full exogenous sample paths. The simulator turns
them into simulated trajectories only after it combines them with a policy and
model.

ScenarioBatch
-------------

``DataModule.batches(stage)`` yields ``ScenarioBatch`` objects. Each batch
contains:

``initial_state``
   The starting state for the scenarios in this batch. It can be a scalar, one
   value per scenario, or a mapping of state fields.

``exogenous``
   A mapping of exogenous information names to arrays. Each array is
   batch-first and time-second.

``scenario_ids``
   Stable identifiers for the sample paths in the batch. Metrics use these ids
   to keep observations aligned.

The array convention is:

.. code-block:: text

   [batch_size, horizon, ...]

For example:

.. code-block:: python

   import numpy as np

   demand = np.array(
       [
           [18, 21, 16, 20],  # scenario 0
           [22, 19, 25, 17],  # scenario 1
           [12, 15, 14, 18],  # scenario 2
       ]
   )

This is a batch of three exogenous sample paths. During rollout, the simulator
passes one time slice to the model after each decision. At ``t=0``, the value
revealed to the transition and cost hooks is ``[18, 22, 12]``; at ``t=1`` it is
``[21, 19, 15]``.

Full Paths, Stepwise Rollout
----------------------------

``sda`` gives each ``ScenarioBatch`` the full exogenous path for every scenario
in the batch. The simulator still advances one decision epoch at a time. It
does not hand the future path to the policy, and it only hands one time slice
to the transition and cost hooks.

Conceptually, each batch is rolled out like this:

.. code-block:: text

   batch.exogenous["demand"] has shape [batch_size, horizon]

   for t in 0, 1, ..., horizon - 1:
       choose decision x_t from the current state
       reveal W_{t+1} = batch.exogenous[:, t, ...]
       compute next state and cost from S_t, x_t, and W_{t+1}

This design keeps policy evaluation stepwise while making the uncertainty
sample path explicit. That matters because:

* policies can be compared on the same sample paths,
* seeded data modules are reproducible,
* bootstrap methods can preserve time structure across a whole path,
* the simulator can validate shapes before rollout,
* metrics can stay aligned by ``scenario_id`` across steps and trajectories.

So a data module provides full exogenous paths, while the simulator reveals
those paths one period at a time after each decision. Observed signals and
other pre-decision information should be part of the state instead.

Choosing A DataModule
---------------------

Different projects have different ways of modeling ``W``. The built-in data
modules cover the common cases.

``ArrayDataModule``
   Use when full sample paths are already constructed. This is best for small
   examples, tests, deterministic scenarios, backtests, cached forecasts, or
   common-random-number policy comparisons where every policy should see the
   same exogenous paths.

``GeneratorDataModule``
   Use when full sample paths come from a stochastic model, forecast model,
   external service, or domain simulator. It generates one batch at a time, so
   you do not need to materialize all scenarios in memory.

``BootstrapDataModule``
   Use when full sample paths should be data-driven and resampled from
   historical observations. Block bootstrap methods help preserve temporal
   dependence in the exogenous process.

Custom ``DataModule``
   Use when preparing ``W`` has lifecycle: loading files, fitting transforms,
   branching by stage, coordinating multiple sources, or implementing
   source-specific batching rules.

ArrayDataModule: Fixed Sample Paths
-----------------------------------

``ArrayDataModule`` wraps exogenous sample paths that already exist in memory.
Every exogenous array must have shape ``[n_scenarios, horizon, ...]``.

.. code-block:: python

   import numpy as np
   from sda import ArrayDataModule

   demand_paths = np.array(
       [
           [18, 21, 16, 20],
           [22, 19, 25, 17],
           [12, 15, 14, 18],
       ]
   )

   data = ArrayDataModule(
       {"demand": demand_paths},
       initial_state=50,
       batch_size=2,
   )

This module is useful when you want the exogenous information to be fixed and
inspectable. If two policies are evaluated on the same ``ArrayDataModule``,
they face the same sample paths; differences in results come from the policies,
not from different draws of uncertainty.

GeneratorDataModule: Model-Based Sample Paths
---------------------------------------------

``GeneratorDataModule`` calls a generator function once per batch. The
generator can return an exogenous mapping or a complete ``ScenarioBatch``.

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

Use this when the exogenous information model is procedural: a probability
distribution, a forecasting model, a simulator, or a service call. The
generator may request any supported context names:

``rng``
   NumPy random generator seeded by the data module.

``shape``
   ``(batch_size, horizon)`` for simple two-dimensional paths.

``scenario_ids``
   The ids included in the current batch.

``horizon``, ``batch_size``, ``start``, ``stop``, ``n_scenarios``
   Batch and experiment metadata.

For example:

.. code-block:: python

   import numpy as np

   def demand_with_weekly_pattern(*, rng, scenario_ids, horizon):
       day = np.arange(horizon) % 7
       baseline = np.where(day < 5, 24, 14)
       return {
           "demand": rng.poisson(
               baseline,
               size=(len(scenario_ids), horizon),
           )
       }

With a seed, repeated iterations over ``batches()`` are deterministic. That
keeps evaluation repeatable while still letting the sample paths be generated
lazily.

BootstrapDataModule: Historical Sample Paths
--------------------------------------------

``BootstrapDataModule`` resamples sample paths from historical observations.
Use it when the historical record is the most natural model of the exogenous
information process.

.. code-block:: python

   from sda import BootstrapDataModule

   data = BootstrapDataModule(
       {"demand": demand_history},
       initial_state=50,
       horizon=12,
       n_scenarios=1000,
       batch_size=128,
       method="stationary",
       block_size=7,
       seed=42,
   )

Bootstrap methods:

``"iid"``
   Samples individual observations independently. Use when the order of
   observations is not important.

``"circular"`` or ``"circular_block"``
   Samples fixed-length contiguous blocks and wraps around the end of history.

``"moving"`` or ``"moving_block"``
   Samples fixed-length contiguous blocks without wrapping.

``"stationary"`` or ``"stationary_block"``
   Samples variable-length blocks. This is often a good default when you want
   to preserve dependence without fixed block boundaries.

Block methods are useful when ``W`` is serially dependent: daily demand with a
weekly pattern, weather, correlated prices, service times, or other processes
where nearby observations should stay together.

Custom DataModules: Data Lifecycle
----------------------------------

Subclass ``DataModule`` when producing the exogenous information process has
setup or stage-specific behavior. The lifecycle is intentionally small:

``prepare_data()``
   One-time preparation, such as loading shared raw data.

``setup(stage)``
   Stage-specific setup, such as choosing a horizon, fitting preprocessing
   state, or selecting a stress regime.

``batches(stage)``
   Yield ``ScenarioBatch`` objects for the requested stage.

.. code-block:: python

   from sda import BootstrapDataModule, DataModule

   class DemandData(DataModule):
       def prepare_data(self):
           self.history = load_history()

       def setup(self, stage=None):
           self.horizon = 24 if stage == "stress" else 12

       def batches(self, stage="evaluate"):
           yield from BootstrapDataModule(
               self.history,
               initial_state=50,
               horizon=self.horizon,
               n_scenarios=1000,
               batch_size=128,
               method="stationary",
               block_size=7,
               seed=42,
           ).batches(stage=stage)

``evaluate`` calls the lifecycle in order:

.. code-block:: text

   data.prepare_data()
   data.setup(stage)
   for batch in data.batches(stage):
       simulate batch

Keep stage-specific data choices inside ``setup(stage)`` or
``batches(stage)``. The policy and model should not need to know whether the
exogenous information came from a normal evaluation stage, a backtest, or a
stress stage.

Invariants
----------

All data modules should preserve these rules:

* Exogenous arrays are batch-first and time-second:
  ``[batch_size, horizon, ...]``.
* Full in-memory sample paths use ``[n_scenarios, horizon, ...]``.
* ``scenario_ids`` align with the first dimension of every exogenous array.
* Step exogenous values preserve the batch dimension and drop only the time
  dimension.
* Costs are scalar or one-dimensional with one value per scenario.
* ``initial_state`` may be scalar, per-scenario, or a mapping of those.
