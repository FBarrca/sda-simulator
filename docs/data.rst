Data Modules
============

``DataModule`` is the only scenario data abstraction. It owns scenario
construction and batching; it does not choose actions, advance model dynamics,
or store metrics.

Lifecycle
---------

.. code-block:: text

   data.prepare_data()
   data.setup(stage)
   for batch in data.batches(stage):
       ...

``prepare_data()``
   One-time shared preparation, such as loading files.

``setup(stage)``
   Stage-specific setup.

``batches(stage)``
   Yields ``ScenarioBatch`` objects.

ScenarioSpec
------------

Each ``ScenarioSpec`` describes one independent SimPy run:

``scenario_id``
   Stable integer identifier used by metric rows.

``end_time``
   Time passed to ``env.run(until=end_time)``.

``initial_state``
   Optional starting state for the model.

``data``
   Mapping of scenario data, such as demand paths, traffic paths, service
   histories, or configuration arrays.

``seed``
   Optional per-scenario seed.

``ScenarioBatch`` is a list-like container for independent specs. The
simulator runs each scenario in its own SimPy environment.

ArrayDataModule
---------------

Use ``ArrayDataModule`` when full paths are already in memory. Arrays are
scenario-first:

.. code-block:: text

   [n_scenarios, horizon, ...]

.. code-block:: python

   import numpy as np
   from sda import ArrayDataModule

   demand = np.array(
       [
           [18, 21, 16, 20],
           [22, 19, 25, 17],
           [12, 15, 14, 18],
       ]
   )

   data = ArrayDataModule(
       {"demand": demand},
       initial_state=50,
       batch_size=2,
   )

Each emitted scenario has ``scenario.data["demand"]`` equal to one row of the
array. The default ``end_time`` is the path length.

GeneratorDataModule
-------------------

Use ``GeneratorDataModule`` when futures should be generated lazily.

.. code-block:: python

   from sda import GeneratorDataModule


   def poisson_demand(*, rng, shape):
       return {"demand": rng.poisson(20, size=shape)}


   data = GeneratorDataModule(
       poisson_demand,
       horizon=12,
       n_scenarios=1000,
       batch_size=128,
       initial_state=50,
       seed=42,
   )

Generator functions may request supported context names such as ``rng``,
``shape``, ``scenario_ids``, ``horizon``, ``end_time``, ``batch_size``,
``start``, ``stop``, and ``n_scenarios``. They may return a mapping, a
``ScenarioBatch``, or a sequence of ``ScenarioSpec`` objects.

BootstrapDataModule
-------------------

Use ``BootstrapDataModule`` when futures should be resampled from historical
observations.

.. code-block:: python

   from sda import BootstrapDataModule

   data = BootstrapDataModule(
       {"demand": [18, 21, 16, 20, 23, 15]},
       horizon=4,
       n_scenarios=100,
       initial_state=50,
       method="iid",
       seed=7,
   )

Supported methods are ``iid``, ``circular_block``, ``moving_block``, and
``stationary_block``.

Custom DataModule
-----------------

Create a custom ``DataModule`` when scenario construction has meaningful
lifecycle or domain structure:

.. code-block:: python

   from sda import DataModule, ScenarioBatch, ScenarioSpec


   class MyDataModule(DataModule):
       def prepare_data(self):
           self.history = load_history()

       def setup(self, stage=None):
           self.stage = stage

       def batches(self, stage="evaluate"):
           yield ScenarioBatch(
               [
                   ScenarioSpec(
                       scenario_id=0,
                       end_time=10,
                       initial_state={"inventory": 50},
                       data={"demand": self.history[:10]},
                       seed=123,
                   )
               ]
           )

Keep policy decisions in ``Policy``, domain dynamics in ``SDAModel``, and
metric logging in the model processes through ``Recorder``.
