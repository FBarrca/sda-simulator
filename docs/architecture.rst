Architecture Map
================

``sda`` is a small kernel with one public workflow:

.. code-block:: python

   policy = MyPolicy(...)
   model = MyModel(policy)
   data = MyDataModule(...)
   result = evaluate(model, data)

Package Boundaries
------------------

``sda.core``
   Canonical contracts and records: ``ScenarioBatch``, ``Policy``,
   ``SDAModel``, ``StepRecord``, and ``TrajectoryRecord``.

``sda.data``
   ``DataModule`` plus concrete data modules for arrays, generators, and
   bootstrap history.

``sda.simulation``
   ``evaluate``, ``Simulator``, and ``SimulationResult``. ``Simulator`` is the
   trainer-like configured runner for simulations.

``sda.metrics``
   Metric records, metric storage, query helpers, and built-in cost metrics.

``sda.tracking``
   Optional experiment tracking integrations such as MLflow result-summary
   logging.

``sda.__init__``
   The preferred public import surface.

Architecture Rules
------------------

* ``sda.core`` has no internal imports.
* ``sda.data`` turns scenario sources into ``ScenarioBatch`` streams through
  ``DataModule.batches(stage)``.
* ``sda.simulation`` orchestrates data lifecycle, rollout order, and metrics.
  It does not define policies, scenario generation, model dynamics, or metric
  meanings.
* ``sda.metrics`` reads records; it does not create data or choose policies.
* ``sda.tracking`` logs completed results; it does not run simulations or
  compute domain metrics.
* The public vocabulary is ``Simulator``, not ``Trainer``. Do not add a
  ``Trainer`` alias or a second orchestration class.

Public Import Map
-----------------

Use the top-level package for application code:

.. code-block:: python

   from sda import (
       ArrayDataModule,
       BootstrapDataModule,
       DataModule,
       GeneratorDataModule,
       Policy,
       SDAModel,
       evaluate,
   )

Data Lifecycle
--------------

``evaluate`` handles the ``DataModule`` lifecycle:

.. code-block:: text

   data.prepare_data()
   data.setup(stage)
   for batch in data.batches(stage):
       rollout batch

Stage-specific behavior belongs inside ``setup(stage)`` or ``batches(stage)``.

Rollout Loop
------------

Each ``ScenarioBatch`` follows the same loop:

.. code-block:: text

   model.initial_state(batch)
   for each t in horizon:
       model.decide(state, t, history)
       model.transition(state, decision, exogenous_t, t)
       model.cost(state, decision, exogenous_t, next_state, t)
       model.info(...)
       metrics.on_step(...)
   metrics.on_trajectory(...)

The decision is made before ``exogenous_t`` is exposed to the model. A policy
sees only ``state``, ``t``, and completed ``history``. Information observed
before the decision should already be part of ``state``; uncertainty realized
after the decision belongs in ``ScenarioBatch.exogenous``.

Minimal Mental Model
--------------------

Keep application code separated by responsibility:

``data``
   A ``DataModule`` that yields scenario batches.

``policy``
   A ``Policy`` implementation.

``model``
   An ``SDAModel`` implementation.

``metrics``
   Optional domain metrics.

Lightning-Inspired Analogy
--------------------------

``Simulator`` plays the role that ``Trainer`` plays in PyTorch Lightning, but
with simulation semantics. ``SDAModel`` plus ``Policy`` replace
``LightningModule``, ``DataModule`` keeps the data lifecycle role, ``Metric``
objects replace ``self.log(...)`` calls, and ``MLflowTracker`` is the minimal
MLflow logger equivalent.
