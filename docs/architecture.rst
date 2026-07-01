Architecture Map
================

``sda`` has one public workflow:

.. code-block:: python

   policy = MyPolicy(...)
   model = MyModel(policy)
   data = MyDataModule(...)
   result = evaluate(model, data)

Package Boundaries
------------------

``sda.core``
   Canonical contracts and records: ``ScenarioSpec``, ``ScenarioBatch``,
   ``EventRecord``, ``Policy``, and ``SDAModel``.

``sda.data``
   ``DataModule`` plus concrete scenario sources for arrays, generators, and
   bootstrap history.

``sda.simulation``
   ``evaluate``, ``Simulator``, and ``SimulationResult``. ``Simulator`` owns
   data lifecycle and SimPy environment execution.

``sda.metrics``
   ``Recorder``, ``MetricStore``, and ``MetricSeries``. Metrics are emitted by
   SimPy processes and queried after evaluation.

``sda.tracking``
   Optional MLflow result-summary logging.

``sda.__init__``
   The preferred public import surface.

Data Lifecycle
--------------

.. code-block:: text

   data.prepare_data()
   data.setup(stage)
   for batch in data.batches(stage):
       for scenario in batch.scenarios:
           run scenario

Stage-specific behavior belongs inside ``setup(stage)`` or
``batches(stage)``.

Scenario Lifecycle
------------------

.. code-block:: text

   env = simpy.Environment()
   recorder = Recorder(store, scenario_id=scenario.scenario_id, env=env)
   state = model.build(env, scenario, recorder)
   env.run(until=scenario.end_time)
   model.finalize(state, scenario, recorder)
   recorder.close()

If ``build`` returns a SimPy generator, the simulator schedules it as a
process. Models may also register processes inside ``build`` and return a
domain state object.

Responsibility Boundaries
-------------------------

``Policy``
   Decision logic only.

``SDAModel``
   SimPy processes, state transitions, costs, and domain diagnostics.

``DataModule``
   Scenario construction and batching.

``Recorder`` / ``MetricStore``
   Metric observation logging and querying.

``Simulator``
   Orchestration, not domain rules.

Lightning-Inspired Analogy
--------------------------

``Simulator`` plays the small configured-runner role that ``Trainer`` plays in
PyTorch Lightning, but with simulation vocabulary. ``SDAModel`` plus
``Policy`` replace ``LightningModule``, ``DataModule`` keeps the data
lifecycle role, and ``MLflowTracker`` is the minimal MLflow logger equivalent.
