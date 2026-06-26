API Reference
=============

The public package API is re-exported from ``sda``. The sections below document
the modules that define scenario data, model contracts, simulation execution,
and metrics.

Data
----

Use ``sda.data`` when preparing exogenous futures. ``ArrayScenarioLoader`` is
the built-in loader for NumPy-compatible arrays. Custom loaders can implement
``ScenarioLoader`` and yield ``ScenarioBatch`` objects.

.. automodule:: sda.data
   :members:
   :undoc-members:
   :show-inheritance:

Model
-----

Use ``sda.model`` to define policies and domain dynamics. A ``Policy`` chooses
decisions, while an ``SDAModel`` owns transition, cost, and optional info
hooks. ``StepRecord`` and ``TrajectoryRecord`` are the records metrics receive
during simulation.

.. automodule:: sda.model
   :members:
   :undoc-members:
   :show-inheritance:

Simulation
----------

Use ``sda.simulation`` to evaluate a model on scenarios. ``Simulator`` runs the
rollout loop and returns a ``SimulationResult`` for metric queries.

.. automodule:: sda.simulation
   :members:
   :undoc-members:
   :show-inheritance:

Metrics
-------

Use ``sda.metrics`` to log and query observations. The built-in metrics record
step cost and total trajectory cost; custom metrics can log any scalar or
per-scenario vector exposed by a step or trajectory record.

.. automodule:: sda.metrics
   :members:
   :undoc-members:
   :show-inheritance:
