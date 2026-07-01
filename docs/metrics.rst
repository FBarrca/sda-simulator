Metrics And Logs
================

Metrics in ``sda`` are event rows emitted by SimPy processes. A model receives
a scenario-local ``Recorder`` in ``SDAModel.build`` and uses it to log numeric
observations.

Logging From A Model
--------------------

.. code-block:: python

   def _run(self, env, scenario, recorder, state):
       for demand in scenario.data["demand"]:
           ...
           recorder.cost(cost)
           recorder.log("inventory", state.inventory)
           recorder.log("fill_rate", fill_rate)
           yield env.timeout(1.0)

   def finalize(self, state, scenario, recorder):
       recorder.trajectory("ending_inventory", state.inventory)

``recorder.cost(value)``
   Logs an event-level ``cost`` metric at ``env.now`` and adds the value to
   the scenario total.

``recorder.log(name, value, tags=None)``
   Logs an event-level domain metric at ``env.now``.

``recorder.trajectory(name, value, tags=None)``
   Logs a trajectory-level metric at ``env.now``.

``recorder.close()``
   Called by the simulator after ``finalize``. It logs trajectory-level
   ``total_cost`` once.

Rows
----

Each row has:

``name``
   Metric name.

``value``
   Numeric observation.

``scenario_id``
   Scenario identifier.

``time``
   SimPy event time from ``env.now``.

``level``
   ``"event"`` or ``"trajectory"``.

``tags``
   Optional string metadata, such as a node id or lane name.

Reading Results
---------------

``SimulationResult`` returns ``MetricSeries`` objects:

.. code-block:: python

   result.names()
   result["total_cost"].values()
   result["total_cost"].mean()
   result["total_cost"].percentile(95)
   result["total_cost"].cvar(0.95)
   result.summary()

Filter series by time, level, or tag:

.. code-block:: python

   result["inventory"].at_time(5).mean()
   result["inventory"].event_level().mean()
   result["total_cost"].trajectory_level().percentile(95)
   result["on_hand"].with_tag("node", "3").values()

Export raw rows:

.. code-block:: python

   result.records("total_cost")
   result.records()

Trajectory Matrices
-------------------

For event metrics that have one value per scenario and time, use
``to_trajectory_matrix``:

.. code-block:: python

   scenario_ids, times, inventory = result["inventory"].to_trajectory_matrix()

The returned matrix has shape ``[n_scenarios, n_times]`` and uses ``nan`` for
missing scenario-time cells.

Optional MLflow Tracking
------------------------

Pass ``MLflowTracker`` to ``evaluate`` to log aggregate summaries to MLflow
after the simulation completes:

.. code-block:: python

   from sda import MLflowTracker, evaluate

   result = evaluate(
       model,
       data,
       tracking=MLflowTracker(
           experiment_name="inventory",
           run_name="order-up-to",
           params={"policy": "order_up_to"},
       ),
   )

MLflow tracking is optional and requires the ``mlflow`` extra dependency.
