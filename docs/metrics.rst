Metrics And Logs
================

Metrics in ``sda`` are distribution-first. During rollout, metrics log numeric
observations into an in-memory ``MetricStore``. After rollout, a
``SimulationResult`` lets you query those observations:

.. code-block:: python

   result["total_cost"].mean()
   result["total_cost"].percentile(95)
   result["total_cost"].cvar(0.95)
   result["ending_inventory"].at_time(5).mean()
   result.summary()

Think of the log as a table of metric rows, not as printed console output.
Each row has a metric name, value, scenario id, optional time index, and level.

Default Logs
------------

``evaluate(model, data)`` records two cost metrics by default:

``step_cost``
   One row per scenario per period. Use it to inspect period-by-period cost.

``total_cost``
   One row per scenario after the full trajectory. Use it to compare policies
   by average cost, percentiles, and downside risk.

Add domain metrics with ``extra_metrics`` when you want to keep those defaults:

.. code-block:: python

   from sda import evaluate, info_metric, step_metric


   result = evaluate(
       model,
       data,
       extra_metrics=[
           step_metric("ending_inventory", lambda step: step.next_state),
           info_metric("lost_sales"),
       ],
   )

Pass ``metrics=[...]`` only when you want to replace the defaults completely.
Pass ``metrics=[]`` to run without recording metrics.

Reading Results
---------------

``SimulationResult`` returns ``MetricSeries`` objects:

.. code-block:: python

   series = result["total_cost"]
   series.values()
   series.records()
   series.count()
   len(series)

Common summary queries:

.. code-block:: python

   result.names()
   result["total_cost"].mean()
   result["total_cost"].std()
   result["total_cost"].min()
   result["total_cost"].max()
   result["total_cost"].percentile(95)
   result["total_cost"].cvar(0.95)
   result.summary()

Use ``mean`` for average performance, ``percentile`` for distribution shape,
and ``cvar(0.95)`` for the average value in the worst 5% of observations. For
cost metrics, lower is usually better.

Raw Rows
--------

Export raw rows from one metric or from the whole result:

.. code-block:: python

   result.records("total_cost")
   result.records()

Each row dictionary has:

``name``
   Metric name.

``value``
   Numeric observation.

``scenario_id``
   Scenario identifier, when the value belongs to one scenario.

``t``
   Time index for step-level rows.

``level``
   ``"step"`` or ``"trajectory"``.

Filtering
---------

Filter by time or level:

.. code-block:: python

   result["ending_inventory"].at_time(5).mean()
   result["step_cost"].step_level().mean()
   result["total_cost"].trajectory_level().percentile(95)

For step metrics, use ``to_trajectory_matrix`` when you need one row per
scenario and one column per time:

.. code-block:: python

   scenario_ids, times, inventory = result["ending_inventory"].to_trajectory_matrix()

This is useful for plotting trajectories, computing per-scenario time-series
features, or exporting a matrix to another analysis tool.

One-Line Metrics
----------------

Use ``step_metric`` for values available on each ``StepRecord``:

.. code-block:: python

   from sda import step_metric


   ending_inventory = step_metric(
       "ending_inventory",
       lambda step: step.next_state,
   )

Use ``info_metric`` for values your model exposes through ``SDAModel.info``:

.. code-block:: python

   from sda import info_metric, step_metric


   lost_sales = info_metric("lost_sales")
   stockout = step_metric(
       "stockout",
       lambda step: step.info["lost_sales"] > 0,
   )

Use ``trajectory_metric`` for values available after a full batch rollout:

.. code-block:: python

   from sda import trajectory_metric


   ending_state = trajectory_metric(
       "ending_state",
       lambda trajectory: trajectory.final_state,
   )

Metric functions may return a scalar or one value per scenario. Scalars are
broadcast when scenario ids are present.

Reusable Metric Classes
-----------------------

When a metric is part of an example or application API, wrap the one-line
metric in a named class:

.. code-block:: python

   from sda import InfoMetric, StepMetric


   class LostSalesMetric(InfoMetric):
       def __init__(self):
           super().__init__("lost_sales")


   class InventoryMetric(StepMetric):
       def __init__(self):
           super().__init__("ending_inventory", lambda step: step.next_state)

For complex metrics that need state or custom behavior, subclass ``Metric``
directly and implement ``on_step`` and/or ``on_trajectory``.

Where Metric Values Should Come From
------------------------------------

Keep metrics thin:

* Put domain calculations in ``SDAModel.transition``, ``SDAModel.cost``, or
  ``SDAModel.info`` when the model already knows how to compute them.
* Use ``info_metric`` for direct ``info`` values.
* Use ``step_metric`` for simple expressions over a ``StepRecord``.
* Use ``trajectory_metric`` for full-trajectory values.
* Subclass ``Metric`` only when a metric needs its own state or custom logging.

Avoid putting policy choices, data loading, or transition logic inside metrics.
Metrics should read records and log observations.

Optional MLflow Tracking
------------------------

``SimulationResult`` is always available locally. If you also want experiment
tracking, pass ``MLflowTracker`` to ``evaluate``. The tracker logs aggregate
summary values such as ``total_cost.mean`` and ``total_cost.p95`` to MLflow
after the simulation completes.

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
