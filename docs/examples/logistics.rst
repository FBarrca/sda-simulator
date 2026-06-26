Logistics Dispatch Walkthrough
==============================

This walkthrough sets up a road-freight dispatch problem across Spain. A
distribution company operates warehouses in Madrid, Barcelona, and Valencia and
serves twelve customer locations with six heterogeneous vehicles.

Each day, the simulator:

* dispatches feasible pending orders from the current state,
* reveals new orders, traffic delays, event labels, and vehicle outages,
* updates inventory, vehicles, backlog, deliveries, and costs,
* records service, backlog, utilization, and tail-risk metrics.

The example lives in ``examples/logistics`` and is not part of the installed
``sda`` package API.

1. Run The Default Dispatch Policy
----------------------------------

Start with the default risk-aware policy:

.. code-block:: bash

   uv run -m examples.logistics

If your environment already has the package requirements installed:

.. code-block:: bash

   python3 -m examples.logistics

The command prints distribution summaries for total cost, worst-tail cost,
on-time service, priority-weighted service, late cost, backlog, dispatch volume,
and vehicle utilization.

2. Inspect The Network
----------------------

The network uses a 3 x 12 warehouse-to-customer distance matrix over an
OpenStreetMap basemap. Heavy lanes show each customer's nearest warehouse;
faint lanes show alternate feasible origins.

.. image:: ../../examples/logistics/logistics_network.png
   :alt: Spanish logistics network over an OpenStreetMap basemap

3. Inspect Synthetic Demand
---------------------------

``synthetic_history(days, seed)`` creates deterministic order history,
traffic multipliers, vehicle outages, and event labels. The seed controls the
entire history, so demand, traffic, and outages are reproducible.

The synthetic demand includes:

* weekday and weekend demand rhythm,
* annual seasonality with a year-end peak,
* promotions and holiday peaks that lift order volume,
* severe weather and port congestion that increase travel time and outages,
* SKU mix across ``AMBIENT_FOOD``, ``COLD_CHAIN``, ``ELECTRONICS``, and
  ``PHARMA``.

.. image:: ../../examples/logistics/logistics_synthetic_demand.png
   :alt: Synthetic logistics demand by SKU and day of week

4. Build Bootstrap Scenarios
----------------------------

``LogisticsScenarioLoader`` samples contiguous 7-day blocks from synthetic
history. Each batch contains:

* ``orders`` with shape ``[batch_size, horizon]`` as tuples of ``Order`` records,
* ``traffic_multiplier`` with shape ``[batch_size, horizon, warehouse, customer]``,
* ``vehicle_outages`` with shape ``[batch_size, horizon, vehicle]``,
* ``event_labels`` and ``history_day_index`` for inspection and repeatability.

The core simulator calls ``decide`` before each day's exogenous information is
revealed. To respect that order, new same-day orders are appended to
``pending_orders`` for the next day's dispatch decision.

.. code-block:: python

   scenarios = LogisticsScenarioLoader(
       horizon=28,
       n_scenarios=500,
       batch_size=64,
       seed=42,
   )

5. Compare Dispatch Policies
----------------------------

The example includes three policy rules:

``NearestFeasiblePolicy``
   Dispatches pending orders FIFO from the nearest warehouse with enough stock
   and an available vehicle.

``PriorityDeadlinePolicy``
   Dispatches high-priority and tight-deadline orders first, then chooses the
   nearest feasible warehouse and vehicle.

``RiskAwareDispatchPolicy``
   Scores priority, deadline slack, lane distance, stock scarcity, and vehicle
   fit before choosing assignments.

.. image:: ../../examples/logistics/logistics_policy_comparison.png
   :alt: Logistics dispatch policy comparison scorecard

6. Read The Metrics
-------------------

The simulation reports built-in ``step_cost`` and ``total_cost`` metrics plus
logistics-specific metrics:

* ``on_time_rate``
* ``priority_weighted_on_time_rate``
* ``late_cost``
* ``dispatch_cost``
* ``pending_backlog``
* ``dispatched_order_count``
* ``vehicle_utilization``

Use the total-cost distribution for risk-sensitive comparisons:

.. code-block:: python

   result.metric("total_cost").mean()
   result.metric("total_cost").percentile(95)
   result.metric("total_cost").cvar(0.95)

Use step-level metrics for trajectory views:

.. code-block:: python

   scenario_ids, times, backlog = result.metric("pending_backlog").to_trajectory_matrix()

The important tradeoff is not only average cost. A useful dispatch policy
should also keep high-priority service high, control backlog growth, and reduce
the worst-case tail of total cost.
