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

1. Define The Business Goal
---------------------------

From a business standpoint, the dispatcher is not trying to simply move the
most orders or minimize today's kilometers. The goal is to protect customer
service and margin under uncertain demand, traffic, and vehicle availability.
A good dispatch policy should:

* deliver high-priority orders on time,
* avoid letting backlog grow into an unserviceable queue,
* control distance, handling, late-delivery, and bad-week tail costs,
* use fleet capacity sensibly without dispatching low-value work just to keep
  trucks busy,
* remain robust across many plausible futures, not only one average day.

This means the business goal has tradeoffs. A policy that dispatches every easy
order can still fail if urgent orders miss their deadlines. A policy that keeps
late cost low by dispatching almost nothing can also fail if backlog growth is
unacceptable. The report therefore reads every policy through both cost and
operational metrics.

2. Translate The Goal Into A Model Objective
--------------------------------------------

The simulator evaluates policies by minimizing expected total cost over sampled
future demand and disruption scenarios. Each day's cost is:

``dispatch cost``
   Distance cost plus per-unit handling cost for orders dispatched that day.

``late cost``
   A penalty for delivered orders that miss their deadline. The penalty scales
   with late days, order quantity, and priority.

``overdue backlog cost``
   A smaller daily penalty for pending orders that are already past deadline.
   In the default model this is ``2.0 * priority`` per overdue order-day.

``invalid assignment cost``
   A penalty for infeasible assignments, such as using an unavailable vehicle or
   drawing more inventory than a warehouse has.

This objective rewards low cost and on-time high-priority delivery. It does not
hard-code a minimum dispatch volume, minimum vehicle utilization, or maximum
backlog constraint. Those operational requirements are tracked as metrics, but
they are not hard constraints unless the model is tuned to make them so. This
matters when interpreting the rollout policy: a policy can look excellent on
total cost while still allowing backlog to grow.

3. Run The Default Dispatch Policy
----------------------------------

Start with the default priority policy:

.. code-block:: bash

   uv run -m examples.logistics

If your environment already has the package requirements installed:

.. code-block:: bash

   python3 -m examples.logistics

The command prints distribution summaries for total cost, worst-tail cost,
on-time service, priority-weighted service, late cost, backlog, dispatch volume,
and vehicle utilization.

4. Inspect The Network
----------------------

The network uses a 3 x 12 warehouse-to-customer distance matrix over an
OpenStreetMap basemap. Heavy lanes show each customer's nearest warehouse;
faint lanes show alternate feasible origins.

.. image:: ../../examples/logistics/logistics_network.png
   :alt: Spanish logistics network over an OpenStreetMap basemap

5. Inspect Synthetic Demand
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

6. Build Bootstrap Scenarios
----------------------------

``LogisticsDataModule`` samples contiguous 7-day blocks from synthetic history.
Each batch contains:

* ``orders`` with shape ``[batch_size, horizon]`` as tuples of ``Order`` records,
* ``traffic_multiplier`` with shape ``[batch_size, horizon, warehouse, customer]``,
* ``vehicle_outages`` with shape ``[batch_size, horizon, vehicle]``,
* ``event_labels`` and ``history_day_index`` for inspection and repeatability.

The core simulator calls ``decide`` before each day's exogenous information is
revealed. To respect that order, new same-day orders are appended to
``pending_orders`` for the next day's dispatch decision.

.. code-block:: python

   data = LogisticsDataModule(
       horizon=28,
       n_scenarios=500,
       batch_size=64,
       seed=42,
   )

7. Compare Dispatch Policies
----------------------------

The example includes eight dispatch policies:

``RandomPolicy``
   Randomly shuffles feasible assignments and greedily keeps a conflict-free
   set. This is the baseline policy.

``GreedyPolicy``
   Sorts feasible assignments by shortest warehouse-to-customer lane, then
   greedily keeps a conflict-free set.

``PriorityPolicy``
   Scores priority, quantity, deadline pressure, rescue pressure, duration, and
   lane distance before greedily selecting compatible assignments.

``MilpPolicy``
   Solves the same priority-distance objective globally with one-order,
   one-vehicle, and warehouse-SKU inventory constraints when SciPy is available.
   It falls back to ``PriorityPolicy`` if no MILP solution is available.

``LookaheadRolloutPolicy``
   Compares priority, greedy, and defer-first decisions by rolling out sampled
   futures with ``PriorityPolicy`` as the continuation policy. If no rollout
   model is bound, it degrades to ``PriorityPolicy``.

``NearestFeasiblePolicy``
   Dispatches pending orders FIFO from the nearest warehouse with enough stock
   and an available vehicle.

``PriorityDeadlinePolicy``
   Dispatches high-priority and tight-deadline orders first, then chooses the
   nearest feasible warehouse and vehicle.

``RiskAwareDispatchPolicy``
   Scores priority, deadline slack, lane distance, stock scarcity, and vehicle
   fit before choosing assignments.

8. Interpreting Results
-----------------------

Running the comparison command prints a text table:

.. code-block:: bash

   uv run -m examples.logistics.policy_comparison --horizon 28 --n-scenarios 500 --batch-size 64 --seed 42 --no-plot

This run produces:

.. code-block:: text

   Logistics policy comparison (28-day horizon, 500 scenarios, seed 42)
   policy                  total_mean        total_ci95  cost_cvar95  prio_ot  late_cost  backlog  dispatch/day   util
   ----------------------  ----------  ----------------  -----------  -------  ---------  -------  ------------  -----
   Random                      281011  (276337, 285685)       418635    25.5%     7330.5     79.0           4.1  68.6%
   Greedy                      140936  (136459, 145412)       282695    58.8%     3228.1     61.0           5.5  91.6%
   Priority                    372243  (365599, 378887)       560571     8.6%    11450.0     77.6           4.2  70.3%
   MILP distance-priority      372243  (365599, 378887)       560571     8.6%    11450.0     77.6           4.2  70.3%
   Lookahead rollout           106336  (104172, 108500)       169106    95.8%      117.0    123.9           0.5   8.5%
   Nearest feasible            379476  (372056, 386896)       588208    12.6%    11608.4     74.0           4.5  74.7%
   Priority deadline           217434  (211623, 223245)       383052    22.7%     5925.5     73.7           4.5  75.1%
   Risk aware                  217707  (211935, 223480)       381989    22.6%     5929.9     73.8           4.5  74.9%

How to read the table columns:

``total_mean`` (lower is better)
   Average total cost over all bootstrap futures.

``total_ci95``
   Approximate 95% confidence interval for mean total cost. If two policies'
   intervals overlap heavily, their ranking is not very robust.

``cost_cvar95`` (lower is better)
   Conditional Value-at-Risk for total cost: the average cost in the worst 5%
   of scenarios. This is the downside-risk number.

``prio_ot`` (higher is better)
   Priority-weighted on-time service. This matters more than raw delivery count
   because missing high-priority orders is expensive.

``late_cost`` (lower is better)
   Average late-delivery penalty per scenario-day.

``backlog`` (lower is better)
   Average number of orders waiting for dispatch.

``dispatch/day``
   Average dispatched orders per day. This is throughput, not value: moving
   more orders is not automatically better if they are the wrong orders.

``util``
   Average fraction of vehicles dispatched each day.

Clear conclusions from this run:

* Best objective value: ``LookaheadRolloutPolicy``. It has the lowest mean cost
  (106,336), lowest tail cost (CVaR 169,106), lowest late cost (117), and best
  priority-weighted on-time service (95.8%). Under the current cost model, this
  is the winning policy.
* Important caveat: rollout wins by being very selective. It dispatches only
  0.5 orders per day and uses only 8.5% of the fleet, while backlog rises to
  123.9 orders. That is good for the current cost objective, but it may be
  operationally unacceptable if clearing backlog or using capacity is a hard
  business requirement.
* Best simple high-throughput policy: ``GreedyPolicy``. It dispatches the most
  orders (5.5 per day), uses the fleet heavily (91.6%), and is the best
  non-rollout policy by mean cost. It is still much worse than rollout on
  late cost, tail risk, and priority service.
* Middle tier: ``PriorityDeadlinePolicy`` and ``RiskAwareDispatchPolicy``.
  They are almost tied. Risk-aware has slightly better tail cost, while
  priority-deadline has slightly better mean cost. Both are better than random
  but worse than greedy and rollout.
* Do not use ``NearestFeasiblePolicy`` for this objective. FIFO nearest-feasible
  dispatch performs worst by mean cost and tail cost because it ignores which
  orders are expensive to miss.
* ``PriorityPolicy`` and ``MilpPolicy`` need recalibration. They tie because the
  MILP optimizes the same one-day score as ``PriorityPolicy``; in this scenario
  that score is misaligned with realized cost, so global optimization does not
  help.

The business decision is therefore conditional. If the objective is exactly the
simulated cost function, choose ``LookaheadRolloutPolicy``. If the operation
must also clear backlog or keep trucks utilized, do not deploy that rollout
unchanged; increase the backlog/deferment penalty or add minimum-dispatch or
service constraints, then rerun the comparison. If a cheap rule is needed
today, ``GreedyPolicy`` is the strongest simple policy in this table, while
``PriorityDeadlinePolicy`` and ``RiskAwareDispatchPolicy`` are safer
deadline-aware candidates after tuning.

9. Read The Metrics
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

   result["total_cost"].mean()
   result["total_cost"].percentile(95)
   result["total_cost"].cvar(0.95)

Use step-level metrics for trajectory views:

.. code-block:: python

   scenario_ids, times, backlog = result["pending_backlog"].to_trajectory_matrix()

The important tradeoff is not only average cost. A useful dispatch policy
should also keep high-priority service high, control backlog growth, and reduce
the worst-case tail of total cost.
