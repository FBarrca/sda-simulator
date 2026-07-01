Use Cases
=========

Use ``sda`` when you have a policy that will make repeated decisions under
uncertainty and you want to evaluate it before deployment.

The practical question is usually:

.. code-block:: text

   If we follow this policy across many plausible futures,
   what cost, service, risk, or operational behavior should we expect?

How To Map A Problem
--------------------

Start by filling in this table:

.. list-table::
   :header-rows: 1

   * - SDA concept
     - Question to answer
   * - State
     - What is known when the decision is made?
   * - Decision
     - What action does the policy choose?
   * - Exogenous information
     - What uncertain events happen after the decision?
   * - Transition
     - How does the system state change?
   * - Cost
     - What does one step cost or reward?
   * - Metrics
     - What outcomes should be compared after simulation?
   * - Data module
     - Where do sampled futures come from?

Then implement those answers:

* ``Policy.act`` for the decision rule.
* ``SDAModel.transition`` and ``SDAModel.cost`` for the domain mechanics.
* A ``DataModule`` for scenario generation.
* ``evaluate(model, data)`` for the rollout.
* ``result["metric_name"]`` queries for the logged outcomes.

Inventory Replenishment
-----------------------

Question:
   How much should a warehouse order each period when demand is uncertain?

State:
   Current inventory, open orders, lead-time pipeline, recent demand, and known
   supply constraints.

Decision:
   Order quantity, reorder trigger, supplier choice, or allocation across
   warehouses.

Exogenous information:
   Customer demand, supplier delays, inbound shortfalls, returns, or price
   changes.

Cost and metrics:
   Ordering cost, holding cost, lost-sales penalty, fill rate, stockout rate,
   ending inventory, and tail cost.

Data module choice:
   Use ``GeneratorDataModule`` for a demand distribution or forecast model,
   ``BootstrapDataModule`` for historical demand, or ``ArrayDataModule`` for a
   fixed scenario set used in policy comparisons.

Logistics Dispatch
------------------

Question:
   Which dispatch policy keeps orders on time while controlling cost and
   backlog?

State:
   Pending orders, inventory by warehouse, vehicle availability, current
   locations, deadlines, and already-known disruptions.

Decision:
   Which order to serve, from which warehouse, with which vehicle, and whether
   to defer work.

Exogenous information:
   New orders, travel-time shocks, traffic, weather, vehicle outages, port
   congestion, and customer changes.

Cost and metrics:
   Distance cost, handling cost, late penalties, invalid-assignment penalties,
   on-time service, priority-weighted service, backlog, utilization, and CVaR.

Data module choice:
   Use a custom ``DataModule`` when orders, outages, event labels, and traffic
   paths need coordinated setup and batching.

Dynamic Pricing And Revenue Management
--------------------------------------

Question:
   What price or capacity allocation should be offered over time?

State:
   Remaining inventory or capacity, current prices, booking curve, competitor
   signals, time to deadline, and customer segment mix.

Decision:
   Price, discount, bid price, allocation limit, or offer set.

Exogenous information:
   Customer arrivals, willingness to pay, cancellations, competitor changes,
   market shocks, or no-shows.

Cost and metrics:
   Revenue, margin, conversion, spoilage, stockout, service denial, regret, and
   downside revenue risk.

Data module choice:
   Use ``GeneratorDataModule`` for synthetic market simulators or
   ``BootstrapDataModule`` for empirical arrival and booking patterns.

Preventive Maintenance
----------------------

Question:
   When should equipment be inspected, repaired, or replaced?

State:
   Asset age, sensor readings, recent failures, current workload, spare parts,
   and maintenance crew availability.

Decision:
   Run, inspect, repair, replace, derate, or schedule downtime.

Exogenous information:
   Failure events, degradation increments, demand load, parts delays, and
   emergency repair times.

Cost and metrics:
   Planned maintenance cost, failure cost, downtime, service interruptions,
   spare-part usage, and tail risk.

Data module choice:
   Use ``GeneratorDataModule`` for degradation or failure models, or
   ``BootstrapDataModule`` when historical event sequences are credible future
   samples.

Staffing And Service Operations
-------------------------------

Question:
   How many people or resources should be scheduled as demand changes?

State:
   Current queue, scheduled staff, skill mix, open shifts, service-level
   commitments, and time of day.

Decision:
   Staffing level, shift extension, routing rule, call-back action, or queue
   priority.

Exogenous information:
   Arrivals, service times, absenteeism, escalations, cancellations, and demand
   surges.

Cost and metrics:
   Labor cost, wait time, abandonment, overtime, service level, utilization,
   and worst-day performance.

Data module choice:
   Use ``BootstrapDataModule`` for historical arrival patterns or a custom
   ``DataModule`` when staffing stages, calendars, and service pools need setup.

Portfolio Or Cash Allocation
----------------------------

Question:
   How should capital, cash, or inventory be allocated over time under market
   uncertainty?

State:
   Current holdings, cash, exposure limits, obligations, risk budget, and known
   signals.

Decision:
   Buy, sell, rebalance, hedge, allocate, reserve cash, or trigger a limit
   action.

Exogenous information:
   Returns, prices, liquidity, demand claims, withdrawals, interest rates, or
   shocks.

Cost and metrics:
   Return, transaction cost, drawdown, shortfall, constraint violation, VaR,
   CVaR, and liquidity usage.

Data module choice:
   Use ``ArrayDataModule`` for fixed stress scenarios, ``BootstrapDataModule``
   for historical returns, or ``GeneratorDataModule`` for model-based paths.

What To Build First
-------------------

For a new application, keep the first version small:

1. Pick one baseline policy that a domain expert understands.
2. Use one state representation that contains only information available at
   decision time.
3. Generate a small number of scenario paths.
4. Log total cost and two or three domain metrics.
5. Compare the baseline against one alternative policy.

Once that works, make the data more realistic, add metrics, and tune the
policy. The value of ``sda`` is that the same evaluation surface keeps working
as the policy and scenario model become more serious.
