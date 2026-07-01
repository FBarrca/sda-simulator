Concepts
========

Sequential Decision Analytics is about evaluating decisions that unfold over
time. A decision changes the state of the system, the new state affects the
next decision, and uncertainty enters along the way.

In one sentence: SDA asks, "If I use this policy over many possible futures,
what outcomes should I expect?"

The Small Mental Model
----------------------

Every SDA problem can be described with four ideas:

``state``
   What the decision maker knows now. Examples: current inventory, open
   orders, available vehicles, cash position, machine health, or staff on duty.

``decision``
   What the policy chooses now. Examples: order quantity, dispatch assignment,
   price, maintenance action, asset allocation, or staffing level.

``exogenous information``
   What happens outside the decision maker's control. Examples: demand,
   traffic, outages, failures, service times, weather, prices, or arrivals.

``cost`` or ``reward``
   How one step is scored. Examples: operating cost, late penalty, lost sales,
   revenue, service-level penalty, risk, or constraint violation.

The repeated loop is:

.. code-block:: text

   observe state
   choose decision
   observe uncertainty
   update state
   log cost and metrics

``sda`` keeps these responsibilities separate so you can compare policies,
models, and scenario data without rewriting the simulator.

How ``sda`` Names The Pieces
----------------------------

``Policy``
   Implements the decision rule. It maps ``state``, ``t``, and completed
   ``history`` to a decision.

``SDAModel``
   Implements the domain rules: initial state, transition, cost, and optional
   diagnostics for metrics.

``DataModule``
   Supplies batches of sampled futures. A future is a full path of exogenous
   information over the horizon.

``Simulator`` and ``evaluate``
   Run the data lifecycle, rollout each batch, and dispatch records to
   metrics. Most users start with ``evaluate(model, data)``.

``SimulationResult``
   Stores the logged metric observations and provides summaries such as means,
   percentiles, and CVaR.

Information Timing
------------------

``sda`` uses a decision-before-uncertainty convention. At time ``t``,
``Policy.act(state, t, history)`` is called before the period's exogenous
sample-path value is exposed to the model.

That gives a simple modeling rule:

* If the policy is allowed to know it before deciding, put it in ``state``.
* If it is uncertain until after the decision, put it in ``exogenous``.

For example, a warehouse policy may know current inventory, open purchase
orders, and yesterday's demand because those are part of the current state. It
should not know today's customer demand before deciding what to order unless
that demand has genuinely already arrived.

Scenarios And Batches
---------------------

A scenario is one possible future path. For inventory, one scenario might be:

.. code-block:: python

   demand = [18, 21, 16, 20]

A richer logistics scenario might contain several paths:

.. code-block:: python

   exogenous = {
       "orders": [...],
       "traffic_multiplier": [...],
       "vehicle_outages": [...],
   }

A ``ScenarioBatch`` groups several scenarios together so the model can run
vectorized NumPy code. Exogenous arrays are batch-first and time-second:

.. code-block:: text

   [batch_size, horizon, ...]

During rollout, the simulator reveals one time slice at a time. The policy
never receives the full future path.

Mathematical Loop
-----------------

The same idea can be written compactly as:

1. Observe the current state, :math:`S_t`.
2. Choose a decision, :math:`x_t = \pi(S_t)`, using policy :math:`\pi`.
3. Observe exogenous information, :math:`W_{t+1}`.
4. Move to the next state, :math:`S_{t+1} = f(S_t, x_t, W_{t+1})`.
5. Record cost, :math:`c_t = c(S_t, x_t, W_{t+1})`.

In ``sda`` code, the simulator uses zero-based indexes
``t = 0, 1, ..., horizon - 1``. The model receives one batch state, one batch
decision, and one batch slice of exogenous information at each step.

Inventory Example
-----------------

For a warehouse inventory problem:

* The state is the current inventory level.
* The decision is how many units to order.
* Exogenous information is random customer demand.
* The transition adds orders, subtracts sales, and carries remaining inventory.
* The cost combines ordering cost, holding cost, and lost-sales penalty.
* Metrics might include total cost, fill rate, ending inventory, and stockout
  frequency.

A simple policy might say: if inventory falls below 30 units, order enough to
reach 80 units. Simulating this policy over many demand paths estimates its
average cost and downside risk.

Monte Carlo Evaluation
----------------------

Many sequential decision problems are too complex to solve exactly. Monte
Carlo simulation uses sampled futures instead:

1. Generate many full sample paths of exogenous information.
2. Run the same policy through each path, one step at a time.
3. Log costs and domain metrics during the rollout.
4. Compare policies using distributions, percentiles, and risk measures.

This is the core workflow of ``sda``. A ``DataModule`` supplies futures, an
``SDAModel`` defines the domain dynamics, a ``Policy`` chooses actions, and
``Metric`` objects record observations.

Policy Classes
--------------

Warren Powell's unified framework groups policies into four broad classes.
``sda`` does not force one class; every policy is just an implementation of
``Policy.act(...)``.

Policy Function Approximations
   Direct functions from state to decision, such as rules, lookup tables,
   parametric functions, or neural networks. The order-up-to inventory policy
   is this kind of policy.

Cost Function Approximations
   Policies that optimize a simplified immediate-cost model, often with
   tunable buffers or penalties that stand in for uncertainty.

Value Function Approximations
   Policies that choose actions using immediate cost plus an approximation of
   downstream value, as in approximate dynamic programming or Q-learning.

Direct Lookahead Approximations
   Policies that simulate or optimize possible futures before choosing the
   current decision, such as model predictive control, rollout, stochastic
   programming, or Monte Carlo tree search.

The simulator's role is to make these choices measurable on the same sampled
futures.

When SDA Fits
-------------

SDA is a good fit when:

* decisions repeat over time,
* the state after one decision affects later decisions,
* uncertainty changes outcomes,
* you can generate or resample plausible futures,
* you care about distributions, not only one forecast.

SDA is less useful when the problem is a one-shot calculation, the decision
does not affect future state, or there is no meaningful uncertainty to test.

References
----------

For deeper background on sequential decision problems and stochastic
optimization:

* Powell, W. B. (2022). *Reinforcement Learning and Stochastic Optimization:
  A Unified Framework for Sequential Decisions*.
* Powell, W. B. (2007). *Approximate Dynamic Programming: Solving the Curses
  of Dimensionality*.
* Bertsekas, D. P. (2005). *Dynamic Programming and Optimal Control*.
* Puterman, M. L. (1994). *Markov Decision Processes: Discrete Stochastic
  Dynamic Programming*.
