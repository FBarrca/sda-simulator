Concepts
========

Sequential decision analytics is about evaluating decisions that unfold over
time. Each decision affects the future state of the system, future decisions
depend on the outcomes of earlier decisions, and uncertainty enters through
exogenous information such as demand, prices, failures, or weather.

Sequential Decision Problems
----------------------------

A sequential decision problem has three recurring ingredients:

* a state that describes the current system,
* a decision that changes what happens next,
* exogenous information that is observed from outside the decision maker.

The objective is usually to minimize expected total cost or maximize expected
total reward across many possible futures.

Mathematical Framework
----------------------

In each period, a sequential decision problem follows the same basic loop:

1. Observe the current state, :math:`s_t`.
2. Choose a decision, :math:`d_t = \pi(s_t)`, using a policy :math:`\pi`.
3. Observe exogenous information, :math:`\omega_t`, such as demand, prices, or
   failures.
4. Move to the next state, :math:`s_{t+1} = f(s_t, d_t, \omega_t)`.
5. Record a cost or reward, :math:`c_t = c(s_t, d_t, \omega_t)`.

In ``sda`` code, the simulator uses zero-based period indexes
``t = 0, 1, ..., horizon - 1``. A model receives the current batch state, a
decision, and one time slice of exogenous data for all scenarios in the batch.

Inventory Example
-----------------

For a warehouse inventory problem:

* The state is the current inventory level.
* The decision is how many units to order.
* Exogenous information is random customer demand.
* The transition adds orders, subtracts sales, and carries remaining inventory.
* The cost combines ordering cost, holding cost, and stockout or lost-sales
  penalties.

A simple policy might say: if inventory falls below 30 units, order enough to
reach 80 units. Simulating this policy over many sampled demand paths estimates
its expected cost, service level, and downside risk.

Monte Carlo Evaluation
----------------------

Many sequential decision problems are too complex to solve analytically. Monte
Carlo simulation avoids enumerating every possible future:

1. Generate many exogenous futures.
2. Run the same policy through each future.
3. Log costs and domain metrics during the rollout.
4. Repeat across many scenarios to estimate expected outcomes.
5. Compare policies using distributions, percentiles, and risk measures.

This is the core workflow of ``sda``. A ``ScenarioLoader`` supplies futures,
an ``SDAModel`` defines the domain dynamics, a ``Policy`` chooses actions, and
``Metric`` objects record observations.

How ``sda`` Maps the Pieces
---------------------------

``sda`` is intentionally small. It provides the simulation loop and the
interfaces needed to plug in domain-specific components.

``ScenarioLoader``
   Produces ``ScenarioBatch`` objects containing initial states and exogenous
   futures. Each exogenous array is batch-first with shape
   ``[batch_size, horizon, ...]``.

``Policy``
   Implements the decision function :math:`\pi`. A policy can be a simple rule,
   a fitted statistical model, an optimizer, or a learned function.

``SDAModel``
   Defines the transition function :math:`f`, the cost or reward function, and
   optional per-step diagnostic information for custom metrics.

``Simulator``
   Advances each batch through time, accumulates total cost, and dispatches
   step and trajectory records to metrics.

``MetricStore`` and ``MetricSeries``
   Store raw metric observations and compute summaries such as means,
   percentiles, and CVaR after simulation.

This separation lets you compare policies, models, and scenario generators on
the same evaluation surface.

Policy Classes
--------------

Warren Powell's unified framework groups policies into four fundamental
classes. The classes fall into two broad strategies: policy search, which tunes
a policy to work well on average, and lookahead approximations, which estimate
the downstream value of a decision.

``sda`` does not force one class. Every policy is just an implementation of
``Policy.act(...)``, so simple rules and more expensive lookahead policies can
be compared on the same sampled scenarios.

Policy Function Approximations
   Direct functions from state to decision, such as rules, lookup tables,
   parametric functions, or neural networks. No optimization is solved at
   decision time. The inventory example's order-up-to rule is this kind of
   policy.

Cost Function Approximations
   Policies that optimize a deterministic or modified immediate-cost model,
   often with tunable penalties or buffers that stand in for uncertainty. In
   inventory, this might be a forecast-demand optimizer with an added safety
   stock parameter.

Value Function Approximations
   Policies that choose actions using immediate cost plus an approximation of
   downstream value, as in approximate dynamic programming or Q-learning. The
   hard part is learning a useful approximation of future value.

Direct Lookahead Approximations
   Policies that explicitly simulate or optimize possible futures before
   choosing the current decision, such as deterministic model predictive
   control, rollout, stochastic programming, or Monte Carlo tree search. These
   policies can be accurate but are often more computationally expensive.

The right policy class depends on problem structure, data, interpretability,
and compute budget. The simulator's role is to make those choices measurable.

Applications
------------

Sequential decision problems appear in many domains:

* inventory management: when to order and how much to stock,
* supply chain planning: routing, procurement, and network decisions,
* revenue management: pricing and capacity allocation,
* portfolio optimization: asset allocation over time,
* control systems: resource allocation and feedback control.

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
