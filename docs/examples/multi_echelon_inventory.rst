Multi-Echelon Inventory Optimization Walkthrough
=================================================

This walkthrough reconstructs the simulation-optimization problem from
Agarwal, A. (2019), `Multi-echelon Supply Chain Inventory Planning using
Simulation-Optimization with Data Resampling
<https://arxiv.org/abs/1901.00090>`_ (arXiv:1901.00090).

The business problem is a classic inventory-optimization use case: a company
ships one product from a single source through a small distribution network
to customer-facing locations. Holding too much stock ties up working capital
across every node; holding too little causes stockouts at the locations that
customers actually order from. The task is to find, per node, how much
safety stock to carry and when to trigger a resupply order, so that the
network meets its service commitments while tying up as little inventory as
possible.

The example lives in ``examples/multi_echelon_inventory`` and is not part of
the installed ``sda`` package API.

1. Understand The Network
--------------------------

The network has one source node, node ``0`` (a plant or vendor that is
assumed to have unlimited supply and is never out of stock), and five
downstream stocking locations connected in a small tree. Node ``1`` is a
regional stocking point that only resupplies other nodes; nodes ``2``,
``4``, and ``5`` sell directly to customers and each carries a 95% fill-rate
target; node ``3`` is an intermediate transshipment point with no service
target of its own (its own service level is tracked for diagnostics, but the
optimizer does not have to keep it above any threshold).

.. image:: ../../examples/multi_echelon_inventory/multi_echelon_network.svg
   :alt: Six-node multi-echelon inventory network with lead times labeled on each lane

Each lane between two nodes has its own empirical lead time (3-4 days for
the upper branch, 2 days for the lower branch), sampled with extra random
delay from real logged delivery data rather than assumed. This matters for
the business story: a node that is farther from the source, or served by a
slower lane, needs more safety stock to protect the same service level, even
if its demand looks identical to a node that is closer to the source.

The adjacency matrix is intentionally small enough to inspect by eye. That
makes the example useful as a reference implementation: policy logic is
separated from SimPy process mechanics, and data generation is isolated in
the data module, so each piece can be understood, tested, and swapped
independently.

2. The Decision: A Base-Stock Policy Function Approximation
------------------------------------------------------------

Every stocking node in the network is controlled by the same decision rule,
a **base-stock policy with a reorder point**: each day, if a node's
inventory position (on-hand stock plus stock already on order, minus
backorders) falls to or below its reorder point, the node places a
replenishment order that brings its inventory position back up to its
base-stock level.

In the vocabulary from :doc:`../concepts` (Warren Powell's four policy
classes), this rule is a **policy function approximation (PFA)**: a fixed,
parametric mapping from state to action, with no embedded forecast, cost
model, or lookahead search. The "intelligence" of the policy is entirely in
the two numbers per node -- the reorder point and the base-stock level -- not
in any run-time computation. This is deliberate: an (s, S)-style PFA is
simple to explain to operations staff, cheap to execute at scale (one
comparison and one subtraction per node per day), and is the same family of
rule most real-world inventory systems already run in production. The value
SDA adds is not a smarter rule; it is a fast, honest way to find the right
numbers for the rule.

``BaseStockReorderPolicy`` (in ``examples/multi_echelon_inventory/policies.py``)
implements this:

.. code-block:: python

   class BaseStockReorderPolicy(Policy):
       def act(self, state, env, history):
           # for every scenario and every node with an upstream supplier:
           #   if inventory_position <= reorder_tolerance * reorder_point[node]:
           #       order_quantity[node] = base_stock[node] - on_hand_inventory[node]

``reorder_tolerance`` (default ``1.05``) adds a small safety margin so an
order fires slightly before the position drops exactly to the reorder point,
matching the reference implementation's behavior under daily, discrete
checks rather than continuous monitoring.

3. The Hyperparameters: What You Are Actually Tuning
------------------------------------------------------

The policy has ten tunable numbers: a reorder point and a base-stock level
for each of the five stocking nodes (the source node is excluded and simply
set to a very large base stock, since it is assumed unconstrained). These
ten numbers are exactly the hyperparameters that a black-box optimizer
searches over, and they are exactly what you would change if you wanted this
policy to behave differently in production.

``policy_parameters_from_guess(...)`` (in
``examples/multi_echelon_inventory/domain.py``) defines the vector layout
used everywhere in this example, matching the original optimizer scripts:

.. code-block:: text

   guess = [excess_1, excess_2, excess_3, excess_4, excess_5,
            rop_1,    rop_2,    rop_3,    rop_4,    rop_5]

   base_stock[node] = reorder_point[node] + excess[node]

Raising a node's reorder point makes that node reorder sooner (more safety
stock, higher service, more capital tied up). Raising its excess (the gap
between base stock and reorder point) makes each replenishment order larger
and less frequent, which changes the height and width of the sawtooth
pattern in the inventory trace without changing when it starts. The chart
below compares the optimizer's initial guess to the published, tuned
solution, so you can see concretely how much each node's buffer moved and in
which direction:

.. image:: ../../examples/multi_echelon_inventory/multi_echelon_policy_parameters.svg
   :alt: Bar chart comparing initial-guess and published reorder-point and base-stock hyperparameters per node

Reading this chart: node 1 (the regional stocking point) is the most
over-provisioned in the initial guess and loses the most inventory after
tuning (base stock ``3000 -> 2516``), because it does not have its own
customer-facing service target and only needs to keep the downstream nodes
supplied through their own lead times. Nodes 3, 4, and 5 barely move,
because their initial guess was already close to what the network needs.
This is the practical value of optimization here: it is not about holding
less inventory everywhere, it is about moving the safety stock to where the
service target actually requires it.

To change these hyperparameters yourself, either edit
``REFERENCE_EXCESS_INVENTORY_GUESS`` / ``REFERENCE_ROP_GUESS`` in
``domain.py``, or pass a new ten-element vector directly:

.. code-block:: python

   from examples.multi_echelon_inventory import build_policy

   policy = build_policy([2000, 350, 700, 150, 400, 1000, 250, 200, 150, 200])
   print(policy.reorder_point, policy.base_stock)

4. Watch The Hyperparameters Drive Behavior
----------------------------------------------

Daily diagnostics make the connection between hyperparameters and simulated
behavior visible day by day. They are opt-in
(``record_daily_metrics=True`` or ``--daily-metrics``) because optimizers
usually only need the scalar objective and enabling them slows down a run.
When enabled, the model records per-day total on-hand inventory plus
node-level demand, shipments, lost sales, backorders, on-hand inventory, and
inventory position.

.. image:: ../../examples/multi_echelon_inventory/multi_echelon_inventory_trace.svg
   :alt: Daily inventory trace for the network total and node 2, with node 2's reorder point and base stock drawn as reference lines

The thick green line is node 2's on-hand inventory; the dashed lines are its
own published reorder point and base-stock level. You can watch the policy
in action: inventory drifts down as customer demand and downstream orders
are served, the line touches the lower dashed line and a replenishment order
fires, and the line jumps back up once that shipment clears its lead time.
This is the base-stock hyperparameters, made visible. If you tightened node
2's reorder point, the dashed floor line would drop and the policy would
tolerate deeper dips before reordering; if you shrank its excess inventory,
each jump back up would be smaller and orders would fire more often.

5. Read The Objective
-----------------------

The reference optimizer evaluates 20 seeded replications and minimizes:

.. code-block:: text

   average on-hand inventory
   + 1.0e6 * sum(max(0, service target - average service level))

This is lexicographic in spirit: missing a service target is enormously
expensive, so a good hyperparameter vector must first clear every service
target and only then compete on how little inventory it holds. Once every
target is met, the service penalty term is zero and the objective is simply
average on-hand inventory, which is why the published solutions below show a
clean inventory reduction with no service tradeoff.

.. list-table::
   :header-rows: 1

   * - Mode
     - Initial objective
     - Published objective
     - Change
   * - Lost sales
     - ``2783.462``
     - ``2445.776``
     - ``-12.1%``
   * - Backorder
     - ``2767.635``
     - ``2515.907``
     - ``-9.1%``

.. image:: ../../examples/multi_echelon_inventory/multi_echelon_objective.svg
   :alt: Multi-echelon objective and service-level scorecard

The scorecard is the main business result: the optimized hyperparameter
vectors cut average on-hand inventory by roughly 9-12% while still clearing
every measured service target, so the service penalty stays at zero in both
modes. Lost-sales and backorder are two different assumptions about what
happens to a customer order the network cannot fill immediately (the order
is lost forever, versus fulfilled late once stock arrives); they lead to
slightly different optimal buffers because a backorder mode can "catch up"
on missed demand while a lost-sales mode cannot, so it front-loads a little
more safety stock at the customer-facing nodes.

6. Why Simulate At All: The Emulator's Job
----------------------------------------------

You cannot safely try ten new reorder points and base-stock levels directly
on a live supply chain: a bad guess means real stockouts or real wasted
capital, and you would not find out for weeks. The SimPy model in this
example is a data-driven emulator of the network that lets you answer "what
would happen if we ran with these hyperparameters?" in seconds, as many
times as you like, before anything touches production:

* it resamples real historical demand and lead-time-delay observations
  (bootstrap sampling, not an assumed distribution), so the simulated days
  look statistically like real ones without assuming a particular demand
  curve or lead-time distribution;
* it runs the same 20 seeded replications every time a hyperparameter vector
  is evaluated, so two candidate vectors are compared on the same set of
  simulated futures rather than on noise;
* it turns any ten-number hyperparameter vector into a single scalar
  objective through ``get_objective(...)``, which is exactly the interface a
  black-box optimizer (``scipy.optimize``, ``skopt.gp_minimize``, or
  ``rbfopt``, as used by the original reference scripts) needs to search the
  hyperparameter space automatically instead of by trial and error;
* it exposes the same dense daily diagnostics used in section 4 above, so
  once an optimizer proposes a candidate vector, an operator can inspect
  *why* it works, not just trust the number.

In short: the model is the safe sandbox in which the policy's hyperparameters
get tuned, and the optimizer is the search procedure that explores that
sandbox on your behalf. Neither replaces judgment about what the objective
should reward; the ``1.0e6`` service penalty, the choice of lost-sales versus
backorder accounting, and which nodes carry a target are all business
decisions encoded before optimization ever runs.

7. Put It Together: The Full Workflow
------------------------------------------

Run the default lost-sales initial-guess evaluation:

.. code-block:: bash

   uv run -m examples.multi_echelon_inventory

Evaluate the backorder mode:

.. code-block:: bash

   uv run -m examples.multi_echelon_inventory --mode backorder

Evaluate the published, already-tuned solution instead of the initial guess:

.. code-block:: bash

   uv run -m examples.multi_echelon_inventory --published-solution

Objective evaluations are fast by default and emit the final objective,
service-level, and average on-hand metrics. Turn on dense per-day
diagnostics when you need to explain a result, not just score it:

.. code-block:: bash

   uv run -m examples.multi_echelon_inventory --daily-metrics

Regenerate the example SVGs (including the ones embedded on this page) with:

.. code-block:: bash

   python3 -m examples.multi_echelon_inventory.visualize

The same data / model / policy / evaluate flow used by the other SDA
examples applies here too:

.. code-block:: python

   from examples.multi_echelon_inventory import (
       build_data,
       build_model,
       build_policy,
       summarize_reference_result,
   )
   from sda import evaluate

   policy = build_policy([2000, 350, 700, 150, 400, 1000, 250, 200, 150, 200])
   data = build_data(n_scenarios=20, batch_size=1)
   model = build_model(policy=policy)
   result = evaluate(model, data)
   summary = summarize_reference_result(result)
   print(summary.objective, summary.average_on_hand, summary.service_level)

To wire this into an external black-box optimizer exactly like the original
``getObj`` reference function:

.. code-block:: python

   from examples.multi_echelon_inventory import get_objective

   value = get_objective([2000, 350, 700, 150, 400, 1000, 250, 200, 150, 200])
   # pass get_objective to scipy.optimize.minimize, skopt.gp_minimize, or
   # rbfopt.RbfoptAlgorithm the same way the reference scripts did

``evaluate_reference_policy(...)`` and ``get_objective(...)`` are convenience
wrappers over ``build_evaluation(...)``; they sit on top of the same
``MultiEchelonInventoryDataModule`` and ``MultiEchelonInventoryModel`` used
everywhere else in this example. Pass ``record_daily_metrics=True`` to
``build_model(...)``, ``build_result(...)``, ``build_evaluation(...)``,
``evaluate_reference_policy(...)``, or ``get_objective(...)`` to include the
dense daily traces from section 4.

Source Layout
-------------

The example follows the same responsibility split as the framework:

* ``examples/multi_echelon_inventory/domain.py`` defines the network
  topology, lead times, service targets, and the hyperparameter-vector
  layout shared by the policy, model, and optimizer wrappers.
* ``examples/multi_echelon_inventory/data.py`` loads the empirical CSV files
  and creates seeded scenario batches.
* ``examples/multi_echelon_inventory/policies.py`` defines the base-stock
  reorder policy function approximation.
* ``examples/multi_echelon_inventory/models.py`` defines the SimPy-backed
  network dynamics, order queues, in-transit shipments, lost-sales
  accounting, and backorder accounting.
* ``examples/multi_echelon_inventory/metrics.py`` lists emitted metric names
  and reconstructs the reference objective from SDA metric records.
* ``examples/multi_echelon_inventory/main.py`` provides ``build_data``,
  ``build_policy``, ``build_model``, ``build_result``, and the command-line
  entrypoint.
* ``examples/multi_echelon_inventory/optimization.py`` exposes
  ``get_objective(...)`` and ``evaluate_reference_policy(...)`` for
  black-box optimization loops.
* ``examples/multi_echelon_inventory/visualize.py`` generates every SVG on
  this page directly from live evaluations, so the figures never drift from
  the code.

The SDA version provides three practical improvements over the original,
copied reference scripts:

* the objective can be called directly from tests, docs, notebooks, or
  black-box optimizers, instead of being buried inside a simulation script;
* the policy (hyperparameters), model (network dynamics), empirical data,
  and metric summary each live in one obvious, independently testable place;
* detailed daily traces are available on demand for explaining a result,
  without slowing down the normal objective-only evaluation loop used during
  optimization.

The copied CSV inputs and ``REFERENCE_LICENSE`` live under the example
directory so the source-tree example is self-contained.
