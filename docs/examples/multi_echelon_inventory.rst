Multi-Echelon Inventory Optimization Walkthrough
================================================

This walkthrough reconstructs the simulation-optimization problem from
Agarwal, A. (2019), `Multi-echelon Supply Chain Inventory Planning using
Simulation-Optimization with Data Resampling
<https://arxiv.org/abs/1901.00090>`_ (arXiv:1901.00090). The network, the
reorder rule, the empirical data, and the objective are kept as published; this
page explains what the resulting tool does and why it is worth building.

Where the :doc:`inventory` example tunes one stocking location, this one tunes a
**network** of them at once -- the setting where inventory decisions are hardest,
because a change at one node ripples through every node downstream.

.. admonition:: The bottom line

   Re-tuning the network's base-stock levels against replayed historical demand
   and lead times holds **9-12% less average inventory while clearing every
   customer service target** -- nothing is traded away for the saving. On this
   six-node network that is ≈ ``$10k-$13.5k`` of working capital released per
   product (illustrative); the percentage reduction is assumption-free and
   scales with every product that shares the network.

1. The Network and the Problem
------------------------------

A company ships one product from a single source through a small distribution
network to the locations that sell to customers. The network has one source (a
plant or vendor assumed never to run out) feeding five downstream stocking
locations: one regional stocking point, one intermediate transshipment point,
and three customer-facing locations.

.. image:: ../../examples/multi_echelon_inventory/multi_echelon_network.svg
   :alt: Six-node multi-echelon inventory network with lead times labeled on each lane

Every lane carries its own delivery **lead time**, and that lead time is not
perfectly reliable -- it varies, drawn from real logged delivery history.
Holding too much stock anywhere ties up capital; holding too little at a
customer-facing location causes stockouts. A node that sits farther from the
source, or behind a slower and less predictable lane, must carry a larger
**safety buffer** to protect the same service level as one closer in. The task
is to decide, for every stocking location, how much buffer is actually worth
carrying -- jointly, because the nodes are coupled.

2. What Happens Each Day
------------------------

Beneath the network, the same four steps play out every simulated day at every
stocking location:

1. **Review inventory position and decide whether to reorder.** If a location's
   inventory position (on-hand plus on-order) has fallen to its reorder point,
   it places a replenishment order on the location upstream.
2. **The upstream location fills that order** from its own stock.
3. **The order travels for its lead time** -- not instantaneous, and the number
   of days is drawn from real historical delivery records, delays included.
4. **Customers place demand.** At each customer-facing location, real historical
   order volumes are served from stock on hand. Unmet demand is either lost
   (lost-sales mode) or filled late once stock arrives (backorder mode),
   depending on which business assumption is under test.

This is not a forecast layered on a spreadsheet: it is the real sequence of
events in the network, replayed against real historical data one day at a time.

3. The Policy: A Base-Stock Rule
--------------------------------

Every stocking location is governed by the same deliberately small decision
rule. In sequential-decision-analytics terms this is a **policy function
approximation** (PFA): a readable function mapping today's observed state to
today's action. Here the state is the location's inventory position and on-hand
stock; the action is how much to order from upstream.

The PFA has two business-readable parameters per node:

* **Reorder point** (``R``): when inventory position drops to this level, start a
  replenishment order.
* **Base-stock / order-up-to level** (``B``): the level the location refills
  toward.

The rule is intentionally this simple:

.. code-block:: python

   def order_quantity(on_hand, inventory_position, trigger, target, *, tolerance=1.05):
       if inventory_position <= tolerance * trigger:
           return target - on_hand
       return 0.0

The trigger test uses **inventory position**, not just physical on-hand stock.
Inventory position counts stock already in transit, so the policy does not
panic-order simply because a shipment is on its way. The ``1.05`` tolerance is
carried over from the reference implementation to keep the comparison robust
around the trigger; conceptually the two numbers the business owns are still the
reorder point ``R`` and base stock ``B``.

Across the five stocking locations the policy therefore has **ten parameters**:
``R1``-``R5`` and ``B1``-``B5``. The source node is treated as unconstrained, so
it is not part of the tuning problem.

4. Releasing Capital: Tuning the Policy Levels
----------------------------------------------

Once the policy is explicit, the improvement question becomes concrete: **can we
release working capital from inventory without degrading the customer
experience?** Here that means optimizing the ten parameters jointly, across the
whole network and across every simulated historical replication.

The optimizer does not tune each node in isolation -- the nodes are coupled. A
lower target at the regional node can starve the customer nodes downstream; a
higher trigger at a customer node protects fill rate but ties up more local
stock. The right answer is the joint set of levels that minimizes the reference
objective:

.. code-block:: text

   objective = average on-hand inventory
             + 1.0e6 * total service-level shortfall

The large penalty encodes a lexicographic priority: **protect service first,
then drive inventory down.** The reference optimizer parameterizes each node as
``excess above trigger`` plus ``trigger``:

.. code-block:: text

   x = [e1, e2, e3, e4, e5, R1, R2, R3, R4, R5]
   B_i = e_i + R_i

so the search is still choosing the two levels the business cares about: reorder
point ``R_i`` and base stock ``B_i`` for every stocking node.

.. image:: ../../examples/multi_echelon_inventory/multi_echelon_policy_parameters.svg
   :alt: Bar chart comparing initial-guess and optimized reorder-point and base-stock parameters per node

Crucially, the optimized lost-sales policy does not simply cut every buffer. It
releases most stock from the regional stocking point while *keeping or slightly
increasing* selected downstream protection:

.. list-table::
   :header-rows: 1

   * - Node
     - Reorder-point change
     - Base-stock change
     - Interpretation
   * - 1
     - ``1000 -> 729``
     - ``3000 -> 2516``
     - Large release from the regional buffer.
   * - 2
     - ``250 -> 276``
     - ``600 -> 643``
     - Slightly more protection near customer demand.
   * - 3
     - ``200 -> 198``
     - ``900 -> 937``
     - Similar trigger, a little more transshipment buffer.
   * - 4
     - ``150 -> 159``
     - ``300 -> 308``
     - Small downstream protection increase.
   * - 5
     - ``200 -> 220``
     - ``600 -> 625``
     - Small downstream protection increase.

That is the business story the chart is built to show: optimization is not "hold
less everywhere." It is "put the buffer where the network actually needs it, and
release the rest."

5. Testing Safely on Historical Data
-------------------------------------

Trying a new set of levels directly on the live supply chain is risky: a bad
guess means real stockouts or real cash tied up, and you would not know it was
wrong for weeks. Instead, the example runs the same network inside a simulation
and replays real historical demand and delivery data against it. Demand values
and lead-time delays come from the CSV history bundled with the reference
problem; each seeded replication **bootstraps** from those histories to
construct another plausible year, so the policy is tested against many realistic
futures rather than one.

.. image:: ../../examples/multi_echelon_inventory/multi_echelon_inventory_trace.svg
   :alt: Daily policy traces for nodes 1 through 5 with a side topology map showing where each node sits in the network

The trace follows all five stocking nodes. Each row uses its own y-scale,
because the regional buffer is far larger than the customer-facing ones; a
shared axis would hide the downstream behavior. Within each row, the green line
is **inventory position** (the signal the policy watches) and the blue line is
**on-hand inventory** (the stock customers or downstream nodes can actually draw
from today). The topology inset maps the row numbers back to the network, so it
is clear which rows are upstream buffers and which are customer-facing.

The dashed lines mark each node's reorder point and base stock, and the
triangles mark inferred reorder days. Reading across rows shows how the same PFA
behaves differently by echelon: upstream nodes carry larger buffers and
replenish downstream facilities, while customer-facing nodes cycle around
smaller local targets. That is more informative than a plain on-hand plot
because it shows both sides of the tradeoff at once -- whether each node reorders
early enough to protect service, and whether it holds more stock than the tuned
policy actually needs.

6. Did It Work?
---------------

Two outcomes matter: how much stock the network holds, and whether every
customer-facing location keeps its promised fill rate. Because a missed service
promise is penalized far more heavily than any inventory saved, a good set of
ten numbers must protect service first and only then compete on how little stock
it needs.

The numbers below are **means over 20 seeded bootstrap replications**, reported
with their 95% confidence intervals. Every service target is met in all four
cases, so the penalty term is zero and the objective equals the network's
**average on-hand inventory** -- the figures are units of stock, not a blended
score:

.. list-table::
   :header-rows: 1

   * - Mode
     - Starting point (avg on-hand, 95% CI)
     - Tuned result (avg on-hand, 95% CI)
     - Change
   * - Lost sales
     - ``2783 (2753-2814)``
     - ``2446 (2408-2484)``
     - ``-12.1%``
   * - Backorder
     - ``2768 (2732-2804)``
     - ``2516 (2488-2544)``
     - ``-9.1%``

.. image:: ../../examples/multi_echelon_inventory/multi_echelon_objective.svg
   :alt: Multi-echelon objective and service-level scorecard

The starting-point and tuned intervals do not overlap in either mode, so the
9-12% reduction is a genuine effect rather than replication noise. And because
every service target is still met, nothing was traded away to achieve it.

7. What It Is Worth in Dollars
------------------------------

Converting that reduction into money needs two figures for your own product: its
unit cost, and an annual inventory holding-cost rate (typically 15-30% of unit
cost once capital, storage, insurance, and obsolescence are counted).

.. list-table::
   :header-rows: 1

   * - Mode
     - Stock reduction
     - Capital freed (@ $40/unit)
     - Ongoing saving (@ 25%/yr)
   * - Lost sales
     - ``337.7`` units
     - ``$13,507``
     - ``$3,377`` / year
   * - Backorder
     - ``251.7`` units
     - ``$10,069``
     - ``$2,517`` / year

These assumptions are illustrative, for a single product on this six-node
network -- substitute your own unit cost and holding rate. The figure that
carries over unchanged is the percentage reduction from section 6 (``-12.1%`` /
``-9.1%``), which depends on no dollar assumption. A real deployment runs many
products through the same rule at once, so the dollar total scales with however
many of them share this network.

8. Try It Yourself
------------------

.. code-block:: bash

   uv run -m examples.multi_echelon_inventory
   uv run -m examples.multi_echelon_inventory --mode backorder
   uv run -m examples.multi_echelon_inventory --published-solution

The source lives in ``examples/multi_echelon_inventory``; the SVGs on this page
are regenerated straight from live evaluations with
``python3 -m examples.multi_echelon_inventory.visualize``, so they never drift
from the code.

Why This Fits Inventory Planning, and Where to Take It Further
--------------------------------------------------------------

Multi-echelon inventory is exactly the kind of problem this approach targets: the
decision repeats every day, today's choice sets tomorrow's opening stock, and the
future -- demand and delivery delays -- is genuinely uncertain rather than
something a single forecast can capture. A tidy forecast hides that risk;
replaying many real historical futures does not. And because the tuned rule stays
simple and explainable, operations staff can audit and trust it directly rather
than take a black-box model's word for why it reordered when it did.

The same approach extends well past the version shown here:

* **More products at once.** The ten-numbers-per-location idea scales to a full
  catalog, each product carrying its own tuned reorder and base-stock levels.
* **Fresher data, on a rolling basis.** Rather than tuning once against a
  snapshot, re-run regularly against demand and delivery data pulled from the
  company's own systems, so the rule keeps adapting as patterns shift.
* **Richer rules, tested the same safe way.** Today's rule is deliberately
  simple. The same simulation can test a smarter rule -- one accounting for
  seasonality, promotions, or a demand forecast -- and compare whether the added
  complexity actually earns a better result before anyone commits to it.
* **A wider view of cost.** Transportation cost, supplier minimum order
  quantities, or multiple sourcing options can all be added to the objective
  without changing how the rest of the network works.
* **A standing digital twin.** Run alongside the live network, the model can
  keep rehearsing "what if we changed this number" in the background, so
  re-tuning inventory policy becomes routine and low-risk rather than a rare,
  high-stakes project.
