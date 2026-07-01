Multi-Echelon Inventory Optimization Walkthrough
=================================================

This walkthrough reconstructs the simulation-optimization problem from
Agarwal, A. (2019), `Multi-echelon Supply Chain Inventory Planning using
Simulation-Optimization with Data Resampling
<https://arxiv.org/abs/1901.00090>`_ (arXiv:1901.00090). The network, the
reorder rule, the data, and the objective are kept as-is; this page walks
through what the resulting tool actually does and why it is worth building,
without assuming a technical background.

1. The Network And The Problem
--------------------------------

A company ships one product from a single source through a small
distribution network to the locations that sell to customers. The network
has one source (a plant or vendor assumed to never run out) feeding five
downstream stocking locations: one regional stocking point, one
intermediate transshipment point, and three locations that sell directly to
customers.

.. image:: ../../examples/multi_echelon_inventory/multi_echelon_network.svg
   :alt: Six-node multi-echelon inventory network with lead times labeled on each lane

Every one of those lanes also has its own delivery lead time, and that lead
time is not perfectly reliable -- it varies, based on real logged delivery
history. Holding too much stock anywhere ties up cash that could be doing
something else for the business; holding too little at a customer-facing
location causes stockouts. A location that sits farther from the source, or
behind a slower and less predictable lane, has to carry more of a buffer to
protect the same level of service as one closer to the source. The task is
to decide, for every stocking location, how much of a buffer is actually
worth carrying.

2. What Happens Every Day
----------------------------

Underneath the network, the same four things happen every simulated day at
every stocking location:

1. **Check inventory position and decide whether to reorder.** If a
   location's inventory position (what is on the shelf plus what is already
   on its way) has dropped to its trigger level, it places a replenishment
   order with the location upstream.
2. **The upstream location prepares that order.** It sets aside the
   quantity requested, drawing from its own stock.
3. **The order travels for its lead time.** Delivery is not instant, and
   the number of days it takes is drawn from real historical delivery
   records, delays included.
4. **Customers place demand.** At every customer-facing location, real
   historical order volumes are served from whatever is on the shelf that
   day. If there is not enough, the order is either lost for good, or
   filled late once stock arrives, depending on which of the two business
   assumptions is being tested.

None of this is guesswork layered on top of a spreadsheet: it is the same
sequence of events that would happen in the real network, replayed with
real historical data, one simulated day at a time.

3. The Policy We're Tuning: A PFA
-------------------------------------

Every stocking location is controlled by the same deliberately small
decision rule. In sequential decision analytics this is a **policy function
approximation**, or **PFA**: a readable function that maps the state we can
observe today to the action we take today. Here the state is the facility's
inventory position and on-hand stock; the action is how much to order from
the upstream facility.

The PFA has two business-readable hyperparameters for each stocking node:

* **Trigger level** / **reorder point** (``R``): when inventory position is
  low enough, start a replenishment order.
* **Target level** / **base stock** (``B``): the level the facility is
  trying to refill toward.

The policy function is intentionally this simple:

.. code-block:: python

   def order_quantity(on_hand, inventory_position, trigger, target, *, tolerance=1.05):
       if inventory_position <= tolerance * trigger:
           return target - on_hand
       return 0.0

The trigger test uses **inventory position**, not just physical stock on the
shelf. Inventory position includes stock that is already on its way, so the
policy does not panic-order simply because a shipment is in transit. The
``1.05`` tolerance is carried over from the reference implementation to make
the reorder comparison robust around the trigger. Conceptually, though, the
two numbers the business owns are still the trigger ``R`` and target ``B``.

Across the five stocking locations, the policy therefore has ten
hyperparameters: ``R1`` through ``R5`` and ``B1`` through ``B5``. The source
node is treated as unconstrained, so it is not part of the tuning problem.

4. Releasing Capital: Optimize The PFA Levels
------------------------------------------------

Once the policy is explicit, the improvement question becomes concrete:
**can we release working capital from inventory without worsening the
customer experience?** In this example, that means optimizing the ten PFA
hyperparameters across the whole network and across all simulated
historical replications.

The optimizer does not tune each location in isolation. A lower target at
the regional node can starve customer nodes downstream; a higher trigger at
a customer node can protect fill rate but tie up more local stock. The
right answer is the joint set of levels that minimizes the reference
objective:

.. code-block:: text

   objective = average on-hand inventory
             + 1.0e6 * total service-level shortfall

The large penalty makes the intent plain: first protect service, then drive
inventory down. The reference optimizer vector stores each node as
``excess above trigger`` plus ``trigger``:

.. code-block:: text

   x = [e1, e2, e3, e4, e5, R1, R2, R3, R4, R5]
   B_i = e_i + R_i

So the search is still choosing the two levels we care about: trigger
``R_i`` and target ``B_i`` for every stocking node.

.. image:: ../../examples/multi_echelon_inventory/multi_echelon_policy_parameters.svg
   :alt: Bar chart comparing initial-guess and optimized reorder-point and base-stock hyperparameters per node

The default lost-sales optimized policy does not simply cut every buffer.
It mostly releases stock from the regional stocking point while keeping, or
slightly increasing, selected downstream protection:

.. list-table::
   :header-rows: 1

   * - Node
     - Trigger change
     - Target change
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

That is the business story the chart is meant to make visible: optimization
is not "hold less everywhere." It is "put the buffer where the network
actually needs it, and release the rest."

5. Simulating The Policy On Historical Data
------------------------------------------------

Trying a new set of levels directly on the real supply chain is risky: a
bad guess means real stockouts, or real cash tied up, and you would not
know it was wrong for weeks. Instead, this example runs the same network
inside a computer model and replays real historical demand and delivery
data against it. The demand values and lead-time delays come from the CSV
history bundled with the reference problem; each seeded replication
bootstraps from those histories to create another plausible year.

.. image:: ../../examples/multi_echelon_inventory/multi_echelon_inventory_trace.svg
   :alt: Daily policy traces for nodes 1 through 5 with a side topology map showing where each node sits in the network

The trace now follows all five stocking nodes. Each row uses its own
y-scale because the regional buffer is much larger than the customer-facing
buffers; putting them on one shared axis would hide the downstream
behavior. Within each row, the green line is **inventory position**, the
signal the policy watches, and the blue line is **on-hand inventory**, the
stock customers or downstream nodes can actually draw from today. The
topology inset on the right maps those node numbers back to the supply
network, so the reader can see which rows are upstream buffers and which
rows are customer-facing locations.

The dashed lines show that node's trigger and target levels, and the
triangles mark inferred reorder days. Reading across rows shows how the
same PFA behaves differently by echelon: upstream nodes carry larger
buffers and replenish downstream facilities, while customer-facing nodes
cycle around smaller local targets. That is more useful than a simple
"current inventory" plot because it shows both sides of the tradeoff:
whether each node orders early enough to protect service, and whether it is
holding more stock than the tuned PFA actually needs.

6. Did It Work?
------------------

Two outcomes matter here: how much stock the network holds, and whether
every customer-facing location keeps its promised fill rate. Missing that
promise counts far more heavily than any inventory saved, so a good set of
ten numbers has to protect service first, and only then compete on how
little stock it needs to do it.

.. list-table::
   :header-rows: 1

   * - Mode
     - Starting point
     - Tuned result
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

The tuned numbers hold 9-12% less average stock across the network and
still clear every service target -- nothing was traded away to get there.

7. What This Is Worth In Dollars
------------------------------------

Turning that reduction into money only needs two extra numbers for your own
product: its unit cost, and an annual inventory holding-cost rate
(typically 15-30% of unit cost per year once capital, storage, insurance,
and obsolescence are all counted).

.. list-table::
   :header-rows: 1

   * - Mode
     - Stock reduction
     - Capital freed (@ $40/unit)
     - Ongoing savings (@ 25%/yr)
   * - Lost sales
     - ``337.7`` units
     - ``$13,507``
     - ``$3,377`` / year
   * - Backorder
     - ``251.7`` units
     - ``$10,069``
     - ``$2,517`` / year

Those two assumptions are illustrative, for a single product on this
six-node network -- substitute your own unit cost and holding-cost rate.
The number that carries over unchanged is the percentage reduction from
section 6 (``-12.1%`` / ``-9.1%``), since it does not depend on any dollar
assumption, and a real deployment runs many products through the same rule
at once, so the dollar total scales with however many of them share this
network.

8. Try It Yourself
----------------------

.. code-block:: bash

   uv run -m examples.multi_echelon_inventory
   uv run -m examples.multi_echelon_inventory --mode backorder
   uv run -m examples.multi_echelon_inventory --published-solution

The source lives in ``examples/multi_echelon_inventory``; the SVGs on this
page are regenerated straight from live evaluations with
``python3 -m examples.multi_echelon_inventory.visualize``, so they never
drift from the code.

Why This Fits Inventory Planning, And Where To Take It Further
------------------------------------------------------------------

Inventory planning is exactly the kind of problem this style of
decision-making tool was built for: the decision repeats every single day,
today's choice changes tomorrow's starting stock, and the future -- demand,
delivery delays -- is genuinely uncertain rather than something a single
forecast number can capture. A tidy forecast hides that risk; replaying
many real historical futures does not. Because the rule being tuned stays
simple and explainable, operations staff can trust and audit it directly,
rather than take a black-box model's word for why it reordered when it did.

This same approach extends well past the version shown here:

* **More products at once.** The same ten-numbers-per-location idea scales
  to a full catalog, each product carrying its own tuned trigger and target
  levels.
* **Fresher data, on a rolling basis.** Instead of tuning once against a
  historical snapshot, the same setup can be re-run regularly against fresh
  demand and delivery data pulled directly from the company's own systems,
  so the rule keeps adapting as real patterns shift.
* **Richer rules, tested the same safe way.** Today's rule is deliberately
  simple. The same simulation can just as easily test a smarter rule --
  one that accounts for seasonality, promotions, or a demand forecast --
  and simply compare whether the added complexity actually earns a better
  result before anyone commits to it.
* **A wider view of cost.** Transportation cost, supplier minimum order
  sizes, or multiple sourcing options can all be added to what the search
  optimizes for, without changing how the rest of the network works.
* **An ongoing habit, not a one-time project.** Run alongside the real
  network as a standing digital twin, it can keep quietly rehearsing "what
  if we changed this number" in the background, so retuning inventory
  policy becomes routine and low-risk instead of a rare, high-stakes
  project.
