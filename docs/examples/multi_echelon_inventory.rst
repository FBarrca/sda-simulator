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

1. **Check stock and decide whether to reorder.** If a location's available
   stock (what is on the shelf plus what is already on its way) has dropped
   to its trigger level, it places a replenishment order with the location
   upstream.
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

3. The Policy We're Tuning: One Simple Rule
------------------------------------------------

Every stocking location is run by the exact same rule, and it is
deliberately simple: **when a location's available stock drops to its
trigger level, order enough to bring it back up to its target level.**
That is the entire decision logic -- no forecast, no black-box model
guessing differently each time it runs. In decision-science terms this kind
of fixed, explainable rule is called a **policy function approximation**;
what matters for the business is that it is a rule an operations team can
read, trust, and audit, not a model whose reasoning has to be taken on
faith.

The rule becomes specific to each location through two numbers: its
**trigger level** (the reorder point) and its **target level** (the base
stock). Across the five stocking locations, that is ten numbers in total,
and those ten numbers are the *only* thing being tuned in this example.

.. image:: ../../examples/multi_echelon_inventory/multi_echelon_policy_parameters.svg
   :alt: Bar chart comparing initial-guess and published reorder-point and base-stock hyperparameters per node

We tune these ten numbers for one reason: **to unlock capital that is
sitting idle as excess stock on the shelf, without letting customer service
drop.** Every unit held above what a location truly needs is cash the
business cannot use anywhere else; every unit missing is a customer left
unserved. Getting these ten numbers right is the single lever available
here, and the chart above shows exactly how far each location's buffer
moved once that lever was pulled.

4. Testing Changes Safely, With Real History
--------------------------------------------------

Trying a new set of ten numbers directly on the real supply chain is
risky: a bad guess means real stockouts, or real cash tied up, and you
would not know it was wrong for weeks. Instead, this example runs the same
network inside a computer model and replays real historical demand and
delivery data against it -- the same kind of order and delivery-delay
history a company already holds in its own systems, not an invented average
or a textbook probability curve. That means a proposed set of ten numbers
can be rehearsed against real, messy conditions -- including the actual
delivery delays and demand swings that occurred historically -- as many
times as needed, in minutes, before anything is decided for real.

.. image:: ../../examples/multi_echelon_inventory/multi_echelon_inventory_trace.svg
   :alt: Daily inventory trace for the network total and node 2, with node 2's reorder point and base stock drawn as reference lines

The thick green line above is one location's stock on hand over time; the
dashed lines are its own trigger and target levels. You can watch the rule
at work day by day: stock drifts down as demand is served, touches the
trigger, a resupply order fires, and the line climbs back up once that
shipment's real lead time has passed.

5. Did It Work?
------------------

Two outcomes matter here, and only one of them can be traded off against
the other, in fact, not in theory: how much stock the network holds, and
whether every customer-facing location keeps its promised fill rate.
Missing that promise counts far more heavily than any inventory saved, so a
good set of ten numbers has to protect service first, and only then
compete on how little stock it needs to do it.

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

6. What This Is Worth In Dollars
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
section 5 (``-12.1%`` / ``-9.1%``), since it does not depend on any dollar
assumption, and a real deployment runs many products through the same rule
at once, so the dollar total scales with however many of them share this
network.

7. Try It Yourself
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
