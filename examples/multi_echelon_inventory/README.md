# Multi-Echelon Inventory Optimization

This example reconstructs the simulation-optimization problem from
Agarwal, A. (2019), [Multi-echelon Supply Chain Inventory Planning using
Simulation-Optimization with Data Resampling](https://arxiv.org/abs/1901.00090)
(arXiv:1901.00090). The network, the reorder rule, the data, and the
objective are kept as-is; this README walks through what the resulting
tool does and why it is worth building, without assuming a technical
background. (See `docs/examples/multi_echelon_inventory.rst` for the full
walkthrough.)

## The Network And The Problem

A company ships one product from a single source through a small
distribution network to the locations that sell to customers: one source,
one regional stocking point, one intermediate transshipment point, and
three locations that sell directly to customers.

![Multi-echelon supply network](multi_echelon_network.svg)

Every lane between two locations has its own delivery lead time, and it is
not perfectly reliable -- it varies, based on real logged delivery history.
Holding too much stock ties up cash that could be doing something else for
the business; holding too little at a customer-facing location causes
stockouts. The task is to decide, for every stocking location, how much of
a buffer is actually worth carrying.

## What Happens Every Day

The same four things happen every simulated day at every stocking location:

1. **Check stock and decide whether to reorder** against a trigger level.
2. **The upstream location prepares that order**, drawing from its own stock.
3. **The order travels for its lead time**, drawn from real historical
   delivery records, delays included.
4. **Customers place demand**, served from whatever is on the shelf; unmet
   demand is either lost for good or filled late, depending on which
   business assumption is being tested.

## The Policy We're Tuning: One Simple Rule

Every stocking location runs the exact same rule: **when a location's
available stock drops to its trigger level, order enough to bring it back
up to its target level.** No forecast, no black-box model -- a rule an
operations team can read, trust, and audit. In decision-science terms this
is a **policy function approximation**.

The rule becomes specific to each location through two numbers: its
trigger level (reorder point) and target level (base stock) -- ten numbers
total across the five stocking locations, and the only thing being tuned
here.

![Reorder-point and base-stock hyperparameters](multi_echelon_policy_parameters.svg)

We tune these ten numbers for one reason: **to unlock capital sitting idle
as excess stock, without letting customer service drop.** Every unit held
above what a location truly needs is cash the business can't use elsewhere;
every unit missing is a customer left unserved.

## Testing Changes Safely, With Real History

Trying new numbers directly on the real supply chain is risky. Instead,
this example runs the network inside a computer model and replays real
historical demand and delivery data against it -- the same kind of order
and delivery-delay history a company already holds in its own systems, not
an invented average. A proposed set of ten numbers can be rehearsed against
real, messy conditions as many times as needed, in minutes, before anything
is decided for real.

![Daily inventory dynamics](multi_echelon_inventory_trace.svg)

The thick green line is one location's stock on hand; the dashed lines are
its own trigger and target levels. Stock drifts down as demand is served,
touches the trigger, a resupply order fires, and the line climbs back up
once that shipment's real lead time has passed.

## Did It Work?

Missing a service promise counts far more heavily than any inventory
saved, so a good set of ten numbers has to protect service first and only
then compete on how little stock it needs.

| Mode | Starting point | Tuned result | Change |
| --- | ---: | ---: | ---: |
| Lost sales | `2783.462` | `2445.776` | `-12.1%` |
| Backorder | `2767.635` | `2515.907` | `-9.1%` |

![Objective and service scorecard](multi_echelon_objective.svg)

The tuned numbers hold 9-12% less average stock and still clear every
service target -- nothing was traded away to get there.

## What This Is Worth In Dollars

Turning that reduction into money needs two extra numbers for your own
product: unit cost, and an annual holding-cost rate (typically 15-30%/year
of unit cost).

| Mode | Stock reduction | Capital freed (@ $40/unit) | Ongoing savings (@ 25%/yr) |
| --- | ---: | ---: | ---: |
| Lost sales | `337.7` units | `$13,507` | `$3,377` / year |
| Backorder | `251.7` units | `$10,069` | `$2,517` / year |

Those two assumptions are illustrative, for one product on this six-node
network -- substitute your own. The number that carries over unchanged is
the percentage reduction (`-12.1%` / `-9.1%`), and a real deployment runs
many products through the same rule at once, so the dollar total scales
with however many share this network.

## Try It Yourself

```bash
uv run -m examples.multi_echelon_inventory
uv run -m examples.multi_echelon_inventory --mode backorder
uv run -m examples.multi_echelon_inventory --published-solution
```

Regenerate the SVGs above from live evaluations with:

```bash
python3 -m examples.multi_echelon_inventory.visualize
```

The copied empirical CSV inputs and `REFERENCE_LICENSE` come from the
reference project and are included here so the example is self-contained.

## Why This Fits Inventory Planning, And Where To Take It Further

Inventory planning is exactly the kind of problem this style of
decision-making tool was built for: the decision repeats every single day,
today's choice changes tomorrow's starting stock, and the future is
genuinely uncertain rather than something a single forecast number can
capture. Because the rule being tuned stays simple and explainable,
operations staff can trust and audit it directly.

This same approach extends well past the version shown here:

- **More products at once** -- the same ten-numbers-per-location idea
  scales to a full catalog.
- **Fresher data, on a rolling basis** -- re-run regularly against demand
  and delivery data pulled directly from the company's own systems, so the
  rule keeps adapting as real patterns shift.
- **Richer rules, tested the same safe way** -- test a smarter rule (one
  that accounts for seasonality or a demand forecast) and simply compare
  whether the added complexity earns a better result before committing.
- **A wider view of cost** -- transportation cost, minimum order sizes, or
  multiple sourcing options can all be added without changing how the rest
  of the network works.
- **An ongoing habit, not a one-time project** -- run alongside the real
  network as a standing digital twin, quietly rehearsing changes so
  retuning inventory policy becomes routine instead of a rare, high-stakes
  project.
