# Logistics Dispatch Problem

A road-freight operator runs three warehouses in Madrid, Barcelona, and
Valencia, serving twelve customer locations across mainland Spain. Each day,
orders arrive for four SKU families and the dispatcher chooses which pending
orders to send, from which warehouse, and on which vehicle.

## Business Goal

From a business standpoint, the dispatcher is not trying to simply move the most
orders or minimize today's kilometers. The goal is to protect customer service
and margin under uncertain demand, traffic, and vehicle availability.

A good dispatch policy should:

- deliver high-priority orders on time,
- avoid letting backlog grow into an unserviceable queue,
- control distance, handling, late-delivery, and bad-week tail costs,
- use fleet capacity sensibly without dispatching low-value work just to keep
  trucks busy,
- remain robust across many plausible futures, not only one average day.

This means the business goal has tradeoffs. A policy that dispatches every easy
order can still fail if urgent orders miss their deadlines. A policy that keeps
late cost low by dispatching almost nothing can also fail if backlog growth is
unacceptable. The report therefore reads every policy through both cost and
operational metrics.

## Modeled Objective

The modeled objective is to minimize expected total cost across sampled futures.
Each day's cost has four parts:

- dispatch cost: distance cost plus per-unit handling cost,
- late cost: deadline-miss penalties that scale with late days, quantity, and
  priority,
- overdue backlog cost: a smaller daily penalty for pending orders already past
  deadline,
- invalid assignment cost: penalties for infeasible dispatches.

The model tracks service, backlog, dispatch volume, fleet utilization, and tail
risk, but it does not impose a hard minimum dispatch volume, minimum utilization,
or maximum backlog constraint. Those are operational metrics, not hard
requirements, unless the cost model is tuned to make them binding. This matters
for interpreting the results: a policy can minimize modeled cost by being very
selective, even if the resulting backlog would be unacceptable in a real
operation.

## Network

The network uses a 3 x 12 warehouse-to-customer lane matrix over an
OpenStreetMap basemap. Madrid, Barcelona, and Valencia each have nearby demand,
but every customer can be served from any warehouse if the nearest site lacks
stock or an available truck.

![Spanish logistics network](logistics_network.png)

Regenerate the network map with:

```bash
uv run --with matplotlib -m examples.logistics.visualize_network --output examples/logistics/logistics_network.png
```

If `uv` is unavailable but the environment has dependencies installed:

```bash
python3 -m examples.logistics.visualize_network --output examples/logistics/logistics_network.png
```

## Synthetic Demand

`synthetic_history(days, seed)` creates deterministic order history, traffic
multipliers, vehicle outages, and event labels. The demand model includes:

- weekday and weekend demand rhythm,
- annual seasonality with a year-end peak,
- promotions and holiday peaks that lift order volume,
- severe weather and port congestion that increase travel time and outages,
- SKU mix across `AMBIENT_FOOD`, `COLD_CHAIN`, `ELECTRONICS`, and `PHARMA`.

![Synthetic logistics demand](logistics_synthetic_demand.png)

Regenerate the demand plot with:

```bash
uv run --with matplotlib -m examples.logistics.visualize_demand --seed 7 --days 365 --output examples/logistics/logistics_synthetic_demand.png
```

## Policies

The example compares eight dispatch rules:

- `RandomPolicy`: randomizes feasible assignments and greedily keeps a
  conflict-free set.
- `GreedyPolicy`: prefers the shortest warehouse-to-customer lane first.
- `PriorityPolicy`: scores priority, quantity, deadline pressure, rescue
  pressure, duration, and travel distance.
- `MilpPolicy`: maximizes the same priority-distance score globally with
  one-order, one-vehicle, and inventory constraints when SciPy is available,
  falling back to `PriorityPolicy` otherwise.
- `LookaheadRolloutPolicy`: compares priority, greedy, and defer-first
  decisions over sampled futures, then commits to the best first-day decision.
- `NearestFeasiblePolicy`: FIFO dispatch from the nearest warehouse with stock
  and an available vehicle.
- `PriorityDeadlinePolicy`: high-priority and tight-deadline orders first,
  using nearest feasible service.
- `RiskAwareDispatchPolicy`: scores priority, deadline slack, lane distance,
  stock scarcity, and vehicle fit.

Run the default priority policy:

```bash
uv run -m examples.logistics
```

Compare all policies:

```bash
uv run -m examples.logistics.policy_comparison --horizon 28 --n-scenarios 500 --batch-size 64 --seed 42 --no-plot
```

## Interpreting Results

The policy comparison command prints a text table:

```bash
uv run -m examples.logistics.policy_comparison --horizon 28 --n-scenarios 500 --batch-size 64 --seed 42 --no-plot
```

```text
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
```

Read the columns this way:

- `total_mean` (lower is better): average total cost over all bootstrap futures.
- `total_ci95`: approximate 95% confidence interval for mean total cost.
  Policies with heavily overlapping intervals are not clearly separated.
- `cost_cvar95` (lower is better): average total cost in the worst 5% of
  scenarios; this is the downside-risk number.
- `prio_ot` (higher is better): priority-weighted on-time service.
- `late_cost` (lower is better): average late-delivery penalty per
  scenario-day.
- `backlog` (lower is better): average pending orders waiting for dispatch.
- `dispatch/day`: average dispatched orders per day. This is throughput, not
  value.
- `util`: average fraction of vehicles dispatched each day.

Clear conclusions from this run:

- Best objective value: `LookaheadRolloutPolicy`. It has the lowest mean cost
  (106,336), lowest tail cost (CVaR 169,106), lowest late cost (117), and best
  priority-weighted on-time service (95.8%). Under the current cost model, this
  is the winning policy.
- Important caveat: rollout wins by being very selective. It dispatches only
  0.5 orders per day and uses only 8.5% of the fleet, while backlog rises to
  123.9 orders. That is good for the current cost objective, but it may be
  operationally unacceptable if clearing backlog or using capacity is a hard
  business requirement.
- Best simple high-throughput policy: `GreedyPolicy`. It dispatches the most
  orders (5.5 per day), uses the fleet heavily (91.6%), and is the best
  non-rollout policy by mean cost. It is still much worse than rollout on late
  cost, tail risk, and priority service.
- Middle tier: `PriorityDeadlinePolicy` and `RiskAwareDispatchPolicy`. They are
  almost tied. Risk-aware has slightly better tail cost, while priority-deadline
  has slightly better mean cost. Both are better than random but worse than
  greedy and rollout.
- Do not use `NearestFeasiblePolicy` for this objective. FIFO nearest-feasible
  dispatch performs worst by mean cost and tail cost because it ignores which
  orders are expensive to miss.
- `PriorityPolicy` and `MilpPolicy` need recalibration. They tie because the
  MILP optimizes the same one-day score as `PriorityPolicy`; in this scenario
  that score is misaligned with realized cost, so global optimization does not
  help.

The business decision is therefore conditional. If the objective is exactly the
simulated cost function, choose `LookaheadRolloutPolicy`. If the operation must
also clear backlog or keep trucks utilized, do not deploy that rollout
unchanged; increase the backlog/deferment penalty or add minimum-dispatch or
service constraints, then rerun the comparison. If a cheap rule is needed
today, `GreedyPolicy` is the strongest simple policy in this table, while
`PriorityDeadlinePolicy` and `RiskAwareDispatchPolicy` are safer deadline-aware
candidates after tuning.

## Metrics

The SimPy model records event-level `cost` and trajectory-level `total_cost`
metrics plus logistics-specific measures:

- `on_time_rate`
- `priority_weighted_on_time_rate`
- `late_cost`
- `dispatch_cost`
- `pending_backlog`
- `dispatched_order_count`
- `vehicle_utilization`

Use percentiles and `cvar(0.95)` on `total_cost` to inspect worst-case dispatch
performance.
