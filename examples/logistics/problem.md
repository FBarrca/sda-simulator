# Logistics Dispatch Problem

A road-freight operator runs three warehouses in Madrid, Barcelona, and
Valencia, serving twelve customer locations across mainland Spain. Each day,
orders arrive for four SKU families and the dispatcher chooses which pending
orders to send, from which warehouse, and on which vehicle.

The goal is to keep high-priority deliveries on time while controlling distance
cost, late-delivery penalties, backlog growth, and tail risk across many
sampled futures.

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

The example compares three dispatch rules:

- `NearestFeasiblePolicy`: FIFO dispatch from the nearest warehouse with stock
  and an available vehicle.
- `PriorityDeadlinePolicy`: high-priority and tight-deadline orders first,
  using nearest feasible service.
- `RiskAwareDispatchPolicy`: scores priority, deadline slack, travel distance,
  remaining stock, and vehicle fit.

Run the default risk-aware policy:

```bash
uv run -m examples.logistics
```

Compare all policies:

```bash
uv run --with matplotlib -m examples.logistics.policy_comparison --output examples/logistics/logistics_policy_comparison.png
```

![Logistics policy comparison](logistics_policy_comparison.png)

## Metrics

The simulation reports built-in `step_cost` and `total_cost` metrics plus
logistics-specific measures:

- `on_time_rate`
- `priority_weighted_on_time_rate`
- `late_cost`
- `dispatch_cost`
- `pending_backlog`
- `dispatched_order_count`
- `vehicle_utilization`

Use percentiles and `cvar(0.95)` on `total_cost` to inspect worst-case dispatch
performance.
