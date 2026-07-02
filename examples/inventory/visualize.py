"""Generate the figures used in the inventory walkthrough.

The plots are built from live evaluations of the same configuration wired up in
``examples/inventory/main.py``, so they never drift from the code. Matplotlib is
an optional dependency; run this module with, for example::

    uv run --with matplotlib -m examples.inventory.visualize

Each figure is written as a PNG into ``examples/inventory`` and referenced from
``docs/examples/inventory.rst``.
"""

from __future__ import annotations

import argparse
from math import sqrt
from pathlib import Path

import numpy as np

from examples.inventory.data import InventoryDataModule
from examples.inventory.models import InventoryModel
from examples.inventory.policies import DemandScaledOrderUpToPolicy, OrderUpToPolicy
from sda import Policy, evaluate

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent

# Shared scenario configuration, matching examples/inventory/main.py.
HORIZON = 12
N_SCENARIOS = 1000
BATCH_SIZE = 128
INITIAL_INVENTORY = 50.0
DEMAND_LAMBDA = 20.0
REORDER_POINT = 30.0
ORDER_UP_TO = 80.0
SEED = 42

ORDER_COST = 1.0
HOLDING_COST = 0.1
STOCKOUT_COST = 8.0

# Palette shared with the logistics figures.
DEMAND_COLOR = "#3a86ff"
ORDER_COLOR = "#2a9d8f"
WARN_COLOR = "#e76f51"
ACCENT_COLOR = "#ff9f1c"
GRID_COLOR = "#e5e5e5"


def _require_pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on optional dep
        raise ImportError(
            "matplotlib is required for the inventory figures. "
            "Run with `uv run --with matplotlib -m examples.inventory.visualize`."
        ) from exc
    return plt


def _data_module(
    *,
    n_scenarios: int = N_SCENARIOS,
    batch_size: int = BATCH_SIZE,
    seed: int = SEED,
) -> InventoryDataModule:
    return InventoryDataModule(
        horizon=HORIZON,
        n_scenarios=n_scenarios,
        batch_size=batch_size,
        initial_inventory=INITIAL_INVENTORY,
        demand_lambda=DEMAND_LAMBDA,
        seed=seed,
    )


def _model(*, reorder_point: float = REORDER_POINT, order_up_to: float = ORDER_UP_TO) -> InventoryModel:
    return InventoryModel(
        policy=OrderUpToPolicy(reorder_point=reorder_point, order_up_to=order_up_to),
        order_cost=ORDER_COST,
        holding_cost=HOLDING_COST,
        stockout_cost=STOCKOUT_COST,
    )


def _evaluate(
    *,
    reorder_point: float = REORDER_POINT,
    order_up_to: float = ORDER_UP_TO,
    n_scenarios: int = N_SCENARIOS,
    seed: int = SEED,
):
    data = _data_module(n_scenarios=n_scenarios, seed=seed)
    model = _model(reorder_point=reorder_point, order_up_to=order_up_to)
    return evaluate(model, data)


def _all_demand(data: InventoryDataModule) -> np.ndarray:
    """Collect every simulated daily-demand value across all scenarios."""
    demand = [
        np.asarray(spec.data["demand"], dtype=float)
        for batch in data.batches()
        for spec in batch.scenarios
    ]
    return np.concatenate(demand)


def save_demand_plot(
    *,
    output: str | Path = DEFAULT_OUTPUT_DIR / "inventory_demand.png",
) -> Path:
    """Histogram of Poisson daily demand against the policy levels."""
    plt = _require_pyplot()
    demand = _all_demand(_data_module())

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 6))
    bins = np.arange(demand.min(), demand.max() + 2) - 0.5
    ax.hist(demand, bins=bins, color=DEMAND_COLOR, alpha=0.85, edgecolor="white", linewidth=0.4)
    ax.set_title(
        f"Daily demand distribution (Poisson λ={DEMAND_LAMBDA:.0f}, {HORIZON}-day × {N_SCENARIOS} scenarios)",
        fontsize=15,
        fontweight="bold",
    )
    ax.set_xlabel("Units demanded in a day")
    ax.set_ylabel("Number of scenario-days")
    ax.grid(True, axis="y", color=GRID_COLOR, linewidth=0.8)

    for level, color, label in (
        (INITIAL_INVENTORY, ACCENT_COLOR, f"Initial inventory = {INITIAL_INVENTORY:.0f}"),
        (REORDER_POINT, ORDER_COLOR, f"Reorder point = {REORDER_POINT:.0f}"),
        (ORDER_UP_TO, WARN_COLOR, f"Order-up-to = {ORDER_UP_TO:.0f}"),
    ):
        ax.axvline(level, color=color, linestyle="--", linewidth=1.8, label=label)
    ax.axvline(float(demand.mean()), color="#0f172a", linestyle=":", linewidth=1.8,
               label=f"Mean demand ≈ {demand.mean():.1f}")
    ax.legend(frameon=False, fontsize=9, loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def save_trace_plot(
    *,
    output: str | Path = DEFAULT_OUTPUT_DIR / "inventory_trace.png",
) -> Path:
    """Single-scenario on-hand inventory sawtooth with reorder markers."""
    plt = _require_pyplot()
    result = _evaluate()

    _, _, inventory = result["inventory"].to_trajectory_matrix()
    _, _, orders = result["order_quantity"].to_trajectory_matrix()
    xs, ys, order_x, order_y = _reconstruct_sawtooth(inventory[0], orders[0])
    horizon = len(inventory[0])

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(xs, ys, color=DEMAND_COLOR, linewidth=2.2, label="On-hand inventory")
    ax.axhline(ORDER_UP_TO, color=WARN_COLOR, linestyle="--", linewidth=1.6,
               label=f"Order-up-to (target) = {ORDER_UP_TO:.0f}")
    ax.axhline(REORDER_POINT, color=ORDER_COLOR, linestyle="--", linewidth=1.6,
               label=f"Reorder point = {REORDER_POINT:.0f}")
    if order_x:
        ax.scatter(order_x, order_y, color="#0f172a", marker="^", s=90, zorder=5,
                   label="Order placed, refilled to target")

    ax.set_title(
        f"Order-up-to policy on one scenario (R={REORDER_POINT:.0f}, S={ORDER_UP_TO:.0f})",
        fontsize=15,
        fontweight="bold",
    )
    ax.set_xlabel("Day")
    ax.set_ylabel("Units on hand")
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0, right=horizon)
    ax.grid(True, axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.legend(frameon=False, fontsize=9, loc="lower right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _reconstruct_sawtooth(end_of_day, order_qty):
    """Rebuild the within-day inventory path from logged end-of-day levels.

    The recorder stores end-of-day inventory and the order placed that day. Each
    day opens at the previous day's ending stock; an order (if any) lifts it to
    the target at the start of the day, then demand draws it down. Returns the
    plot points plus the (x, y) of each reorder jump.
    """
    xs = [0.0]
    ys = [INITIAL_INVENTORY]
    order_x: list[float] = []
    order_y: list[float] = []
    prev_end = INITIAL_INVENTORY
    for day in range(len(end_of_day)):
        if order_qty[day] > 0:
            after_order = prev_end + float(order_qty[day])  # equals the target on reorder days
            xs.append(float(day))
            ys.append(after_order)
            order_x.append(float(day))
            order_y.append(after_order)
        xs.append(float(day + 1))
        ys.append(float(end_of_day[day]))
        prev_end = float(end_of_day[day])
    return xs, ys, order_x, order_y


def _policy_sawtooth(policy: Policy):
    """Evaluate one policy and return the first scenario's sawtooth path."""
    data = _data_module()
    model = InventoryModel(
        policy=policy,
        order_cost=ORDER_COST,
        holding_cost=HOLDING_COST,
        stockout_cost=STOCKOUT_COST,
    )
    result = evaluate(model, data)
    _, _, inventory = result["inventory"].to_trajectory_matrix()
    _, _, orders = result["order_quantity"].to_trajectory_matrix()
    return _reconstruct_sawtooth(inventory[0], orders[0])


def save_trace_comparison_plot(
    *,
    output: str | Path = DEFAULT_OUTPUT_DIR / "inventory_trace_comparison.png",
) -> Path:
    """Overlay the sawtooth of an over-provisioned vs a right-sized policy."""
    plt = _require_pyplot()

    over = OrderUpToPolicy(reorder_point=REORDER_POINT, order_up_to=ORDER_UP_TO)
    right = DemandScaledOrderUpToPolicy(
        demand_mean=DEMAND_LAMBDA,
        demand_std=sqrt(DEMAND_LAMBDA),
        stockout_cost=STOCKOUT_COST,
        holding_cost=HOLDING_COST,
    )
    over_xs, over_ys, _, _ = _policy_sawtooth(over)
    right_xs, right_ys, _, _ = _policy_sawtooth(right)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(over_xs, over_ys, color=DEMAND_COLOR, linewidth=2.2,
            label=f"Order-up-to (s, S), S={ORDER_UP_TO:.0f}  — over-provisioned")
    ax.plot(right_xs, right_ys, color=ORDER_COLOR, linewidth=2.2,
            label=f"Demand-scaled, S≈{right.order_up_to:.0f}  — right-sized")
    ax.axhline(ORDER_UP_TO, color=DEMAND_COLOR, linestyle=":", linewidth=1.2, alpha=0.6)
    ax.axhline(right.order_up_to, color=ORDER_COLOR, linestyle=":", linewidth=1.2, alpha=0.6)

    ax.set_title(
        "Same demand, two buffers: over-provisioned vs right-sized",
        fontsize=15,
        fontweight="bold",
    )
    ax.set_xlabel("Day")
    ax.set_ylabel("Units on hand")
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0, right=HORIZON)
    ax.grid(True, axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.legend(frameon=False, fontsize=10, loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def save_cost_distribution_plot(
    *,
    output: str | Path = DEFAULT_OUTPUT_DIR / "inventory_cost_distribution.png",
) -> Path:
    """Histogram of total cost across all scenarios with tail markers."""
    plt = _require_pyplot()
    result = _evaluate()

    total_cost = result["total_cost"]
    values = total_cost.values()
    mean = float(total_cost.mean())
    p95 = float(total_cost.percentile(95))
    cvar95 = float(total_cost.cvar(0.95))

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.hist(values, bins=40, color=DEMAND_COLOR, alpha=0.85, edgecolor="white", linewidth=0.4)
    ax.set_title(
        f"Total-cost distribution across {values.size} demand futures",
        fontsize=15,
        fontweight="bold",
    )
    ax.set_xlabel("Total cost over the horizon")
    ax.set_ylabel("Number of scenarios")
    ax.grid(True, axis="y", color=GRID_COLOR, linewidth=0.8)

    for level, color, label in (
        (mean, "#0f172a", f"Mean = {mean:.1f}"),
        (p95, ACCENT_COLOR, f"p95 = {p95:.1f}"),
        (cvar95, WARN_COLOR, f"CVaR 95 = {cvar95:.1f}"),
    ):
        ax.axvline(level, color=color, linestyle="--", linewidth=1.8, label=label)
    ax.legend(frameon=False, fontsize=10, loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def save_policy_sweep_plot(
    *,
    reorder_points: np.ndarray | None = None,
    buffer: float = ORDER_UP_TO - REORDER_POINT,
    n_scenarios: int = 400,
    output: str | Path = DEFAULT_OUTPUT_DIR / "inventory_policy_sweep.png",
) -> Path:
    """Sweep both policy levels down together and plot cost against fill rate.

    The reorder point ``R`` is swept while the order quantity stays fixed at
    ``buffer`` (so the target ``S = R + buffer`` moves with it, matching the
    default ``80 - 30`` gap). Lowering both levels frees holding cost until the
    thinning buffer starts to lose sales -- the crossover the walkthrough
    describes.
    """
    plt = _require_pyplot()

    if reorder_points is None:
        reorder_points = np.arange(0.0, 61.0, 5.0)

    mean_costs, fill_rates, _ = _sweep_reorder_points(reorder_points, buffer, n_scenarios)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, cost_ax = plt.subplots(figsize=(11, 6))
    fill_ax = cost_ax.twinx()

    cost_line = cost_ax.plot(reorder_points, mean_costs, color=DEMAND_COLOR, linewidth=2.4,
                             marker="o", markersize=5, label="Mean total cost")
    fill_line = fill_ax.plot(reorder_points, fill_rates, color=WARN_COLOR, linewidth=2.4,
                             marker="s", markersize=5, label="Mean fill rate")

    cost_ax.set_title(
        f"Tuning the buffer (target S = R + {buffer:.0f}, {n_scenarios} scenarios)",
        fontsize=15,
        fontweight="bold",
    )
    cost_ax.set_xlabel("Reorder point R  (target S moves with it)")
    cost_ax.set_ylabel("Mean total cost", color=DEMAND_COLOR)
    cost_ax.tick_params(axis="y", labelcolor=DEMAND_COLOR)
    cost_ax.grid(True, axis="y", color=GRID_COLOR, linewidth=0.8)
    fill_ax.set_ylabel("Mean fill rate", color=WARN_COLOR)
    fill_ax.tick_params(axis="y", labelcolor=WARN_COLOR)
    fill_ax.set_ylim(min(fill_rates) - 0.02, 1.01)

    # Mark the cost-minimising reorder point.
    best = int(np.argmin(mean_costs))
    knee = float(reorder_points[best])
    cost_ax.axvline(knee, color="#0f172a", linestyle=":", linewidth=1.6)
    cost_ax.annotate(
        f"lowest mean cost at R={knee:.0f}\n(fill rate {fill_rates[best]:.3f})",
        xy=(knee, mean_costs[best]),
        xytext=(10, 18),
        textcoords="offset points",
        fontsize=9,
        color="#0f172a",
    )

    lines = cost_line + fill_line
    cost_ax.legend(lines, [line.get_label() for line in lines], frameon=False, fontsize=10,
                   loc="upper center")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _sweep_reorder_points(reorder_points, buffer: float, n_scenarios: int):
    """Evaluate a range of reorder points (target following at ``buffer`` above).

    Returns arrays of mean total cost, mean fill rate, and mean on-hand inventory
    -- shared by the sweep and the exchange-curve figures.
    """
    costs: list[float] = []
    fills: list[float] = []
    inventory: list[float] = []
    for reorder_point in reorder_points:
        result = _evaluate(
            reorder_point=float(reorder_point),
            order_up_to=float(reorder_point) + buffer,
            n_scenarios=n_scenarios,
        )
        costs.append(float(result["total_cost"].mean()))
        fills.append(float(result["fill_rate"].mean()))
        inventory.append(float(result["inventory"].mean()))
    return np.asarray(costs), np.asarray(fills), np.asarray(inventory)


def _sweep_base_stock(levels, n_scenarios: int):
    """Evaluate base-stock levels (order-up-to every period) -- the efficient
    single-item family. Returns mean fill rate and mean on-hand inventory."""
    fills: list[float] = []
    inventory: list[float] = []
    for level in levels:
        # reorder point == target makes the order-up-to rule a daily base stock.
        result = _evaluate(
            reorder_point=float(level),
            order_up_to=float(level),
            n_scenarios=n_scenarios,
        )
        fills.append(float(result["fill_rate"].mean()))
        inventory.append(float(result["inventory"].mean()))
    return np.asarray(fills), np.asarray(inventory)


def save_service_frontier_plot(
    *,
    base_stock_levels: np.ndarray | None = None,
    n_scenarios: int = 400,
    output: str | Path = DEFAULT_OUTPUT_DIR / "inventory_service_frontier.png",
) -> Path:
    """Inventory-service exchange curve: stock needed for each service level.

    Built by sweeping the base-stock level ``S`` (the efficient single-item
    family), so the curve traces the *minimum* inventory that achieves each fill
    rate -- consistent with the demand-scaled and optimized policies elsewhere on
    this page.
    """
    plt = _require_pyplot()
    from matplotlib.ticker import PercentFormatter

    if base_stock_levels is None:
        base_stock_levels = np.arange(12.0, 61.0, 3.0)
    fills, inventory = _sweep_base_stock(base_stock_levels, n_scenarios)

    order = np.argsort(fills)
    fills_sorted = fills[order]
    inventory_sorted = inventory[order]

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(fills_sorted, inventory_sorted, color=DEMAND_COLOR, linewidth=2.4,
            marker="o", markersize=5)

    # Efficient operating point: least stock that still achieves ~full service.
    full = np.where(fills >= 0.995)[0]
    if full.size:
        knee = full[np.argmin(inventory[full])]
        ax.scatter([fills[knee]], [inventory[knee]], color=ORDER_COLOR, s=150, zorder=5,
                   edgecolor="white", linewidth=1.4,
                   label=f"Efficient point: {inventory[knee]:.0f} units at {fills[knee]:.1%} fill")
        ax.annotate(
            "least stock for ~100% service;\nabove here, more stock buys no service",
            xy=(fills[knee], inventory[knee]),
            xytext=(-16, 40),
            textcoords="offset points",
            fontsize=9,
            ha="right",
            color="#0f172a",
        )

    ax.set_title(
        f"Inventory-service exchange curve ({n_scenarios} scenarios)",
        fontsize=15,
        fontweight="bold",
    )
    ax.set_xlabel("Achieved fill rate")
    ax.set_ylabel("Average inventory (working capital, units)")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.legend(frameon=False, fontsize=10, loc="upper left")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def save_all_visualizations(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> list[Path]:
    """Regenerate every inventory figure into ``output_dir``."""
    output_dir = Path(output_dir)
    return [
        save_demand_plot(output=output_dir / "inventory_demand.png"),
        save_trace_plot(output=output_dir / "inventory_trace.png"),
        save_trace_comparison_plot(output=output_dir / "inventory_trace_comparison.png"),
        save_cost_distribution_plot(output=output_dir / "inventory_cost_distribution.png"),
        save_policy_sweep_plot(output=output_dir / "inventory_policy_sweep.png"),
        save_service_frontier_plot(output=output_dir / "inventory_service_frontier.png"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the inventory walkthrough figures.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="directory to write the PNG files into",
    )
    args = parser.parse_args()

    paths = save_all_visualizations(args.output_dir)
    for path in paths:
        print(f"Saved {path.resolve()}")


if __name__ == "__main__":
    main()
