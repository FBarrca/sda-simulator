"""Compare inventory replenishment policies on shared demand futures.

The module evaluates a *ladder* of policies -- from a naive chase-demand rule up
to a demand-scaled rule and a SciPy-optimised base stock -- against the same
seeded Poisson demand scenarios, prints a metrics table, and renders a scorecard
PNG.

Run it from the repository root::

    uv run --with scipy -m examples.inventory.policy_comparison --no-plot
    uv run --with scipy --with matplotlib -m examples.inventory.policy_comparison

SciPy powers only the optimised policy; it falls back to the analytic optimum
when SciPy is missing. Matplotlib is needed only for the figure.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import sqrt
from pathlib import Path

import numpy as np

from examples.inventory.main import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DEMAND_LAMBDA,
    DEFAULT_HOLDING_COST,
    DEFAULT_HORIZON,
    DEFAULT_N_SCENARIOS,
    DEFAULT_ORDER_COST,
    DEFAULT_SEED,
    DEFAULT_STOCKOUT_COST,
    build_result,
)
from examples.inventory.policies import (
    BaseStockPolicy,
    ChaseDemandPolicy,
    DemandScaledOrderUpToPolicy,
    FixedOrderQuantityPolicy,
    OptimizedBaseStockPolicy,
    OrderUpToPolicy,
)
from sda import Policy, SimulationResult

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent

_LAMBDA = DEFAULT_DEMAND_LAMBDA
_STD = sqrt(DEFAULT_DEMAND_LAMBDA)


@dataclass(frozen=True)
class PolicySpec:
    """A named policy plus a zero-arg factory that constructs it."""

    name: str
    label: str
    factory: Callable[[], Policy]


# The ladder, from naive to optimiser -- each rung isolates one lever.
DEFAULT_POLICIES: tuple[PolicySpec, ...] = (
    PolicySpec("chase_demand", "Chase demand", ChaseDemandPolicy),
    PolicySpec(
        "reorder_fixed_qty",
        "Fixed order qty (s, Q)",
        lambda: FixedOrderQuantityPolicy(reorder_point=30, order_quantity=50),
    ),
    PolicySpec(
        "reorder_up_to",
        "Order-up-to (s, S)",
        lambda: OrderUpToPolicy(reorder_point=30, order_up_to=80),
    ),
    PolicySpec("base_stock", "Base stock (daily top-up)", lambda: BaseStockPolicy(base_stock=80)),
    PolicySpec(
        "demand_scaled",
        "Demand-scaled (fair)",
        lambda: DemandScaledOrderUpToPolicy(
            demand_mean=_LAMBDA,
            demand_std=_STD,
            stockout_cost=DEFAULT_STOCKOUT_COST,
            holding_cost=DEFAULT_HOLDING_COST,
        ),
    ),
    PolicySpec(
        "optimized_base_stock",
        "Optimised (SciPy)",
        lambda: OptimizedBaseStockPolicy(
            demand_mean=_LAMBDA,
            holding_cost=DEFAULT_HOLDING_COST,
            stockout_cost=DEFAULT_STOCKOUT_COST,
        ),
    ),
)

# Stable per-policy colours for the scorecard.
POLICY_COLORS: dict[str, str] = {
    "chase_demand": "#00b4d8",
    "reorder_fixed_qty": "#f9c74f",
    "reorder_up_to": "#3a86ff",
    "base_stock": "#e76f51",
    "demand_scaled": "#2a9d8f",
    "optimized_base_stock": "#7b2cbf",
}


@dataclass(frozen=True)
class PolicySummary:
    """Per-policy metric bundle used by the table and the figure."""

    name: str
    label: str
    mean_total_cost: float
    ci95_total_cost: tuple[float, float]
    p95_total_cost: float
    cvar95_total_cost: float
    fill_rate: float
    stockout_rate: float
    inventory_mean: float
    orders_per_day: float
    order_cost: float
    holding_cost: float
    stockout_cost: float


def evaluate_policy(
    spec: PolicySpec,
    *,
    horizon: int = DEFAULT_HORIZON,
    n_scenarios: int = DEFAULT_N_SCENARIOS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seed: int | None = DEFAULT_SEED,
) -> SimulationResult:
    """Evaluate one policy against the shared seeded demand futures."""
    return build_result(
        policy=spec.factory(),
        horizon=horizon,
        n_scenarios=n_scenarios,
        batch_size=batch_size,
        seed=seed,
    )


def summarize_policy(spec: PolicySpec, result: SimulationResult, *, horizon: int) -> PolicySummary:
    """Reduce one evaluation into a :class:`PolicySummary`."""
    total_cost = result.metric("total_cost")
    values = total_cost.values()
    orders_per_day = result.metric("order_quantity").mean()
    inventory_mean = result.metric("inventory").mean()
    lost_sales_mean = result.metric("lost_sales").mean()

    # Average per-scenario cost split (each scenario logs `horizon` daily values).
    return PolicySummary(
        name=spec.name,
        label=spec.label,
        mean_total_cost=float(np.mean(values)),
        ci95_total_cost=_mean_ci95(values),
        p95_total_cost=float(total_cost.percentile(95)),
        cvar95_total_cost=float(total_cost.cvar(0.95)),
        fill_rate=float(result.metric("fill_rate").mean()),
        stockout_rate=float(result.metric("stockout").mean()),
        inventory_mean=float(inventory_mean),
        orders_per_day=float(orders_per_day),
        order_cost=float(DEFAULT_ORDER_COST * orders_per_day * horizon),
        holding_cost=float(DEFAULT_HOLDING_COST * inventory_mean * horizon),
        stockout_cost=float(DEFAULT_STOCKOUT_COST * lost_sales_mean * horizon),
    )


def evaluate_policy_set(
    policies: Sequence[PolicySpec] = DEFAULT_POLICIES,
    *,
    horizon: int = DEFAULT_HORIZON,
    n_scenarios: int = DEFAULT_N_SCENARIOS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seed: int | None = DEFAULT_SEED,
) -> tuple[list[PolicySummary], dict[str, SimulationResult]]:
    """Evaluate every policy and return summaries plus the raw results."""
    results: dict[str, SimulationResult] = {}
    summaries: list[PolicySummary] = []
    for spec in policies:
        result = evaluate_policy(
            spec, horizon=horizon, n_scenarios=n_scenarios, batch_size=batch_size, seed=seed
        )
        results[spec.name] = result
        summaries.append(summarize_policy(spec, result, horizon=horizon))
    return summaries, results


def _mean_ci95(values: np.ndarray) -> tuple[float, float]:
    """Normal-approximation 95% confidence interval for the mean."""
    array = np.asarray(values, dtype=float)
    if array.size < 2:
        mean = float(array.mean()) if array.size else float("nan")
        return (mean, mean)
    mean = float(array.mean())
    half = 1.96 * float(array.std(ddof=1)) / sqrt(array.size)
    return (mean - half, mean + half)


def format_policy_table(summaries: Sequence[PolicySummary]) -> str:
    """Render the per-policy comparison as an aligned text table."""
    headers = [
        "policy",
        "cost_mean",
        "cost_ci95",
        "cost_cvar95",
        "fill",
        "stockout",
        "inv_mean",
        "order/day",
    ]
    rows = [
        [
            summary.label,
            f"{summary.mean_total_cost:.1f}",
            f"({summary.ci95_total_cost[0]:.1f}, {summary.ci95_total_cost[1]:.1f})",
            f"{summary.cvar95_total_cost:.1f}",
            f"{summary.fill_rate:.3f}",
            f"{summary.stockout_rate:.3f}",
            f"{summary.inventory_mean:.1f}",
            f"{summary.orders_per_day:.1f}",
        ]
        for summary in summaries
    ]

    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]

    def render(cells: Sequence[str]) -> str:
        first = cells[0].ljust(widths[0])
        rest = [cell.rjust(widths[index]) for index, cell in enumerate(cells[1:], start=1)]
        return "  ".join([first, *rest])

    separator = "  ".join("-" * width for width in widths)
    return "\n".join([render(headers), separator, *(render(row) for row in rows)])


def save_policy_comparison(
    output: str | Path = DEFAULT_OUTPUT_DIR / "inventory_policy_comparison.png",
    *,
    summaries: Sequence[PolicySummary] | None = None,
    results: dict[str, SimulationResult] | None = None,
    horizon: int = DEFAULT_HORIZON,
    n_scenarios: int = DEFAULT_N_SCENARIOS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seed: int | None = DEFAULT_SEED,
) -> Path:
    """Render the six-panel policy scorecard PNG."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on optional dep
        raise ImportError(
            "matplotlib is required for the policy scorecard. "
            "Run with `uv run --with scipy --with matplotlib "
            "-m examples.inventory.policy_comparison`."
        ) from exc

    if summaries is None or results is None:
        summaries, results = evaluate_policy_set(
            horizon=horizon, n_scenarios=n_scenarios, batch_size=batch_size, seed=seed
        )

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 9))
    grid = fig.add_gridspec(2, 3, left=0.07, right=0.98, top=0.86, bottom=0.10, hspace=0.42, wspace=0.32)
    fig.suptitle("Inventory replenishment policy scorecard", fontsize=18, fontweight="bold", y=0.965)
    fig.text(
        0.5,
        0.915,
        f"{n_scenarios} shared Poisson demand futures, {horizon}-day horizon, seed {seed}",
        ha="center",
        fontsize=11,
        color="#4a4a4a",
    )

    _plot_cost_capital(fig.add_subplot(grid[0, 0]), summaries)
    _plot_tail_risk(fig.add_subplot(grid[0, 1]), summaries)
    _plot_cost_breakdown(fig.add_subplot(grid[0, 2]), summaries)
    _plot_cost_distribution(fig.add_subplot(grid[1, 0]), summaries, results)
    _plot_inventory_paths(fig.add_subplot(grid[1, 1]), summaries, results)
    _plot_summary_table(fig.add_subplot(grid[1, 2]), summaries)

    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _color(summary: PolicySummary) -> str:
    return POLICY_COLORS.get(summary.name, "#3a86ff")


def _short_label(label: str) -> str:
    """Drop the parenthetical qualifier so table cells are not clipped."""
    return label.split(" (")[0]


def _plot_cost_capital(ax, summaries) -> None:
    # Service is ~100% for every ordering policy, so the informative tradeoff is
    # cost against working capital (average inventory), not cost against fill.
    for summary in summaries:
        ax.scatter(
            summary.mean_total_cost,
            summary.inventory_mean,
            s=140,
            color=_color(summary),
            edgecolor="white",
            linewidth=1.2,
            label=summary.label,
            zorder=3,
        )
    ax.set_title("Cost vs working capital (lower-left is best)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Mean total cost")
    ax.set_ylabel("Avg inventory (units held)")
    ax.grid(True, color="#e5e5e5", linewidth=0.8)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")


def _hbar_axis(ax, summaries):
    labels = [summary.label for summary in summaries]
    positions = np.arange(len(summaries))
    ax.set_yticks(positions, labels, fontsize=8)
    ax.invert_yaxis()
    return positions


def _plot_tail_risk(ax, summaries) -> None:
    positions = _hbar_axis(ax, summaries)
    ax.barh(
        positions,
        [summary.cvar95_total_cost for summary in summaries],
        height=0.62,
        color=[_color(summary) for summary in summaries],
    )
    ax.set_title("Worst-case cost (CVaR 95)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Mean cost in worst 5% of futures")
    ax.grid(True, axis="x", color="#e5e5e5", linewidth=0.8)


def _plot_cost_breakdown(ax, summaries) -> None:
    positions = _hbar_axis(ax, summaries)
    order = np.array([summary.order_cost for summary in summaries])
    holding = np.array([summary.holding_cost for summary in summaries])
    stockout = np.array([summary.stockout_cost for summary in summaries])
    ax.barh(positions, order, height=0.62, color="#3a86ff", label="Order")
    ax.barh(positions, holding, height=0.62, left=order, color="#2a9d8f", label="Holding")
    ax.barh(positions, stockout, height=0.62, left=order + holding, color="#e76f51", label="Stockout")
    ax.set_title("Where the cost goes", fontsize=12, fontweight="bold")
    ax.set_xlabel("Mean cost per scenario")
    ax.grid(True, axis="x", color="#e5e5e5", linewidth=0.8)
    ax.legend(frameon=False, fontsize=8, loc="lower right", ncol=3)


def _plot_cost_distribution(ax, summaries, results) -> None:
    data = [results[summary.name].metric("total_cost").values() for summary in summaries]
    boxes = ax.boxplot(
        data,
        patch_artist=True,
        showfliers=False,
        showmeans=True,
        meanline=True,
    )
    for patch, summary in zip(boxes["boxes"], summaries):
        patch.set_facecolor(_color(summary))
        patch.set_alpha(0.48)
    ax.set_xticks(
        np.arange(1, len(summaries) + 1),
        [summary.label for summary in summaries],
        rotation=25,
        ha="right",
        fontsize=7.5,
    )
    ax.set_title("Total-cost distribution", fontsize=12, fontweight="bold")
    ax.set_ylabel("Total cost")
    ax.grid(True, axis="y", color="#e5e5e5", linewidth=0.8)


def _plot_inventory_paths(ax, summaries, results) -> None:
    for summary in summaries:
        _, times, values = results[summary.name].metric("inventory").to_trajectory_matrix()
        if values.size == 0:
            continue
        mean = np.nanmean(values, axis=0)
        ax.plot(times, mean, color=_color(summary), linewidth=2.0, label=summary.label)
    ax.set_title("Average inventory over time", fontsize=12, fontweight="bold")
    ax.set_xlabel("Day")
    ax.set_ylabel("Units on hand")
    ax.grid(True, color="#e5e5e5", linewidth=0.8)
    ax.legend(frameon=False, fontsize=7, loc="upper right", ncol=2)


def _plot_summary_table(ax, summaries) -> None:
    ax.axis("off")
    columns = ["Policy", "Cost", "CVaR95", "Fill", "Inv"]
    best_cost = min(summary.mean_total_cost for summary in summaries)
    best_fill = max(summary.fill_rate for summary in summaries)

    cell_text = []
    cell_colors = []
    for summary in summaries:
        cell_text.append(
            [
                _short_label(summary.label),
                f"{summary.mean_total_cost:.0f}",
                f"{summary.cvar95_total_cost:.0f}",
                f"{summary.fill_rate:.3f}",
                f"{summary.inventory_mean:.0f}",
            ]
        )
        row_colors = ["#ffffff"] * len(columns)
        row_colors[0] = _color(summary)
        if abs(summary.mean_total_cost - best_cost) < 1e-6:
            row_colors[1] = "#e8f5e9"
        if abs(summary.fill_rate - best_fill) < 1e-9:
            row_colors[3] = "#e8f5e9"
        cell_colors.append(row_colors)

    table = ax.table(
        cellText=cell_text,
        colLabels=columns,
        cellColours=cell_colors,
        colWidths=[0.42, 0.15, 0.16, 0.14, 0.13],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.0)
    table.scale(1, 1.5)
    for index in range(len(columns)):
        header = table[0, index]
        header.set_facecolor("#202020")
        header.set_text_props(color="white", fontweight="bold")
    ax.set_title("Summary (green = best)", fontsize=12, fontweight="bold")


def save_working_capital_plot(
    output: str | Path = DEFAULT_OUTPUT_DIR / "inventory_working_capital.png",
    *,
    summaries: Sequence[PolicySummary] | None = None,
    baseline_name: str = "reorder_up_to",
    unit_cost: float = 40.0,
    horizon: int = DEFAULT_HORIZON,
    n_scenarios: int = DEFAULT_N_SCENARIOS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seed: int | None = DEFAULT_SEED,
) -> Path:
    """Render the headline working-capital chart: average stock held per policy.

    Bars show average on-hand inventory (working capital) for each policy against
    the hand-tuned ``(s, S)`` baseline, with the capital freed by the leaner
    rules annotated in units and dollars at ``unit_cost``.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on optional dep
        raise ImportError(
            "matplotlib is required for the working-capital chart. "
            "Run with `uv run --with scipy --with matplotlib "
            "-m examples.inventory.policy_comparison`."
        ) from exc

    if summaries is None:
        summaries, _ = evaluate_policy_set(
            horizon=horizon, n_scenarios=n_scenarios, batch_size=batch_size, seed=seed
        )

    ordered = sorted(summaries, key=lambda s: s.inventory_mean, reverse=True)
    baseline = next((s for s in summaries if s.name == baseline_name), None)
    base_inv = baseline.inventory_mean if baseline else max(s.inventory_mean for s in summaries)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 6))
    positions = np.arange(len(ordered))
    ax.barh(
        positions,
        [s.inventory_mean for s in ordered],
        height=0.64,
        color=[_color(s) for s in ordered],
    )
    ax.set_yticks(positions, [s.label for s in ordered], fontsize=9)
    ax.invert_yaxis()
    ax.axvline(
        base_inv,
        color="#0f172a",
        linestyle="--",
        linewidth=1.6,
        label=f"(s, S) baseline = {base_inv:.0f} units",
    )

    # Round to one decimal so the annotations match the published metric tables.
    base_disp = round(base_inv, 1)
    for position, summary in zip(positions, ordered):
        inv = round(summary.inventory_mean, 1)
        freed = base_disp - inv
        text = f"{inv:.1f} u"
        if freed > 0.05:
            text += f"  (−{freed:.1f} u ≈ ${freed * unit_cost:,.0f} freed)"
        ax.text(summary.inventory_mean + base_inv * 0.02, position, text, va="center", fontsize=9)

    ax.set_title(
        "Working capital carried per policy", fontsize=15, fontweight="bold"
    )
    ax.set_xlabel(f"Average units on hand  (capital at ${unit_cost:.0f}/unit)")
    ax.set_xlim(0, base_inv * 1.5)
    ax.grid(True, axis="x", color="#e5e5e5", linewidth=0.8)
    ax.legend(frameon=False, fontsize=10, loc="lower right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare inventory replenishment policies.")
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--n-scenarios", type=int, default=DEFAULT_N_SCENARIOS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_DIR / "inventory_policy_comparison.png"),
        help="path to the scorecard PNG",
    )
    parser.add_argument("--no-plot", action="store_true", help="skip rendering the scorecard PNG")
    args = parser.parse_args()

    summaries, results = evaluate_policy_set(
        horizon=args.horizon,
        n_scenarios=args.n_scenarios,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    print(
        f"Inventory policy comparison "
        f"({args.horizon}-day horizon, {args.n_scenarios} scenarios, seed {args.seed})"
    )
    print(format_policy_table(summaries))

    if not args.no_plot:
        output_path = save_policy_comparison(
            args.output,
            summaries=summaries,
            results=results,
            horizon=args.horizon,
            n_scenarios=args.n_scenarios,
        )
        print(f"Saved scorecard to {output_path.resolve()}")
        capital_path = save_working_capital_plot(
            summaries=summaries,
            horizon=args.horizon,
            n_scenarios=args.n_scenarios,
        )
        print(f"Saved working-capital chart to {capital_path.resolve()}")


if __name__ == "__main__":
    main()
