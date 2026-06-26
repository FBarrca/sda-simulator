from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from examples.logistics.main import build_result
from examples.logistics.policies import (
    GreedyPolicy,
    LookaheadRolloutPolicy,
    MilpPolicy,
    NearestFeasiblePolicy,
    PriorityDeadlinePolicy,
    PriorityPolicy,
    RandomPolicy,
    RiskAwareDispatchPolicy,
)
from sda import SimulationResult


@dataclass(frozen=True)
class PolicySpec:
    name: str
    factory: Callable[[], object]


@dataclass(frozen=True)
class PolicySummary:
    name: str
    mean_total_cost: float
    ci95_total_cost: tuple[float, float]
    p95_total_cost: float
    cvar95_total_cost: float
    on_time_rate: float
    priority_weighted_on_time_rate: float
    late_cost: float
    pending_backlog: float
    dispatched_order_count: float
    vehicle_utilization: float


DEFAULT_POLICIES = (
    PolicySpec("Random", lambda: RandomPolicy(seed=42)),
    PolicySpec("Greedy", GreedyPolicy),
    PolicySpec("Priority", PriorityPolicy),
    PolicySpec("MILP distance-priority", MilpPolicy),
    PolicySpec("Lookahead rollout", LookaheadRolloutPolicy),
    PolicySpec("Nearest feasible", NearestFeasiblePolicy),
    PolicySpec("Priority deadline", PriorityDeadlinePolicy),
    PolicySpec("Risk aware", RiskAwareDispatchPolicy),
)


def evaluate_policy(
    spec: PolicySpec,
    *,
    horizon: int = 28,
    n_scenarios: int = 500,
    batch_size: int = 64,
    seed: int = 42,
) -> SimulationResult:
    return build_result(
        policy=spec.factory(),
        horizon=horizon,
        n_scenarios=n_scenarios,
        batch_size=batch_size,
        seed=seed,
    )


def summarize_policy(spec: PolicySpec, result: SimulationResult) -> PolicySummary:
    total_cost_values = result.metric("total_cost").values()
    return PolicySummary(
        name=spec.name,
        mean_total_cost=float(np.mean(total_cost_values)),
        ci95_total_cost=_mean_ci95(total_cost_values),
        p95_total_cost=result.metric("total_cost").percentile(95),
        cvar95_total_cost=result.metric("total_cost").cvar(0.95),
        on_time_rate=result.metric("on_time_rate").mean(),
        priority_weighted_on_time_rate=result.metric("priority_weighted_on_time_rate").mean(),
        late_cost=result.metric("late_cost").mean(),
        pending_backlog=result.metric("pending_backlog").mean(),
        dispatched_order_count=result.metric("dispatched_order_count").mean(),
        vehicle_utilization=result.metric("vehicle_utilization").mean(),
    )


def evaluate_policy_set(
    policies: tuple[PolicySpec, ...] = DEFAULT_POLICIES,
    *,
    horizon: int = 28,
    n_scenarios: int = 500,
    batch_size: int = 64,
    seed: int = 42,
) -> tuple[list[PolicySummary], dict[str, SimulationResult]]:
    results = {
        spec.name: evaluate_policy(
            spec,
            horizon=horizon,
            n_scenarios=n_scenarios,
            batch_size=batch_size,
            seed=seed,
        )
        for spec in policies
    }
    summaries = [summarize_policy(spec, results[spec.name]) for spec in policies]
    return summaries, results


def format_policy_table(summaries: list[PolicySummary]) -> str:
    rows = [
        [
            summary.name,
            f"{summary.mean_total_cost:.0f}",
            _format_ci(summary.ci95_total_cost),
            f"{summary.cvar95_total_cost:.0f}",
            f"{summary.priority_weighted_on_time_rate:.1%}",
            f"{summary.late_cost:.1f}",
            f"{summary.pending_backlog:.1f}",
            f"{summary.dispatched_order_count:.1f}",
            f"{summary.vehicle_utilization:.1%}",
        ]
        for summary in summaries
    ]
    headers = [
        "policy",
        "total_mean",
        "total_ci95",
        "cost_cvar95",
        "prio_ot",
        "late_cost",
        "backlog",
        "dispatch/day",
        "util",
    ]
    widths = [
        max(len(str(row[index])) for row in [headers, *rows])
        for index in range(len(headers))
    ]
    header = "  ".join(
        value.ljust(width) if index == 0 else value.rjust(width)
        for index, (value, width) in enumerate(zip(headers, widths, strict=True))
    )
    separator = "  ".join("-" * width for width in widths)
    body = [
        "  ".join(
            value.ljust(width) if index == 0 else value.rjust(width)
            for index, (value, width) in enumerate(zip(row, widths, strict=True))
        )
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def save_policy_comparison(
    output: str | Path = "examples/logistics/logistics_policy_comparison.png",
    *,
    summaries: list[PolicySummary] | None = None,
    results: dict[str, SimulationResult] | None = None,
    horizon: int = 28,
    n_scenarios: int = 500,
) -> Path:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import PercentFormatter
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for logistics policy comparison plots. "
            "Run with `uv run --with matplotlib ...`."
        ) from exc

    if summaries is None or results is None:
        summaries, results = evaluate_policy_set()
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    colors = {
        "Random": "#6c757d",
        "Greedy": "#3a86ff",
        "Priority": "#2a9d8f",
        "MILP distance-priority": "#e76f51",
        "Lookahead rollout": "#7b2cbf",
        "Nearest feasible": "#00b4d8",
        "Priority deadline": "#f9c74f",
        "Risk aware": "#4d908e",
    }

    fig = plt.figure(figsize=(15, 9), constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        3,
        left=0.06,
        right=0.98,
        top=0.84,
        bottom=0.08,
        hspace=0.36,
        wspace=0.36,
    )
    fig.suptitle("Logistics dispatch policy scorecard", fontsize=18, fontweight="bold", y=0.96)
    fig.text(
        0.5,
        0.915,
        f"{n_scenarios} shared bootstrap futures, {horizon} simulated days, Spanish warehouse-to-customer lanes.",
        ha="center",
        fontsize=10,
        color="#4a4a4a",
    )

    _plot_cost_service(fig.add_subplot(grid[0, 0]), summaries, colors, PercentFormatter)
    _plot_tail_risk(fig.add_subplot(grid[0, 1]), summaries, colors)
    _plot_backlog(fig.add_subplot(grid[0, 2]), summaries, colors)
    _plot_cost_distribution(fig.add_subplot(grid[1, 0]), summaries, results, colors)
    _plot_backlog_paths(fig.add_subplot(grid[1, 1]), results, colors)
    _plot_summary_table(fig.add_subplot(grid[1, 2]), summaries, colors)

    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _plot_cost_service(ax, summaries, colors, percent_formatter) -> None:
    for summary in summaries:
        ax.scatter(
            summary.mean_total_cost,
            summary.priority_weighted_on_time_rate,
            s=130,
            color=colors[summary.name],
            edgecolor="white",
            linewidth=1.2,
            label=summary.name,
            zorder=3,
        )
    ax.yaxis.set_major_formatter(percent_formatter(1.0))
    ax.set_title("Cost and priority service")
    ax.set_xlabel("Mean total cost")
    ax.set_ylabel("Priority-weighted on-time rate")
    ax.grid(True, color="#e5e5e5", linewidth=0.8)
    ax.legend(frameon=False, fontsize=8)


def _plot_tail_risk(ax, summaries, colors) -> None:
    y = np.arange(len(summaries))
    ax.barh(
        y,
        [summary.cvar95_total_cost for summary in summaries],
        color=[colors[summary.name] for summary in summaries],
        height=0.62,
    )
    ax.set_yticks(y, [summary.name for summary in summaries])
    ax.invert_yaxis()
    ax.set_title("Worst-tail cost")
    ax.set_xlabel("CVaR 95 total cost")
    ax.grid(True, axis="x", color="#e5e5e5", linewidth=0.8)


def _plot_backlog(ax, summaries, colors) -> None:
    y = np.arange(len(summaries))
    ax.barh(
        y,
        [summary.pending_backlog for summary in summaries],
        color=[colors[summary.name] for summary in summaries],
        height=0.62,
    )
    ax.set_yticks(y, [summary.name for summary in summaries])
    ax.invert_yaxis()
    ax.set_title("Pending backlog")
    ax.set_xlabel("Mean orders awaiting dispatch")
    ax.grid(True, axis="x", color="#e5e5e5", linewidth=0.8)


def _plot_cost_distribution(ax, summaries, results, colors) -> None:
    data = [results[summary.name].metric("total_cost").values() for summary in summaries]
    box = ax.boxplot(data, patch_artist=True, showfliers=False, showmeans=True, meanline=True)
    for patch, summary in zip(box["boxes"], summaries, strict=True):
        patch.set_facecolor(colors[summary.name])
        patch.set_alpha(0.48)
        patch.set_edgecolor("#303030")
    ax.set_xticks(np.arange(1, len(summaries) + 1), [summary.name for summary in summaries], rotation=25, ha="right")
    ax.set_title("Total cost distribution")
    ax.set_ylabel("Total cost")
    ax.grid(True, axis="y", color="#e5e5e5", linewidth=0.8)


def _plot_backlog_paths(ax, results, colors) -> None:
    for name in results:
        _, times, values = results[name].metric("pending_backlog").to_trajectory_matrix()
        mean = np.nanmean(values, axis=0)
        p10 = np.nanpercentile(values, 10, axis=0)
        p90 = np.nanpercentile(values, 90, axis=0)
        ax.plot(times, mean, label=name, color=colors[name], linewidth=2)
        ax.fill_between(times, p10, p90, color=colors[name], alpha=0.10)
    ax.set_title("Backlog over time")
    ax.set_xlabel("Day")
    ax.set_ylabel("Pending orders")
    ax.grid(True, color="#e5e5e5", linewidth=0.8)
    ax.legend(frameon=False, fontsize=8)


def _plot_summary_table(ax, summaries, colors) -> None:
    ax.axis("off")
    rows = [
        [
            summary.name,
            f"{summary.mean_total_cost:.0f}",
            f"{summary.p95_total_cost:.0f}",
            f"{summary.priority_weighted_on_time_rate:.1%}",
            f"{summary.pending_backlog:.1f}",
            f"{summary.vehicle_utilization:.1%}",
        ]
        for summary in summaries
    ]
    table = ax.table(
        cellText=rows,
        colLabels=["Policy", "Mean", "P95", "Prio OT", "Backlog", "Util"],
        loc="center",
        cellLoc="center",
        colLoc="center",
        colWidths=[0.30, 0.13, 0.13, 0.15, 0.14, 0.13],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.0)
    table.scale(1, 1.35)
    best_cost = min(summary.mean_total_cost for summary in summaries)
    best_service = max(summary.priority_weighted_on_time_rate for summary in summaries)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d0d0d0")
        if row == 0:
            cell.set_facecolor("#202020")
            cell.set_text_props(color="white", fontweight="bold")
            continue
        summary = summaries[row - 1]
        if col == 0:
            cell.set_facecolor(colors[summary.name])
            cell.set_text_props(color="white", fontweight="bold")
        elif (col == 1 and summary.mean_total_cost == best_cost) or (
            col == 3 and summary.priority_weighted_on_time_rate == best_service
        ):
            cell.set_facecolor("#e8f5e9")
        else:
            cell.set_facecolor("white")
    ax.set_title("Policy summary", pad=12)


def _mean_ci95(values: np.ndarray) -> tuple[float, float]:
    if values.size == 0:
        return (float("nan"), float("nan"))
    if values.size == 1:
        mean = float(np.mean(values))
        return (mean, mean)

    mean = float(np.mean(values))
    half_width = 1.96 * float(np.std(values, ddof=1)) / np.sqrt(values.size)
    return (mean - half_width, mean + half_width)


def _format_ci(bounds: tuple[float, float]) -> str:
    low, high = bounds
    return f"({low:.0f}, {high:.0f})"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="examples/logistics/logistics_policy_comparison.png",
        help="path to the PNG file to create",
    )
    parser.add_argument("--horizon", type=int, default=28, help="days to simulate")
    parser.add_argument(
        "--n-scenarios",
        type=int,
        default=500,
        help="number of bootstrap futures to evaluate",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="scenario batch size")
    parser.add_argument("--seed", type=int, default=42, help="bootstrap seed")
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="print the table without creating the PNG plot",
    )
    args = parser.parse_args()

    summaries, results = evaluate_policy_set(
        horizon=args.horizon,
        n_scenarios=args.n_scenarios,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    print(
        "Logistics policy comparison "
        f"({args.horizon}-day horizon, {args.n_scenarios} scenarios, seed {args.seed})"
    )
    print(format_policy_table(summaries))

    if not args.no_plot:
        output_path = save_policy_comparison(
            output=args.output,
            summaries=summaries,
            results=results,
            horizon=args.horizon,
            n_scenarios=args.n_scenarios,
        )
        print(f"Saved logistics policy comparison plot to {output_path.resolve()}")


if __name__ == "__main__":
    main()
