from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from examples.logistics.data import DEMAND_BY_DAY, synthetic_history
from examples.logistics.network import SKUS


def save_demand_plot(
    *,
    seed: int = 7,
    days: int = 365,
    output: str | Path = "examples/logistics/logistics_synthetic_demand.png",
) -> Path:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for logistics demand plots. "
            "Run with `uv run --with matplotlib ...`."
        ) from exc

    history = synthetic_history(days=days, seed=seed)
    daily_by_sku = _daily_quantity_by_sku(history.orders)
    rolling = _rolling_mean(daily_by_sku, window=7)
    day_of_week = np.arange(days) % 7
    weekday_average = np.asarray(
        [
            daily_by_sku[day_of_week == index].sum(axis=1).mean()
            for index in range(7)
        ],
        dtype=float,
    )

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    colors = ["#3a86ff", "#2a9d8f", "#ff9f1c", "#e76f51"]

    fig, (top, bottom) = plt.subplots(
        2,
        1,
        figsize=(13, 8),
        gridspec_kw={"height_ratios": [2.1, 1.0], "hspace": 0.32},
    )
    x = np.arange(days)
    top.stackplot(x, rolling.T, labels=SKUS, colors=colors, alpha=0.82)
    top.set_title(f"Synthetic logistics demand, seed {seed}", fontsize=15, fontweight="bold")
    top.set_ylabel("7-day rolling ordered units")
    top.grid(True, axis="y", color="#e5e5e5", linewidth=0.8)
    top.legend(ncol=4, loc="upper left", frameon=False, fontsize=8)

    bottom.bar(
        np.arange(7),
        weekday_average,
        color=["#3a86ff" if value >= 1 else "#9aa0a6" for value in DEMAND_BY_DAY],
        width=0.68,
    )
    bottom.set_xticks(
        np.arange(7),
        ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
    )
    bottom.set_ylabel("Average units/day")
    bottom.set_title("Average demand by day of week")
    bottom.grid(True, axis="y", color="#e5e5e5", linewidth=0.8)

    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _daily_quantity_by_sku(orders: np.ndarray) -> np.ndarray:
    values = np.zeros((len(orders), len(SKUS)), dtype=float)
    sku_index = {sku: index for index, sku in enumerate(SKUS)}
    for day, day_orders in enumerate(orders):
        for order in day_orders:
            values[day, sku_index[order.sku]] += order.quantity
    return values


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    padded = np.pad(values, ((window - 1, 0), (0, 0)), mode="edge")
    cumsum = np.cumsum(padded, axis=0)
    cumsum[window:] = cumsum[window:] - cumsum[:-window]
    return cumsum[window - 1 :] / window


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7, help="synthetic history seed")
    parser.add_argument("--days", type=int, default=365, help="number of history days")
    parser.add_argument(
        "--output",
        default="examples/logistics/logistics_synthetic_demand.png",
        help="path to the PNG file to create",
    )
    args = parser.parse_args()

    output_path = save_demand_plot(seed=args.seed, days=args.days, output=args.output)
    print(f"Saved logistics demand plot to {output_path.resolve()}")


if __name__ == "__main__":
    main()
