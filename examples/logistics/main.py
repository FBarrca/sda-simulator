from __future__ import annotations

from examples.logistics.data import LogisticsDataModule
from examples.logistics.metrics import (
    DispatchCostMetric,
    DispatchedOrderMetric,
    LateCostMetric,
    OnTimeRateMetric,
    PendingBacklogMetric,
    PriorityWeightedOnTimeMetric,
    VehicleUtilizationMetric,
)
from examples.logistics.models import LogisticsModel
from examples.logistics.policies import PriorityPolicy
from sda import SimulationResult, evaluate


def logistics_metrics():
    return [
        OnTimeRateMetric(),
        PriorityWeightedOnTimeMetric(),
        LateCostMetric(),
        DispatchCostMetric(),
        PendingBacklogMetric(),
        DispatchedOrderMetric(),
        VehicleUtilizationMetric(),
    ]


def build_result(
    *,
    policy=None,
    horizon: int = 28,
    n_scenarios: int = 500,
    batch_size: int = 64,
    seed: int = 42,
) -> SimulationResult:
    data = LogisticsDataModule(
        horizon=horizon,
        n_scenarios=n_scenarios,
        batch_size=batch_size,
        seed=seed,
    )
    model = LogisticsModel(policy=policy or PriorityPolicy())
    return evaluate(model, data, extra_metrics=logistics_metrics())


def main() -> None:
    result = build_result()
    print(f"Total cost mean: {result['total_cost'].mean():.2f}")
    print(f"Total cost p95: {result['total_cost'].percentile(95):.2f}")
    print(f"Total cost CVaR 95: {result['total_cost'].cvar(0.95):.2f}")
    print(f"On-time rate mean: {result['on_time_rate'].mean():.3f}")
    print(
        "Priority-weighted on-time mean: "
        f"{result['priority_weighted_on_time_rate'].mean():.3f}"
    )
    print(f"Late cost mean: {result['late_cost'].mean():.2f}")
    print(f"Pending backlog mean: {result['pending_backlog'].mean():.2f}")
    print(f"Dispatched orders/day mean: {result['dispatched_order_count'].mean():.2f}")
    print(f"Vehicle utilization mean: {result['vehicle_utilization'].mean():.3f}")


if __name__ == "__main__":
    main()
