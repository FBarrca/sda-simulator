from __future__ import annotations

from examples.logistics.data import LogisticsScenarioLoader
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
from examples.logistics.policies import RiskAwareDispatchPolicy
from sda import SimulationResult, Simulator, StepCostMetric, TotalCostMetric


def logistics_metrics():
    return [
        StepCostMetric(),
        TotalCostMetric(),
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
    scenarios = LogisticsScenarioLoader(
        horizon=horizon,
        n_scenarios=n_scenarios,
        batch_size=batch_size,
        seed=seed,
    )
    model = LogisticsModel(policy=policy or RiskAwareDispatchPolicy())
    simulator = Simulator(metrics=logistics_metrics())
    return simulator.evaluate(model, scenarios)


def main() -> None:
    result = build_result()
    print(f"Total cost mean: {result.metric('total_cost').mean():.2f}")
    print(f"Total cost p95: {result.metric('total_cost').percentile(95):.2f}")
    print(f"Total cost CVaR 95: {result.metric('total_cost').cvar(0.95):.2f}")
    print(f"On-time rate mean: {result.metric('on_time_rate').mean():.3f}")
    print(
        "Priority-weighted on-time mean: "
        f"{result.metric('priority_weighted_on_time_rate').mean():.3f}"
    )
    print(f"Late cost mean: {result.metric('late_cost').mean():.2f}")
    print(f"Pending backlog mean: {result.metric('pending_backlog').mean():.2f}")
    print(f"Dispatched orders/day mean: {result.metric('dispatched_order_count').mean():.2f}")
    print(f"Vehicle utilization mean: {result.metric('vehicle_utilization').mean():.3f}")


if __name__ == "__main__":
    main()
