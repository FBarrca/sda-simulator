from __future__ import annotations

from sda import InfoMetric


class OnTimeRateMetric(InfoMetric):
    def __init__(self) -> None:
        super().__init__("on_time_rate")


class PriorityWeightedOnTimeMetric(InfoMetric):
    def __init__(self) -> None:
        super().__init__("priority_weighted_on_time_rate")


class LateCostMetric(InfoMetric):
    def __init__(self) -> None:
        super().__init__("late_cost")


class DispatchCostMetric(InfoMetric):
    def __init__(self) -> None:
        super().__init__("dispatch_cost")


class PendingBacklogMetric(InfoMetric):
    def __init__(self) -> None:
        super().__init__("pending_backlog")


class DispatchedOrderMetric(InfoMetric):
    def __init__(self) -> None:
        super().__init__("dispatched_order_count")


class VehicleUtilizationMetric(InfoMetric):
    def __init__(self) -> None:
        super().__init__("vehicle_utilization")
