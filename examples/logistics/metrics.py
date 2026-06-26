from __future__ import annotations

from sda.metrics import Metric, MetricStore
from sda.model import StepRecord


class OnTimeRateMetric(Metric):
    name = "on_time_rate"

    def on_step(self, step: StepRecord, store: MetricStore) -> None:
        store.log(self.name, step.info[self.name], step.scenario_ids, step.t, "step")


class PriorityWeightedOnTimeMetric(Metric):
    name = "priority_weighted_on_time_rate"

    def on_step(self, step: StepRecord, store: MetricStore) -> None:
        store.log(self.name, step.info[self.name], step.scenario_ids, step.t, "step")


class LateCostMetric(Metric):
    name = "late_cost"

    def on_step(self, step: StepRecord, store: MetricStore) -> None:
        store.log(self.name, step.info[self.name], step.scenario_ids, step.t, "step")


class DispatchCostMetric(Metric):
    name = "dispatch_cost"

    def on_step(self, step: StepRecord, store: MetricStore) -> None:
        store.log(self.name, step.info[self.name], step.scenario_ids, step.t, "step")


class PendingBacklogMetric(Metric):
    name = "pending_backlog"

    def on_step(self, step: StepRecord, store: MetricStore) -> None:
        store.log(self.name, step.info[self.name], step.scenario_ids, step.t, "step")


class DispatchedOrderMetric(Metric):
    name = "dispatched_order_count"

    def on_step(self, step: StepRecord, store: MetricStore) -> None:
        store.log(self.name, step.info[self.name], step.scenario_ids, step.t, "step")


class VehicleUtilizationMetric(Metric):
    name = "vehicle_utilization"

    def on_step(self, step: StepRecord, store: MetricStore) -> None:
        store.log(self.name, step.info[self.name], step.scenario_ids, step.t, "step")
