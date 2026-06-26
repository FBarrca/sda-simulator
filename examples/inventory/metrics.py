from __future__ import annotations

import numpy as np

from sda.metrics import Metric, MetricStore
from sda.model import StepRecord


class InventoryMetric(Metric):
    name = "inventory"

    def on_step(self, step: StepRecord, store: MetricStore) -> None:
        store.log(
            name=self.name,
            values=step.next_state,
            scenario_ids=step.scenario_ids,
            t=step.t,
            level="step",
        )


class StockoutMetric(Metric):
    name = "stockout"

    def on_step(self, step: StepRecord, store: MetricStore) -> None:
        store.log(
            name=self.name,
            values=np.asarray(step.info["lost_sales"]) > 0,
            scenario_ids=step.scenario_ids,
            t=step.t,
            level="step",
        )


class FillRateMetric(Metric):
    name = "fill_rate"

    def on_step(self, step: StepRecord, store: MetricStore) -> None:
        store.log(
            name=self.name,
            values=step.info["fill_rate"],
            scenario_ids=step.scenario_ids,
            t=step.t,
            level="step",
        )


class OrderQuantityMetric(Metric):
    name = "order_quantity"

    def on_step(self, step: StepRecord, store: MetricStore) -> None:
        store.log(
            name=self.name,
            values=step.info["order_quantity"],
            scenario_ids=step.scenario_ids,
            t=step.t,
            level="step",
        )
