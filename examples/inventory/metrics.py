from __future__ import annotations

import numpy as np

from sda import InfoMetric, StepMetric


class InventoryMetric(StepMetric):
    def __init__(self) -> None:
        super().__init__("inventory", lambda step: step.next_state)


class StockoutMetric(StepMetric):
    def __init__(self) -> None:
        super().__init__(
            "stockout",
            lambda step: np.asarray(step.info["lost_sales"]) > 0,
        )


class FillRateMetric(InfoMetric):
    def __init__(self) -> None:
        super().__init__("fill_rate")


class OrderQuantityMetric(InfoMetric):
    def __init__(self) -> None:
        super().__init__("order_quantity")
