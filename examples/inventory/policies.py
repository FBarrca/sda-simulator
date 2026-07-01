from __future__ import annotations

import numpy as np

from sda import Policy, StepRecord


class OrderUpToPolicy(Policy):
    """Orders up to a target level once inventory falls below a reorder point."""

    def __init__(self, reorder_point: float, order_up_to: float) -> None:
        if order_up_to < reorder_point:
            raise ValueError("order_up_to must be greater than or equal to reorder_point")
        self.reorder_point = float(reorder_point)
        self.order_up_to = float(order_up_to)

    def act(self, state, t: int, history: list[StepRecord]):
        inventory = np.asarray(state, dtype=float)
        return np.where(
            inventory < self.reorder_point,
            np.maximum(self.order_up_to - inventory, 0.0),
            0.0,
        )
