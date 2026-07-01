from __future__ import annotations

from sda import EventRecord, Policy


class OrderUpToPolicy(Policy):
    """Orders up to a target level once inventory falls below a reorder point."""

    def __init__(self, reorder_point: float, order_up_to: float) -> None:
        if order_up_to < reorder_point:
            raise ValueError("order_up_to must be greater than or equal to reorder_point")
        self.reorder_point = float(reorder_point)
        self.order_up_to = float(order_up_to)

    def act(self, state, env, history: list[EventRecord]):
        """Return the replenishment quantity for the current inventory."""
        del env, history
        if state.inventory < self.reorder_point:
            return max(self.order_up_to - state.inventory, 0.0)
        return 0.0
