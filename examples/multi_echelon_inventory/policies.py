from __future__ import annotations

import numpy as np

from examples.multi_echelon_inventory.domain import (
    DEFAULT_NETWORK,
    ReferencePolicyParameters,
    policy_parameters_from_guess,
    upstream_nodes,
)
from sda import EventRecord, Policy


class BaseStockReorderPolicy(Policy):
    """Reference base-stock policy with a reorder point."""

    def __init__(
        self,
        reorder_point,
        base_stock,
        *,
        network=DEFAULT_NETWORK,
        reorder_tolerance: float = 1.05,
    ) -> None:
        self.reorder_point = np.asarray(reorder_point, dtype=float)
        self.base_stock = np.asarray(base_stock, dtype=float)
        if self.reorder_point.ndim != 1 or self.base_stock.ndim != 1:
            raise ValueError("reorder_point and base_stock must be one-dimensional")
        if self.reorder_point.shape != self.base_stock.shape:
            raise ValueError("reorder_point and base_stock must have the same length")

        self.network = np.asarray(network, dtype=int)
        if self.network.shape != (self.num_nodes, self.num_nodes):
            raise ValueError("network shape must match policy parameter length")
        self.upstream = upstream_nodes(self.network)
        self.reorder_tolerance = float(reorder_tolerance)

    @property
    def num_nodes(self) -> int:
        """Return the number of nodes controlled by the policy."""
        return int(self.base_stock.shape[0])

    @classmethod
    def from_optimizer_guess(
        cls,
        initial_guess,
        *,
        network=DEFAULT_NETWORK,
    ) -> "BaseStockReorderPolicy":
        """Build a policy from the reference optimizer vector."""
        parameters = policy_parameters_from_guess(initial_guess)
        return cls(
            reorder_point=parameters.reorder_point,
            base_stock=parameters.base_stock,
            network=network,
        )

    @classmethod
    def from_parameters(
        cls,
        parameters: ReferencePolicyParameters,
        *,
        network=DEFAULT_NETWORK,
    ) -> "BaseStockReorderPolicy":
        """Build a policy from explicit reference parameter arrays."""
        return cls(
            reorder_point=parameters.reorder_point,
            base_stock=parameters.base_stock,
            network=network,
        )

    def act(self, state, env, history: list[EventRecord]):
        """Return replenishment orders for every scenario and node."""
        del env, history
        batch_size = len(state)
        order_quantity = np.zeros((batch_size, self.num_nodes), dtype=float)
        order_placed = np.zeros((batch_size, self.num_nodes), dtype=bool)

        for scenario_index, scenario in enumerate(state):
            for node_index, upstream in enumerate(self.upstream):
                if upstream is None:
                    continue
                node = scenario.nodes[node_index]
                if (
                    node.inventory_position
                    <= self.reorder_tolerance * self.reorder_point[node_index]
                ):
                    order_quantity[scenario_index, node_index] = (
                        self.base_stock[node_index] - node.on_hand_inventory
                    )
                    order_placed[scenario_index, node_index] = True

        return {
            "order_quantity": order_quantity,
            "order_placed": order_placed,
        }

    def order_for_node(self, scenario, node_index: int) -> tuple[bool, float]:
        """Return whether one facility should order and how much."""
        upstream = self.upstream[node_index]
        if upstream is None:
            return False, 0.0

        node = scenario.nodes[node_index]
        if (
            node.inventory_position
            <= self.reorder_tolerance * self.reorder_point[node_index]
        ):
            return True, float(self.base_stock[node_index] - node.on_hand_inventory)
        return False, 0.0
