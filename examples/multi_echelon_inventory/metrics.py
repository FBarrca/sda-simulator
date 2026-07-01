from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from examples.multi_echelon_inventory.domain import (
    DEFAULT_SERVICE_TARGET,
    REFERENCE_SERVICE_PENALTY,
)


@dataclass(frozen=True)
class ReferenceObjectiveSummary:
    """Aggregate objective components matching the reference scripts."""

    objective: float
    average_on_hand: float
    service_level: np.ndarray
    service_shortfall: np.ndarray
    service_penalty: float


def reference_metric_names(
    num_nodes: int = 6,
    *,
    include_daily_metrics: bool = False,
) -> list[str]:
    """Return metric names emitted by the reference SimPy model."""
    names = [
        "reference_average_on_hand",
        "reference_service_penalty",
        "replication_objective",
    ]
    names.extend(f"service_level_node_{node_index}" for node_index in range(num_nodes))
    names.extend(f"average_on_hand_node_{node_index}" for node_index in range(num_nodes))
    if include_daily_metrics:
        names.append("total_on_hand")
        daily_prefixes = [
            "demand",
            "shipped",
            "lost_sales",
            "backorder",
            "on_hand_inventory",
            "inventory_position",
        ]
        for prefix in daily_prefixes:
            names.extend(
                f"{prefix}_node_{node_index}" for node_index in range(num_nodes)
            )
    return names


def reference_metrics(
    num_nodes: int = 6,
    *,
    include_daily_metrics: bool = False,
    **_: object,
) -> list[str]:
    """Return standard reference metric names.

    The SimPy-native framework no longer accepts metric hook objects at
    ``evaluate`` time; the model emits these metrics directly through its
    recorder. This helper is kept as a small discoverability aid for the
    example.
    """
    return reference_metric_names(
        num_nodes=num_nodes,
        include_daily_metrics=include_daily_metrics,
    )


def summarize_reference_result(
    result,
    *,
    service_target=DEFAULT_SERVICE_TARGET,
    penalty_weight: float = REFERENCE_SERVICE_PENALTY,
) -> ReferenceObjectiveSummary:
    """Summarize an SDA result with the reference objective formula."""
    service_target_array = np.asarray(service_target, dtype=float)
    service_level = np.array(
        [
            result[f"service_level_node_{node_index}"].trajectory_level().mean()
            for node_index in range(len(service_target_array))
        ],
        dtype=float,
    )
    average_on_hand = result["reference_average_on_hand"].trajectory_level().mean()
    service_shortfall = np.maximum(0.0, service_target_array - service_level)
    service_penalty = float(penalty_weight * np.sum(service_shortfall))
    return ReferenceObjectiveSummary(
        objective=float(average_on_hand + service_penalty),
        average_on_hand=float(average_on_hand),
        service_level=service_level,
        service_shortfall=service_shortfall,
        service_penalty=service_penalty,
    )
