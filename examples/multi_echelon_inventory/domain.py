from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

RUN_UNTIL = 360
REFERENCE_HORIZON = RUN_UNTIL - 1
REFERENCE_REPLICATIONS = 20
REFERENCE_SERVICE_PENALTY = 1.0e6
SOURCE_NODE = 0
SOURCE_BASE_STOCK = 10000.0

DEFAULT_DATA_DIR = Path(__file__).with_name("data")

DEFAULT_NETWORK = np.zeros((6, 6), dtype=int)
DEFAULT_NETWORK[0, 1] = 1
DEFAULT_NETWORK[1, 2] = 1
DEFAULT_NETWORK[1, 3] = 1
DEFAULT_NETWORK[3, 4] = 1
DEFAULT_NETWORK[3, 5] = 1

DEFAULT_LEAD_TIME = np.array([0, 3, 4, 4, 2, 2], dtype=float)
DEFAULT_SERVICE_TARGET = np.array([0.0, 0.95, 0.95, 0.0, 0.95, 0.95], dtype=float)

REFERENCE_EXCESS_INVENTORY_GUESS = np.array(
    [2000, 350, 700, 150, 400],
    dtype=float,
)
REFERENCE_ROP_GUESS = np.array(
    [1000, 250, 200, 150, 200],
    dtype=float,
)
REFERENCE_INITIAL_GUESS = np.concatenate(
    [REFERENCE_EXCESS_INVENTORY_GUESS, REFERENCE_ROP_GUESS]
)

PUBLISHED_BACKORDER_SOLUTION = np.array(
    [
        1931.86354959,
        377.72872864,
        736.96821740,
        158.17153186,
        398.36439462,
        804.25245856,
        257.37396920,
        212.75370283,
        148.60471354,
        199.13092160,
    ],
    dtype=float,
)
PUBLISHED_LOST_SALES_SOLUTION = np.array(
    [
        1786.42510894,
        366.71494378,
        738.71965624,
        149.67503675,
        404.78298398,
        729.16331933,
        276.13613632,
        197.82782439,
        158.76006432,
        219.87721725,
    ],
    dtype=float,
)


@dataclass(frozen=True)
class ReferencePolicyParameters:
    """Base-stock and reorder-point arrays, including the source node."""

    reorder_point: np.ndarray
    base_stock: np.ndarray

    @property
    def initial_inventory(self) -> np.ndarray:
        """Return the reference initial inventory, 90% of base stock."""
        return 0.9 * self.base_stock


def reference_network() -> np.ndarray:
    """Return the six-node reference supply-chain adjacency matrix."""
    return DEFAULT_NETWORK.copy()


def upstream_nodes(network: np.ndarray) -> list[int | None]:
    """Return each node's single upstream provider from an adjacency matrix."""
    upstream: list[int | None] = []
    for node in range(network.shape[1]):
        providers = np.flatnonzero(network[:, node])
        if len(providers) == 0:
            upstream.append(None)
        elif len(providers) == 1:
            upstream.append(int(providers[0]))
        else:
            raise ValueError(f"node {node} has multiple upstream providers")
    return upstream


def policy_parameters_from_guess(
    initial_guess,
    *,
    source_base_stock: float = SOURCE_BASE_STOCK,
) -> ReferencePolicyParameters:
    """Convert the reference optimizer vector into policy parameter arrays.

    The first five values are excess inventory over the reorder point. The
    final five values are reorder points. The source node is inserted at index
    zero, matching the original scripts.
    """
    guess = np.asarray(initial_guess, dtype=float)
    if guess.ndim != 1 or guess.shape[0] % 2 != 0:
        raise ValueError("initial_guess must contain excess and ROP values")

    n_stocking_nodes = guess.shape[0] // 2
    excess_inventory = guess[:n_stocking_nodes]
    reorder_point = guess[n_stocking_nodes:]
    base_stock = excess_inventory + reorder_point

    return ReferencePolicyParameters(
        reorder_point=np.insert(reorder_point, 0, 0.0),
        base_stock=np.insert(base_stock, 0, float(source_base_stock)),
    )
