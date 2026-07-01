from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def scenario_initial_state(value: Any, index: int, n_scenarios: int) -> Any:
    """Return the initial-state value for one scenario."""
    if isinstance(value, Mapping):
        return {
            key: scenario_initial_state(item, index, n_scenarios)
            for key, item in value.items()
        }

    array = np.asarray(value)
    if array.ndim == 0:
        return array.item()
    if array.shape[0] != n_scenarios:
        raise ValueError(
            "initial_state must be scalar, mapping of scalar/vector values, "
            "or have one entry per scenario"
        )
    return array[index]
