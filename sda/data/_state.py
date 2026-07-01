from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def slice_initial_state(value: Any, start: int, stop: int, n_scenarios: int) -> Any:
    """Slice or broadcast an initial-state value for one scenario batch."""
    if isinstance(value, Mapping):
        return {
            key: slice_initial_state(item, start, stop, n_scenarios)
            for key, item in value.items()
        }

    array = np.asarray(value)
    if array.ndim == 0:
        return np.full(stop - start, array.item())
    if array.shape[0] != n_scenarios:
        raise ValueError(
            "initial_state must be scalar, mapping of scalar/vector values, "
            "or have one entry per scenario"
        )
    return array[start:stop]
