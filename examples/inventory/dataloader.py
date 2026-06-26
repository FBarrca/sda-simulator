from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from sda.data import ScenarioBatch, ScenarioLoader


class InventoryScenarioLoader(ScenarioLoader):
    """Poisson-demand inventory futures."""

    def __init__(
        self,
        horizon: int,
        n_scenarios: int,
        batch_size: int,
        initial_inventory: float,
        demand_lambda: float,
        seed: int | None = None,
    ) -> None:
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if n_scenarios <= 0:
            raise ValueError("n_scenarios must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if demand_lambda < 0:
            raise ValueError("demand_lambda must be non-negative")

        self.horizon = int(horizon)
        self.n_scenarios = int(n_scenarios)
        self.batch_size = int(batch_size)
        self.initial_inventory = float(initial_inventory)
        self.demand_lambda = float(demand_lambda)
        self.seed = seed

    def __iter__(self) -> Iterator[ScenarioBatch]:
        rng = np.random.default_rng(self.seed)
        for start in range(0, self.n_scenarios, self.batch_size):
            stop = min(start + self.batch_size, self.n_scenarios)
            size = stop - start
            yield ScenarioBatch(
                initial_state=np.full(size, self.initial_inventory, dtype=float),
                exogenous={
                    "demand": rng.poisson(
                        self.demand_lambda,
                        size=(size, self.horizon),
                    ).astype(float)
                },
                scenario_ids=list(range(start, stop)),
            )
