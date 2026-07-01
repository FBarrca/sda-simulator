from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from sda import DataModule, GeneratorDataModule, ScenarioBatch


class InventoryDataModule(DataModule):
    """Data module for Poisson-demand inventory futures."""

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

    def batches(self, stage: str = "evaluate") -> Iterator[ScenarioBatch]:
        del stage

        def demand_generator(*, rng, shape):
            return {
                "demand": rng.poisson(
                    self.demand_lambda,
                    size=shape,
                ).astype(float)
            }

        yield from GeneratorDataModule(
            demand_generator,
            horizon=self.horizon,
            n_scenarios=self.n_scenarios,
            batch_size=self.batch_size,
            initial_state=self.initial_inventory,
            seed=self.seed,
        ).batches()
