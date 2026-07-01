from __future__ import annotations

from collections.abc import Iterator, Sequence
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np

from examples.multi_echelon_inventory.domain import (
    DEFAULT_DATA_DIR,
    DEFAULT_NETWORK,
    REFERENCE_HORIZON,
    REFERENCE_REPLICATIONS,
)
from sda import DataModule, ScenarioBatch, ScenarioSpec


class MultiEchelonInventoryDataModule(DataModule):
    """Empirical bootstrap data for the reference multi-echelon problem."""

    def __init__(
        self,
        *,
        horizon: int = REFERENCE_HORIZON,
        n_scenarios: int = REFERENCE_REPLICATIONS,
        batch_size: int | None = None,
        data_dir: str | Path = DEFAULT_DATA_DIR,
        historical_demand: Any | None = None,
        lead_time_delay_history: Any | None = None,
        scenario_seeds: Sequence[int] | None = None,
        scenario_ids: Sequence[int] | None = None,
        network: Any = DEFAULT_NETWORK,
    ) -> None:
        """Create the reference data module.

        Historical demand contains all nodes except the source node. Lead-time
        delay history is sampled for replenishment shipments.
        """
        self.horizon = _positive_int("horizon", horizon)
        self.n_scenarios = _positive_int("n_scenarios", n_scenarios)
        self.batch_size = (
            self.n_scenarios
            if batch_size is None
            else _positive_int("batch_size", batch_size)
        )
        self.data_dir = Path(data_dir)
        self.network = np.asarray(network, dtype=int)
        if self.network.ndim != 2 or self.network.shape[0] != self.network.shape[1]:
            raise ValueError("network must be a square adjacency matrix")
        self.num_nodes = int(self.network.shape[0])

        self._historical_demand = (
            None if historical_demand is None else _prepare_historical_demand(
                historical_demand,
                self.num_nodes,
            )
        )
        self._lead_time_delay_history = (
            None
            if lead_time_delay_history is None
            else _prepare_lead_time_delay_history(lead_time_delay_history)
        )
        if (self._historical_demand is None) != (self._lead_time_delay_history is None):
            raise ValueError(
                "provide both historical_demand and lead_time_delay_history, or neither"
            )

        self.scenario_seeds = _prepare_scenario_seeds(
            scenario_seeds,
            self.n_scenarios,
        )
        self.scenario_ids = _prepare_scenario_ids(
            scenario_ids,
            self.scenario_seeds,
            self.n_scenarios,
        )

    def prepare_data(self) -> None:
        """Load the reference CSV data when arrays were not supplied."""
        if self._historical_demand is not None:
            return

        demand_path = self.data_dir / "demandData.csv"
        lead_time_path = self.data_dir / "leadTimeExtraDays.csv"
        self._historical_demand = _prepare_historical_demand(
            np.loadtxt(demand_path, delimiter=",", skiprows=1),
            self.num_nodes,
        )
        self._lead_time_delay_history = _prepare_lead_time_delay_history(
            np.loadtxt(lead_time_path, delimiter=","),
        )

    def setup(self, stage: str | None = None) -> None:
        """Load the empirical histories for the requested stage."""
        del stage
        self.prepare_data()
        assert self._historical_demand is not None
        assert self._lead_time_delay_history is not None

    def batches(self, stage: str = "evaluate") -> Iterator[ScenarioBatch]:
        """Yield deterministic seeded SimPy scenarios in batches."""
        del stage
        self.setup()

        assert self._historical_demand is not None
        assert self._lead_time_delay_history is not None
        for start in range(0, self.n_scenarios, self.batch_size):
            stop = min(start + self.batch_size, self.n_scenarios)
            scenarios = [
                ScenarioSpec(
                    scenario_id=int(self.scenario_ids[index]),
                    end_time=float(self.horizon + 1),
                    data={
                        "horizon": self.horizon,
                        "historical_demand": self._historical_demand,
                        "lead_time_delay_history": self._lead_time_delay_history,
                    },
                    seed=int(self.scenario_seeds[index]),
                )
                for index in range(start, stop)
            ]
            yield ScenarioBatch(scenarios)


def load_reference_history(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    *,
    network: Any = DEFAULT_NETWORK,
) -> tuple[np.ndarray, np.ndarray]:
    """Load historical demand and lead-time-delay arrays from CSV files."""
    network_array = np.asarray(network)
    data_path = Path(data_dir)
    demand = _prepare_historical_demand(
        np.loadtxt(data_path / "demandData.csv", delimiter=",", skiprows=1),
        int(network_array.shape[0]),
    )
    lead_time_delay = _prepare_lead_time_delay_history(
        np.loadtxt(data_path / "leadTimeExtraDays.csv", delimiter=","),
    )
    return demand, lead_time_delay


def _prepare_historical_demand(values: Any, num_nodes: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError("historical_demand must be a two-dimensional array")
    if array.shape[0] == 0:
        raise ValueError("historical_demand must contain observations")
    if array.shape[1] != num_nodes - 1:
        raise ValueError(
            "historical_demand must contain one column for each non-source node"
        )
    return array


def _prepare_lead_time_delay_history(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.shape[0] == 0:
        raise ValueError("lead_time_delay_history must contain observations")
    return array


def _prepare_scenario_seeds(
    scenario_seeds: Sequence[int] | None,
    n_scenarios: int,
) -> np.ndarray:
    if scenario_seeds is None:
        return np.arange(n_scenarios, dtype=int)
    if len(scenario_seeds) != n_scenarios:
        raise ValueError("scenario_seeds length must match n_scenarios")
    return np.asarray(scenario_seeds, dtype=int)


def _prepare_scenario_ids(
    scenario_ids: Sequence[int] | None,
    scenario_seeds: np.ndarray,
    n_scenarios: int,
) -> np.ndarray:
    if scenario_ids is None:
        return scenario_seeds.copy()
    if len(scenario_ids) != n_scenarios:
        raise ValueError("scenario_ids length must match n_scenarios")
    return np.asarray(scenario_ids)


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return int(value)
