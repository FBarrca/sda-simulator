from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from examples.multi_echelon_inventory.data import MultiEchelonInventoryDataModule
from examples.multi_echelon_inventory.domain import (
    DEFAULT_DATA_DIR,
    DEFAULT_LEAD_TIME,
    DEFAULT_NETWORK,
    DEFAULT_SERVICE_TARGET,
    PUBLISHED_BACKORDER_SOLUTION,
    PUBLISHED_LOST_SALES_SOLUTION,
    REFERENCE_HORIZON,
    REFERENCE_INITIAL_GUESS,
    REFERENCE_REPLICATIONS,
)
from examples.multi_echelon_inventory.metrics import (
    ReferenceObjectiveSummary,
    summarize_reference_result,
)
from examples.multi_echelon_inventory.models import (
    MultiEchelonInventoryModel,
    ServiceMode,
)
from examples.multi_echelon_inventory.policies import BaseStockReorderPolicy
from sda import SimulationResult, evaluate


@dataclass(frozen=True)
class PolicyEvaluation:
    """Result bundle for one reference policy evaluation."""

    result: SimulationResult
    summary: ReferenceObjectiveSummary
    policy: BaseStockReorderPolicy


def build_data(
    *,
    horizon: int = REFERENCE_HORIZON,
    n_scenarios: int = REFERENCE_REPLICATIONS,
    batch_size: int | None = None,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    historical_demand: Any | None = None,
    lead_time_delay_history: Any | None = None,
    scenario_seeds=None,
    scenario_ids=None,
    network=DEFAULT_NETWORK,
) -> MultiEchelonInventoryDataModule:
    """Build the empirical data module for the reference problem."""
    return MultiEchelonInventoryDataModule(
        horizon=horizon,
        n_scenarios=n_scenarios,
        batch_size=batch_size,
        data_dir=data_dir,
        historical_demand=historical_demand,
        lead_time_delay_history=lead_time_delay_history,
        scenario_seeds=scenario_seeds,
        scenario_ids=scenario_ids,
        network=network,
    )


def build_policy(
    initial_guess=REFERENCE_INITIAL_GUESS,
    *,
    service_mode: ServiceMode = "lost_sales",
    use_published_solution: bool = False,
    network=DEFAULT_NETWORK,
) -> BaseStockReorderPolicy:
    """Build the reference base-stock reorder policy."""
    guess = _selected_guess(
        initial_guess,
        service_mode=service_mode,
        use_published_solution=use_published_solution,
    )
    return BaseStockReorderPolicy.from_optimizer_guess(guess, network=network)


def build_model(
    *,
    policy: BaseStockReorderPolicy | None = None,
    initial_guess=REFERENCE_INITIAL_GUESS,
    service_mode: ServiceMode = "lost_sales",
    use_published_solution: bool = False,
    network=DEFAULT_NETWORK,
    default_lead_time=DEFAULT_LEAD_TIME,
    initial_inventory=None,
    service_target=DEFAULT_SERVICE_TARGET,
    record_daily_metrics: bool = False,
) -> MultiEchelonInventoryModel:
    """Build the SimPy-native model for the reference network."""
    network_array = np.asarray(network, dtype=int)
    model_policy = policy or build_policy(
        initial_guess,
        service_mode=service_mode,
        use_published_solution=use_published_solution,
        network=network_array,
    )
    return MultiEchelonInventoryModel(
        model_policy,
        network=network_array,
        default_lead_time=default_lead_time,
        initial_inventory=initial_inventory,
        service_mode=service_mode,
        service_target=service_target,
        record_daily_metrics=record_daily_metrics,
    )


def build_result(
    *,
    policy: BaseStockReorderPolicy | None = None,
    data: MultiEchelonInventoryDataModule | None = None,
    initial_guess=REFERENCE_INITIAL_GUESS,
    service_mode: ServiceMode = "lost_sales",
    horizon: int = REFERENCE_HORIZON,
    replications: int = REFERENCE_REPLICATIONS,
    batch_size: int | None = None,
    use_published_solution: bool = False,
    network=DEFAULT_NETWORK,
    default_lead_time=DEFAULT_LEAD_TIME,
    service_target=DEFAULT_SERVICE_TARGET,
    record_daily_metrics: bool = False,
) -> SimulationResult:
    """Evaluate the reference model using the standard ``evaluate`` flow."""
    network_array = np.asarray(network, dtype=int)
    model_policy = policy or build_policy(
        initial_guess,
        service_mode=service_mode,
        use_published_solution=use_published_solution,
        network=network_array,
    )
    model = build_model(
        policy=model_policy,
        service_mode=service_mode,
        network=network_array,
        default_lead_time=default_lead_time,
        service_target=service_target,
        record_daily_metrics=record_daily_metrics,
    )
    data_module = data or build_data(
        horizon=horizon,
        n_scenarios=replications,
        batch_size=batch_size,
        network=network_array,
    )
    return evaluate(model, data_module)


def build_evaluation(
    initial_guess=REFERENCE_INITIAL_GUESS,
    *,
    service_mode: ServiceMode = "lost_sales",
    horizon: int = REFERENCE_HORIZON,
    replications: int = REFERENCE_REPLICATIONS,
    batch_size: int | None = None,
    data: MultiEchelonInventoryDataModule | None = None,
    network=DEFAULT_NETWORK,
    default_lead_time=DEFAULT_LEAD_TIME,
    service_target=DEFAULT_SERVICE_TARGET,
    use_published_solution: bool = False,
    record_daily_metrics: bool = False,
) -> PolicyEvaluation:
    """Run the reference evaluation and summarize the objective."""
    network_array = np.asarray(network, dtype=int)
    policy = build_policy(
        initial_guess,
        service_mode=service_mode,
        use_published_solution=use_published_solution,
        network=network_array,
    )
    result = build_result(
        policy=policy,
        data=data,
        service_mode=service_mode,
        horizon=horizon,
        replications=replications,
        batch_size=batch_size,
        network=network_array,
        default_lead_time=default_lead_time,
        service_target=service_target,
        record_daily_metrics=record_daily_metrics,
    )
    summary = summarize_reference_result(result, service_target=service_target)
    return PolicyEvaluation(result=result, summary=summary, policy=policy)


def main(argv: list[str] | None = None) -> None:
    """Run the example from the command line."""
    parser = argparse.ArgumentParser(
        description="Evaluate the reference multi-echelon inventory policy in SDA.",
    )
    parser.add_argument(
        "--mode",
        choices=["lost_sales", "backorder"],
        default="lost_sales",
        help="Reference service accounting mode.",
    )
    parser.add_argument(
        "--replications",
        type=int,
        default=REFERENCE_REPLICATIONS,
        help="Number of seeded simulation replications.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Scenario batch size.",
    )
    parser.add_argument(
        "--published-solution",
        action="store_true",
        help="Evaluate the published SciPy final solution instead of the initial guess.",
    )
    parser.add_argument(
        "--daily-metrics",
        action="store_true",
        help="Record dense per-day diagnostic metrics.",
    )
    args = parser.parse_args(argv)

    evaluation = build_evaluation(
        service_mode=args.mode,
        replications=args.replications,
        batch_size=args.batch_size,
        use_published_solution=args.published_solution,
        record_daily_metrics=args.daily_metrics,
    )
    summary = evaluation.summary

    print(f"Mode: {args.mode}")
    print(f"Replications: {args.replications}")
    print(f"Objective: {summary.objective:.3f}")
    print(f"Average on-hand: {summary.average_on_hand:.3f}")
    print(f"Service penalty: {summary.service_penalty:.3f}")
    print(
        "Service levels: "
        + np.array2string(summary.service_level, precision=4, separator=", ")
    )
    print(
        "Reorder points: "
        + np.array2string(evaluation.policy.reorder_point, precision=3, separator=", ")
    )
    print(
        "Base stock: "
        + np.array2string(evaluation.policy.base_stock, precision=3, separator=", ")
    )


def _selected_guess(
    initial_guess,
    *,
    service_mode: ServiceMode,
    use_published_solution: bool,
) -> np.ndarray:
    if not use_published_solution:
        return np.asarray(initial_guess, dtype=float)
    if service_mode == "lost_sales":
        return PUBLISHED_LOST_SALES_SOLUTION
    return PUBLISHED_BACKORDER_SOLUTION


if __name__ == "__main__":
    main()
