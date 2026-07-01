from __future__ import annotations

from examples.multi_echelon_inventory.data import MultiEchelonInventoryDataModule
from examples.multi_echelon_inventory.domain import (
    DEFAULT_LEAD_TIME,
    DEFAULT_NETWORK,
    DEFAULT_SERVICE_TARGET,
    REFERENCE_HORIZON,
    REFERENCE_INITIAL_GUESS,
    REFERENCE_REPLICATIONS,
)
from examples.multi_echelon_inventory.main import PolicyEvaluation, build_evaluation
from examples.multi_echelon_inventory.models import ServiceMode


def evaluate_reference_policy(
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
    record_daily_metrics: bool = False,
) -> PolicyEvaluation:
    """Evaluate one optimizer vector with the reference objective setup."""
    return build_evaluation(
        initial_guess,
        service_mode=service_mode,
        horizon=horizon,
        replications=replications,
        batch_size=batch_size,
        data=data,
        network=network,
        default_lead_time=default_lead_time,
        service_target=service_target,
        record_daily_metrics=record_daily_metrics,
    )


def get_objective(
    initial_guess=REFERENCE_INITIAL_GUESS,
    *,
    service_mode: ServiceMode = "lost_sales",
    horizon: int = REFERENCE_HORIZON,
    replications: int = REFERENCE_REPLICATIONS,
    batch_size: int | None = None,
    record_daily_metrics: bool = False,
) -> float:
    """Return the scalar black-box objective value for an optimizer."""
    return evaluate_reference_policy(
        initial_guess,
        service_mode=service_mode,
        horizon=horizon,
        replications=replications,
        batch_size=batch_size,
        record_daily_metrics=record_daily_metrics,
    ).summary.objective
