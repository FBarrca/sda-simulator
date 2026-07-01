"""Multi-echelon inventory optimization example for the SDA framework.

Reproduces the network, base-stock policy, data-resampling approach, and
objective from Agarwal, A. (2019), "Multi-echelon Supply Chain Inventory
Planning using Simulation-Optimization with Data Resampling"
(arXiv:1901.00090), framed in SDA's state/policy/model/data terms.
"""

from examples.multi_echelon_inventory.data import (
    MultiEchelonInventoryDataModule,
    load_reference_history,
)
from examples.multi_echelon_inventory.domain import (
    DEFAULT_LEAD_TIME,
    DEFAULT_NETWORK,
    DEFAULT_SERVICE_TARGET,
    PUBLISHED_BACKORDER_SOLUTION,
    PUBLISHED_LOST_SALES_SOLUTION,
    REFERENCE_HORIZON,
    REFERENCE_INITIAL_GUESS,
    REFERENCE_REPLICATIONS,
    REFERENCE_SERVICE_PENALTY,
    policy_parameters_from_guess,
    reference_network,
)
from examples.multi_echelon_inventory.metrics import (
    ReferenceObjectiveSummary,
    reference_metric_names,
    reference_metrics,
    summarize_reference_result,
)
from examples.multi_echelon_inventory.models import (
    FacilityState,
    MultiEchelonInventoryModel,
    ScenarioNetworkState,
)
from examples.multi_echelon_inventory.main import (
    PolicyEvaluation,
    build_data,
    build_evaluation,
    build_model,
    build_policy,
    build_result,
)
from examples.multi_echelon_inventory.optimization import (
    evaluate_reference_policy,
    get_objective,
)
from examples.multi_echelon_inventory.policies import BaseStockReorderPolicy

__all__ = [
    "BaseStockReorderPolicy",
    "DEFAULT_LEAD_TIME",
    "DEFAULT_NETWORK",
    "DEFAULT_SERVICE_TARGET",
    "FacilityState",
    "MultiEchelonInventoryDataModule",
    "MultiEchelonInventoryModel",
    "PUBLISHED_BACKORDER_SOLUTION",
    "PUBLISHED_LOST_SALES_SOLUTION",
    "PolicyEvaluation",
    "REFERENCE_HORIZON",
    "REFERENCE_INITIAL_GUESS",
    "REFERENCE_REPLICATIONS",
    "REFERENCE_SERVICE_PENALTY",
    "ReferenceObjectiveSummary",
    "ScenarioNetworkState",
    "build_data",
    "build_evaluation",
    "build_model",
    "build_policy",
    "build_result",
    "evaluate_reference_policy",
    "get_objective",
    "load_reference_history",
    "policy_parameters_from_guess",
    "reference_metric_names",
    "reference_metrics",
    "reference_network",
    "summarize_reference_result",
]
