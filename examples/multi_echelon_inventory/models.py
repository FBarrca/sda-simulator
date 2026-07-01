from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import simpy

from examples.multi_echelon_inventory.domain import (
    DEFAULT_LEAD_TIME,
    DEFAULT_NETWORK,
    DEFAULT_SERVICE_TARGET,
    REFERENCE_SERVICE_PENALTY,
    SOURCE_NODE,
    upstream_nodes,
)
from sda import Policy, Recorder, SDAModel, ScenarioSpec

ServiceMode = Literal["lost_sales", "backorder"]


@dataclass
class ReplenishmentOrder:
    """Order placed by one stocking node to its upstream node."""

    requester: int
    quantity: float


@dataclass
class ScenarioNetworkState:
    """Full SimPy-backed network state for one simulated replication."""

    env: simpy.Environment
    model: "MultiEchelonInventoryModel"
    rng: np.random.RandomState
    historical_demand: np.ndarray
    lead_time_delay_history: np.ndarray
    horizon: int
    day: int = 0
    nodes: list["FacilityState"] = field(default_factory=list)
    last_demand: np.ndarray = field(default_factory=lambda: np.empty(0))
    last_shipped: np.ndarray = field(default_factory=lambda: np.empty(0))
    last_lost_sales: np.ndarray = field(default_factory=lambda: np.empty(0))
    last_backorder: np.ndarray = field(default_factory=lambda: np.empty(0))
    last_on_hand_inventory: np.ndarray = field(default_factory=lambda: np.empty(0))
    last_inventory_position: np.ndarray = field(default_factory=lambda: np.empty(0))

    def reset_step_observations(self) -> None:
        """Clear arrays populated by the next SimPy day."""
        num_nodes = self.model.num_nodes
        self.last_demand = np.zeros(num_nodes, dtype=float)
        self.last_shipped = np.zeros(num_nodes, dtype=float)
        self.last_lost_sales = np.zeros(num_nodes, dtype=float)
        self.last_backorder = np.zeros(num_nodes, dtype=float)
        self.last_on_hand_inventory = np.zeros(num_nodes, dtype=float)
        self.last_inventory_position = np.zeros(num_nodes, dtype=float)


@dataclass
class FacilityState:
    """SimPy process bundle for one stocking facility."""

    scenario: ScenarioNetworkState
    node_index: int
    is_source: bool
    on_hand_inventory: float
    inventory_position: float
    default_lead_time: float
    historical_demand: np.ndarray
    order_queue: list[ReplenishmentOrder] = field(default_factory=list)
    total_demand: float = 0.0
    total_shipped: float = 0.0
    total_backorder: float = 0.0
    total_late_sales: float = 0.0
    on_hand_monitor: list[float] = field(default_factory=list)

    @property
    def env(self) -> simpy.Environment:
        """Return the scenario's SimPy environment."""
        return self.scenario.env

    @property
    def average_on_hand(self) -> float:
        """Return the mean monitored on-hand inventory."""
        if not self.on_hand_monitor:
            return 0.0
        return float(np.mean(self.on_hand_monitor))

    def start(self) -> None:
        """Start the facility's reference SimPy processes."""
        self.env.process(self.check_inventory())
        self.env.process(self.prepare_replenishment())
        self.env.process(self.serve_customer())

    def check_inventory(self):
        """Place a replenishment order when inventory position reaches ROP."""
        while True:
            yield self.env.timeout(1.0)
            placed, quantity = self.scenario.model.order_for_node(
                self.scenario,
                self.node_index,
            )
            if placed:
                upstream = self.scenario.model.upstream[self.node_index]
                if upstream is None:
                    continue
                order = ReplenishmentOrder(
                    requester=self.node_index,
                    quantity=float(quantity),
                )
                self.scenario.nodes[upstream].order_queue.append(order)
                self.inventory_position += float(quantity)

    def prepare_replenishment(self):
        """Prepare queued replenishment orders and launch delivery processes."""
        while True:
            if len(self.order_queue) > 0:
                order = self.order_queue.pop(0)

                shipment = min(order.quantity, self.on_hand_inventory)
                if not self.is_source:
                    self.inventory_position -= shipment
                    self.on_hand_inventory -= shipment

                remaining_order = order.quantity - shipment
                if remaining_order:
                    while not self.on_hand_inventory >= remaining_order:
                        yield self.env.timeout(1.0)
                    if not self.is_source:
                        self.inventory_position -= remaining_order
                        self.on_hand_inventory -= remaining_order
                self.env.process(self.ship(order.quantity, order.requester))
            else:
                yield self.env.timeout(1.0)

    def ship(self, quantity: float, requester: int):
        """Deliver a replenishment after empirical lead time."""
        delay = self.scenario.rng.choice(
            self.scenario.lead_time_delay_history,
            replace=True,
        )
        lead_time = self.scenario.model.default_lead_time[requester] + delay
        yield self.env.timeout(lead_time)
        self.scenario.nodes[requester].on_hand_inventory += float(quantity)

    def serve_customer(self):
        """Serve daily customer demand using lost-sales or backorder accounting."""
        while True:
            self._monitor_on_hand()
            yield self.env.timeout(1.0)
            demand = float(self.scenario.rng.choice(self.historical_demand, replace=True))
            self.total_demand += demand
            self.scenario.last_demand[self.node_index] = demand

            if self.scenario.model.service_mode == "lost_sales":
                self._serve_lost_sales(demand)
            else:
                self._serve_backorder(demand)

    def _serve_lost_sales(self, demand: float) -> None:
        shipment = min(demand, self.on_hand_inventory)
        lost_sales = demand - shipment
        self.total_shipped += shipment
        self.on_hand_inventory -= shipment
        self.inventory_position -= shipment
        self.scenario.last_shipped[self.node_index] = shipment
        self.scenario.last_lost_sales[self.node_index] = lost_sales

    def _serve_backorder(self, demand: float) -> None:
        shipment = min(demand + self.total_backorder, self.on_hand_inventory)
        self.on_hand_inventory -= shipment
        self.inventory_position -= shipment
        backorder = demand - shipment
        self.total_backorder += backorder
        late_sales = max(0.0, backorder)
        self.total_late_sales += late_sales
        self.scenario.last_shipped[self.node_index] = shipment
        self.scenario.last_backorder[self.node_index] = self.total_backorder
        self.scenario.last_lost_sales[self.node_index] = late_sales

    def _monitor_on_hand(self) -> None:
        self.on_hand_monitor.append(self.on_hand_inventory)
        self.scenario.last_on_hand_inventory[self.node_index] = self.on_hand_inventory
        self.scenario.last_inventory_position[self.node_index] = (
            self.inventory_position
        )


class MultiEchelonInventoryModel(SDAModel):
    """Reference SimPy multi-echelon inventory dynamics in SDA model form."""

    def __init__(
        self,
        policy: Policy,
        *,
        network=DEFAULT_NETWORK,
        default_lead_time=DEFAULT_LEAD_TIME,
        initial_inventory=None,
        service_mode: ServiceMode = "lost_sales",
        source_node: int = SOURCE_NODE,
        service_target=DEFAULT_SERVICE_TARGET,
        penalty_weight: float = REFERENCE_SERVICE_PENALTY,
        record_daily_metrics: bool = False,
    ) -> None:
        """Create the model with lost-sales or backorder service accounting."""
        super().__init__(policy)
        self.network = np.asarray(network, dtype=int)
        if self.network.ndim != 2 or self.network.shape[0] != self.network.shape[1]:
            raise ValueError("network must be a square adjacency matrix")
        self.num_nodes = int(self.network.shape[0])
        self.upstream = upstream_nodes(self.network)
        self.default_lead_time = np.asarray(default_lead_time, dtype=float)
        if self.default_lead_time.shape != (self.num_nodes,):
            raise ValueError("default_lead_time length must match network size")

        if service_mode not in {"lost_sales", "backorder"}:
            raise ValueError("service_mode must be 'lost_sales' or 'backorder'")
        self.service_mode: ServiceMode = service_mode
        self.source_node = int(source_node)
        self.service_target = np.asarray(service_target, dtype=float)
        if self.service_target.shape != (self.num_nodes,):
            raise ValueError("service_target length must match network size")
        self.penalty_weight = float(penalty_weight)
        self.record_daily_metrics = bool(record_daily_metrics)

        if initial_inventory is None:
            if not hasattr(policy, "base_stock"):
                raise TypeError(
                    "initial_inventory is required when policy has no base_stock"
                )
            initial_inventory = 0.9 * np.asarray(policy.base_stock, dtype=float)
        self.initial_inventory = np.asarray(initial_inventory, dtype=float)
        if self.initial_inventory.shape != (self.num_nodes,):
            raise ValueError("initial_inventory length must match network size")

    def build(
        self,
        env: simpy.Environment,
        scenario: ScenarioSpec,
        recorder: Recorder,
    ) -> ScenarioNetworkState:
        """Register the reference facility processes for one scenario."""
        required = {"horizon", "historical_demand", "lead_time_delay_history"}
        missing = required.difference(scenario.data)
        if missing:
            raise ValueError(f"scenario data missing keys: {sorted(missing)}")

        historical_demand = np.asarray(
            scenario.data["historical_demand"],
            dtype=float,
        )
        lead_time_delay_history = np.asarray(
            scenario.data["lead_time_delay_history"],
            dtype=float,
        )
        seed = scenario.seed if scenario.seed is not None else int(scenario.scenario_id)
        horizon = int(scenario.data["horizon"])

        state = ScenarioNetworkState(
            env=env,
            model=self,
            rng=np.random.RandomState(int(seed)),
            historical_demand=historical_demand,
            lead_time_delay_history=lead_time_delay_history,
            horizon=horizon,
        )
        state.reset_step_observations()
        for node_index, initial_inventory in enumerate(self.initial_inventory):
            facility = FacilityState(
                scenario=state,
                node_index=node_index,
                is_source=node_index == self.source_node,
                on_hand_inventory=float(initial_inventory),
                inventory_position=float(initial_inventory),
                default_lead_time=float(self.default_lead_time[node_index]),
                historical_demand=(
                    np.zeros(100, dtype=float)
                    if node_index == self.source_node
                    else historical_demand[:, node_index - 1]
                ),
            )
            state.nodes.append(facility)
            facility.start()
        if self.record_daily_metrics:
            env.process(self._record_days(state, recorder))
        return state

    def finalize(
        self,
        state: ScenarioNetworkState,
        scenario: ScenarioSpec,
        recorder: Recorder,
    ) -> None:
        """Log final reference objective components for the scenario."""
        del scenario
        average_on_hand = self.average_on_hand([state])[0]
        service_level = self.service_level([state])[0]
        non_source_average = average_on_hand.copy()
        non_source_average[self.source_node] = 0.0
        shortfall = np.maximum(0.0, self.service_target - service_level)
        service_penalty = self.penalty_weight * float(np.sum(shortfall))

        recorder.trajectory(
            "reference_average_on_hand",
            float(np.sum(non_source_average)),
        )
        recorder.trajectory("reference_service_penalty", service_penalty)
        recorder.trajectory(
            "replication_objective",
            float(np.sum(non_source_average) + service_penalty),
        )
        for node_index, value in enumerate(service_level):
            recorder.trajectory(f"service_level_node_{node_index}", value)
        for node_index, value in enumerate(average_on_hand):
            recorder.trajectory(f"average_on_hand_node_{node_index}", value)

    def _record_days(
        self,
        scenario: ScenarioNetworkState,
        recorder: Recorder,
    ):
        """Emit daily event diagnostics after each reference day completes."""
        while scenario.day < scenario.horizon:
            yield scenario.env.timeout(1.0)
            scenario.day += 1
            self._log_day(scenario, recorder)
            scenario.reset_step_observations()

    def _log_day(
        self,
        scenario: ScenarioNetworkState,
        recorder: Recorder,
    ) -> None:
        on_hand = scenario.last_on_hand_inventory.copy()
        on_hand[self.source_node] = 0.0
        recorder.log("total_on_hand", float(np.sum(on_hand)))
        for node_index in range(self.num_nodes):
            tags = {"node": str(node_index)}
            recorder.log(
                f"demand_node_{node_index}",
                scenario.last_demand[node_index],
                tags=tags,
            )
            recorder.log(
                f"shipped_node_{node_index}",
                scenario.last_shipped[node_index],
                tags=tags,
            )
            recorder.log(
                f"lost_sales_node_{node_index}",
                scenario.last_lost_sales[node_index],
                tags=tags,
            )
            recorder.log(
                f"backorder_node_{node_index}",
                scenario.last_backorder[node_index],
                tags=tags,
            )
            recorder.log(
                f"on_hand_inventory_node_{node_index}",
                scenario.last_on_hand_inventory[node_index],
                tags=tags,
            )
            recorder.log(
                f"inventory_position_node_{node_index}",
                scenario.last_inventory_position[node_index],
                tags=tags,
            )

    def order_for_node(
        self,
        scenario: ScenarioNetworkState,
        node_index: int,
    ) -> tuple[bool, float]:
        """Ask the policy whether one SimPy facility should order."""
        if hasattr(self.policy, "order_for_node"):
            return self.policy.order_for_node(scenario, node_index)

        decision = self.policy.act([scenario], scenario.env, [])
        quantity, placed = _decision_arrays(
            decision,
            batch_size=1,
            num_nodes=self.num_nodes,
        )
        return bool(placed[0, node_index]), float(quantity[0, node_index])

    def average_on_hand(self, states) -> np.ndarray:
        """Return average on-hand inventory by scenario and node."""
        values = np.zeros((len(states), self.num_nodes), dtype=float)
        for scenario_index, scenario in enumerate(states):
            for node_index, node in enumerate(scenario.nodes):
                if node_index == self.source_node:
                    values[scenario_index, node_index] = 0.0
                else:
                    values[scenario_index, node_index] = node.average_on_hand
        return values

    def service_level(self, states) -> np.ndarray:
        """Return cumulative service level by scenario and node."""
        values = np.zeros((len(states), self.num_nodes), dtype=float)
        for scenario_index, scenario in enumerate(states):
            for node_index, node in enumerate(scenario.nodes):
                denominator = node.total_demand + 1.0e-5
                if self.service_mode == "lost_sales":
                    values[scenario_index, node_index] = (
                        node.total_shipped / denominator
                    )
                else:
                    values[scenario_index, node_index] = (
                        1.0 - node.total_late_sales / denominator
                    )
        return values


def _decision_arrays(
    decision,
    *,
    batch_size: int,
    num_nodes: int,
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(decision, dict):
        quantity = np.asarray(decision["order_quantity"], dtype=float)
        placed = np.asarray(
            decision.get("order_placed", quantity != 0.0),
            dtype=bool,
        )
    else:
        quantity = np.asarray(decision, dtype=float)
        placed = quantity != 0.0

    expected_shape = (batch_size, num_nodes)
    if quantity.shape != expected_shape:
        raise ValueError(f"order_quantity must have shape {expected_shape}")
    if placed.shape != expected_shape:
        raise ValueError(f"order_placed must have shape {expected_shape}")
    return quantity, placed
