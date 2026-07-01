import numpy as np
import pytest

from examples.logistics import (
    Assignment,
    GreedyPolicy,
    LookaheadRolloutPolicy,
    LogisticsDataModule,
    LogisticsModel,
    MilpPolicy,
    NearestFeasiblePolicy,
    Order,
    PriorityDeadlinePolicy,
    PriorityPolicy,
    RandomPolicy,
    RiskAwareDispatchPolicy,
    initial_logistics_state,
    synthetic_history,
)
from examples.logistics.main import build_result
from examples.logistics.network import CUSTOMERS, VEHICLE_INDEX, VEHICLES, WAREHOUSES


def test_synthetic_history_is_deterministic_and_has_weekend_dip():
    first = synthetic_history(days=365, seed=7)
    second = synthetic_history(days=365, seed=7)

    assert first.orders[42] == second.orders[42]
    np.testing.assert_allclose(first.traffic_multiplier, second.traffic_multiplier)
    np.testing.assert_array_equal(first.vehicle_outages, second.vehicle_outages)

    daily_units = np.asarray(
        [sum(order.quantity for order in day_orders) for day_orders in first.orders],
        dtype=float,
    )
    day_of_week = np.arange(365) % 7
    weekday_mean = daily_units[day_of_week < 5].mean()
    weekend_mean = daily_units[day_of_week >= 5].mean()

    assert weekend_mean < weekday_mean * 0.7


def test_logistics_data_module_batches_and_bootstraps_seven_day_blocks():
    data = LogisticsDataModule(
        horizon=10,
        n_scenarios=5,
        batch_size=2,
        history_days=35,
        seed=11,
    )

    batches = list(data.batches())

    assert [batch.batch_size for batch in batches] == [2, 2, 1]
    first_batch = batches[0]
    first_scenario = first_batch.scenarios[0]
    assert first_batch.end_time == 10
    assert len(first_batch.scenarios) == 2
    assert first_scenario.data["orders"].shape == (10,)
    assert first_scenario.data["traffic_multiplier"].shape == (
        10,
        len(WAREHOUSES),
        len(CUSTOMERS),
    )
    assert first_scenario.data["vehicle_outages"].shape == (10, len(VEHICLES))
    np.testing.assert_array_equal(
        np.diff(first_scenario.data["history_day_index"][:7]),
        np.ones(6, dtype=int),
    )

    repeated = next(
        iter(
            LogisticsDataModule(
                horizon=10,
                n_scenarios=5,
                batch_size=2,
                history_days=35,
                seed=11,
            ).batches()
        )
    )
    np.testing.assert_array_equal(
        first_scenario.data["history_day_index"],
        repeated.scenarios[0].data["history_day_index"],
    )
    assert first_scenario.data["orders"][0] == repeated.scenarios[0].data["orders"][0]


def test_logistics_model_enforces_assignment_feasibility():
    state = initial_logistics_state()
    state.inventory["W_BARCELONA"]["PHARMA"] = 1
    state.pending_orders = (
        Order(1, 0, "C_MADRID_CENTRO", "AMBIENT_FOOD", 5, 1, 3),
        Order(2, 0, "C_MADRID_CENTRO", "AMBIENT_FOOD", 4, 1, 3),
        Order(3, 0, "C_MADRID_CENTRO", "AMBIENT_FOOD", 40, 1, 3),
        Order(4, 0, "C_MADRID_CENTRO", "AMBIENT_FOOD", 5, 1, 3),
        Order(5, 0, "C_BARCELONA_PORT", "PHARMA", 2, 3, 1),
    )
    assignments = (
        Assignment(1, "W_MADRID", "V_MAD_1"),
        Assignment(2, "W_MADRID", "V_MAD_1"),
        Assignment(3, "W_MADRID", "V_MAD_2"),
        Assignment(4, "W_MADRID", "V_MAD_2"),
        Assignment(5, "W_BARCELONA", "V_BAR_2"),
    )
    orders = np.empty(1, dtype=object)
    orders[0] = ()
    outages = np.zeros((1, len(VEHICLES)), dtype=bool)
    outages[0, VEHICLE_INDEX["V_MAD_2"]] = True
    model = LogisticsModel(GreedyPolicy())

    next_state, info, cost = model._transition_one(
        state=state,
        assignments=assignments,
        new_orders=tuple(orders[0]),
        traffic=np.ones((len(WAREHOUSES), len(CUSTOMERS))),
        outages=outages[0],
        t=0,
    )

    assert len(next_state.completed_orders) == 1
    assert next_state.completed_orders[0].order_id == 1
    assert next_state.inventory["W_MADRID"]["AMBIENT_FOOD"] == pytest.approx(555)
    assert {order.order_id for order in next_state.pending_orders} == {2, 3, 4, 5}
    assert info["invalid_assignment_count"] == pytest.approx(4.0)
    assert cost >= 4 * 20


@pytest.mark.parametrize(
    "policy",
    [
        RandomPolicy(seed=23),
        GreedyPolicy(),
        PriorityPolicy(),
        MilpPolicy(),
        LookaheadRolloutPolicy(),
        NearestFeasiblePolicy(),
        PriorityDeadlinePolicy(),
        RiskAwareDispatchPolicy(),
    ],
)
def test_logistics_policies_produce_well_formed_metrics(policy):
    horizon = 8
    n_scenarios = 12
    result = build_result(
        policy=policy,
        horizon=horizon,
        n_scenarios=n_scenarios,
        batch_size=6,
        seed=23,
    )

    assert len(result.metric("total_cost").values()) == n_scenarios
    assert result.metric("total_cost").percentile(95) >= 0
    assert result.metric("total_cost").cvar(0.95) >= 0
    assert len(result.metric("on_time_rate").values()) == n_scenarios * horizon
    assert 0 <= result.metric("on_time_rate").mean() <= 1
    assert 0 <= result.metric("priority_weighted_on_time_rate").mean() <= 1
    assert result.metric("late_cost").min() >= 0
    assert result.metric("dispatch_cost").min() >= 0
    assert result.metric("pending_backlog").min() >= 0
    assert result.metric("dispatched_order_count").min() >= 0
    assert 0 <= result.metric("vehicle_utilization").mean() <= 1
