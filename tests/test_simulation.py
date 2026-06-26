import numpy as np

from sda import (
    ArrayScenarioLoader,
    Policy,
    SDAModel,
    Simulator,
    StepCostMetric,
    TotalCostMetric,
)


class ZeroPolicy(Policy):
    def act(self, state, t, history):
        return np.zeros_like(state, dtype=float)


class DemandAccumulationModel(SDAModel):
    def transition(self, state, decision, exogenous, t):
        return np.asarray(state, dtype=float) + np.asarray(exogenous["demand"], dtype=float)

    def cost(self, state, decision, exogenous, next_state, t):
        return np.asarray(exogenous["demand"], dtype=float)


def make_loader(demand):
    return ArrayScenarioLoader(
        initial_state=np.zeros(demand.shape[0]),
        exogenous={"demand": demand},
        batch_size=2,
    )


def test_simulator_runs_deterministic_model_and_logs_metric_shapes():
    demand = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ],
        dtype=float,
    )
    result = Simulator(metrics=[StepCostMetric(), TotalCostMetric()]).evaluate(
        DemandAccumulationModel(ZeroPolicy()),
        make_loader(demand),
    )

    assert result.metric("total_cost").values().tolist() == [6, 15, 24]
    assert len(result.metric("total_cost").values()) == 3
    assert len(result.metric("step_cost").values()) == 9
    assert result.metric("step_cost").at_time(1).values().tolist() == [2, 5, 8]


def test_scenarios_models_and_metrics_are_swappable():
    first_loader = make_loader(np.ones((2, 2)))
    second_loader = make_loader(np.full((2, 3), 2.0))
    model = DemandAccumulationModel(ZeroPolicy())

    only_total = Simulator(metrics=[TotalCostMetric()]).evaluate(model, first_loader)
    with_steps = Simulator(metrics=[StepCostMetric(), TotalCostMetric()]).evaluate(
        model,
        second_loader,
    )

    assert only_total.metric("total_cost").values().tolist() == [2, 2]
    assert only_total.metric("step_cost").values().size == 0
    assert with_steps.metric("total_cost").values().tolist() == [6, 6]
    assert with_steps.metric("step_cost").values().tolist() == [2, 2, 2, 2, 2, 2]
