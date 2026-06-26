import numpy as np
import pytest

from sda import ArrayScenarioLoader, ScenarioBatch


def test_scenario_batch_validates_exogenous_shapes():
    with pytest.raises(ValueError, match="batch size"):
        ScenarioBatch(
            initial_state=np.array([1, 2]),
            exogenous={"demand": np.ones((3, 2))},
            scenario_ids=[1, 2],
        )

    with pytest.raises(ValueError, match="horizon"):
        ScenarioBatch(
            initial_state=np.array([1, 2]),
            exogenous={
                "demand": np.ones((2, 3)),
                "price": np.ones((2, 4)),
            },
            scenario_ids=[1, 2],
        )


def test_array_scenario_loader_batches_arrays():
    loader = ArrayScenarioLoader(
        initial_state=np.array([10, 11, 12]),
        exogenous={"demand": np.arange(12).reshape(3, 4)},
        batch_size=2,
    )

    batches = list(loader)

    assert len(batches) == 2
    assert batches[0].scenario_ids == [0, 1]
    assert batches[0].initial_state.tolist() == [10, 11]
    assert batches[0].horizon == 4
    assert batches[1].scenario_ids == [2]
    assert batches[1].exogenous["demand"].shape == (1, 4)


def test_array_scenario_loader_broadcasts_scalar_initial_state():
    loader = ArrayScenarioLoader(
        initial_state=50,
        exogenous={"demand": np.ones((3, 2))},
        batch_size=2,
    )

    assert [batch.initial_state.tolist() for batch in loader] == [[50, 50], [50]]
