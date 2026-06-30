import numpy as np
import pytest

from sda import (
    ArrayScenarioLoader,
    BootstrapScenarioLoader,
    CircularBlockBootstrap,
    CircularBlockBootstrapScenarioLoader,
    IIDBootstrap,
    IIDBootstrapScenarioLoader,
    MovingBlockBootstrap,
    MovingBlockBootstrapScenarioLoader,
    ScenarioBatch,
    StationaryBootstrap,
    StationaryBlockBootstrapScenarioLoader,
)


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


def test_iid_bootstrap_scenario_loader_samples_individual_observations():
    loader = IIDBootstrapScenarioLoader(
        initial_state=np.array([10, 11, 12]),
        history={
            "returns": np.arange(6),
            "volume": np.arange(6) + 100,
        },
        horizon=4,
        n_scenarios=3,
        batch_size=2,
        seed=23,
    )

    batches = list(loader)

    rng = np.random.default_rng(23)
    expected_first = rng.integers(0, 6, size=(2, 4), dtype=np.int64)
    expected_second = rng.integers(0, 6, size=(1, 4), dtype=np.int64)

    assert [batch.batch_size for batch in batches] == [2, 1]
    assert batches[0].initial_state.tolist() == [10, 11]
    assert batches[1].initial_state.tolist() == [12]
    np.testing.assert_array_equal(batches[0].exogenous["returns"], expected_first)
    np.testing.assert_array_equal(batches[0].exogenous["volume"], expected_first + 100)
    np.testing.assert_array_equal(batches[1].exogenous["returns"], expected_second)


def test_circular_block_bootstrap_scenario_loader_wraps_fixed_blocks():
    loader = CircularBlockBootstrapScenarioLoader(
        initial_state=0,
        history={"source_index": np.arange(4)},
        horizon=5,
        n_scenarios=2,
        batch_size=2,
        block_size=5,
        seed=7,
    )

    batch = next(iter(loader))

    assert batch.initial_state.tolist() == [0, 0]
    assert batch.exogenous["source_index"].shape == (2, 5)
    for sampled_path in batch.exogenous["source_index"]:
        np.testing.assert_array_equal(
            np.diff(sampled_path) % 4,
            np.ones(4, dtype=int),
        )


def test_moving_block_bootstrap_scenario_loader_uses_non_wrapping_blocks():
    loader = MovingBlockBootstrapScenarioLoader(
        initial_state=0,
        history={"source_index": np.arange(6)},
        horizon=7,
        n_scenarios=2,
        batch_size=2,
        block_size=3,
        seed=11,
    )

    batch = next(iter(loader))

    expected = np.array(
        [
            [0, 1, 2, 0, 1, 2, 3],
            [1, 2, 3, 2, 3, 4, 2],
        ],
        dtype=np.int64,
    )
    np.testing.assert_array_equal(batch.exogenous["source_index"], expected)
    assert np.all(batch.exogenous["source_index"] < 6)


def test_stationary_block_bootstrap_scenario_loader_uses_random_restarts():
    loader = StationaryBlockBootstrapScenarioLoader(
        initial_state=0,
        history={"source_index": np.arange(7)},
        horizon=6,
        n_scenarios=2,
        batch_size=2,
        average_block_size=3,
        seed=5,
    )

    batch = next(iter(loader))

    expected = np.array(
        [
            [4, 5, 6, 5, 1, 2],
            [2, 0, 1, 2, 0, 1],
        ],
        dtype=np.int64,
    )
    np.testing.assert_array_equal(batch.exogenous["source_index"], expected)


def test_arch_named_bootstrap_loaders_are_available():
    history = {"source_index": np.arange(6)}

    loaders = [
        IIDBootstrap(
            initial_state=0,
            history=history,
            horizon=4,
            n_scenarios=2,
            batch_size=2,
            seed=1,
        ),
        CircularBlockBootstrap(
            initial_state=0,
            history=history,
            horizon=4,
            n_scenarios=2,
            batch_size=2,
            block_size=2,
            seed=1,
        ),
        MovingBlockBootstrap(
            initial_state=0,
            history=history,
            horizon=4,
            n_scenarios=2,
            batch_size=2,
            block_size=2,
            seed=1,
        ),
        StationaryBootstrap(
            initial_state=0,
            history=history,
            horizon=4,
            n_scenarios=2,
            batch_size=2,
            block_size=2,
            seed=1,
        ),
    ]

    for loader in loaders:
        batch = next(iter(loader))
        assert batch.exogenous["source_index"].shape == (2, 4)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"method": "unknown"}, "method must be"),
        ({"method": "circular_block"}, "block_size is required"),
        ({"method": "moving_block"}, "block_size is required"),
        (
            {"method": "circular_block", "block_size": 0},
            "block_size must be positive",
        ),
        (
            {"method": "moving_block", "block_size": 6},
            "less than or equal",
        ),
        (
            {"method": "circular_block", "block_size": 2.5},
            "block_size must be an integer",
        ),
        (
            {"method": "iid", "block_size": 2},
            "block_size is only supported",
        ),
        ({"method": "stationary_block"}, "average_block_size is required"),
        (
            {"method": "stationary_block", "average_block_size": 0.5},
            "average_block_size must be finite",
        ),
    ],
)
def test_bootstrap_scenario_loader_validates_method_parameters(kwargs, message):
    with pytest.raises(ValueError, match=message):
        BootstrapScenarioLoader(
            initial_state=0,
            history={"source_index": np.arange(5)},
            horizon=3,
            n_scenarios=2,
            batch_size=1,
            **kwargs,
        )
