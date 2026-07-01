import numpy as np
import pytest

from sda import (
    ArrayDataModule,
    BootstrapDataModule,
    GeneratorDataModule,
    ScenarioBatch,
)


def batches(data):
    return list(data.batches())


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


def test_array_data_module_batches_arrays():
    data = ArrayDataModule(
        {"demand": np.arange(12).reshape(3, 4)},
        initial_state=np.array([10, 11, 12]),
        batch_size=2,
    )

    result = batches(data)

    assert len(result) == 2
    assert result[0].scenario_ids == [0, 1]
    assert result[0].initial_state.tolist() == [10, 11]
    assert result[0].horizon == 4
    assert result[1].scenario_ids == [2]
    assert result[1].exogenous["demand"].shape == (1, 4)


def test_array_data_module_broadcasts_scalar_initial_state():
    data = ArrayDataModule(
        {"demand": np.ones((3, 2))},
        initial_state=50,
        batch_size=2,
    )

    assert [batch.initial_state.tolist() for batch in batches(data)] == [
        [50, 50],
        [50],
    ]


def test_array_data_module_defaults_to_one_batch_and_zero_initial_state():
    demand = np.arange(6).reshape(3, 2)

    result = batches(ArrayDataModule({"demand": demand}))

    assert len(result) == 1
    assert result[0].scenario_ids == [0, 1, 2]
    assert result[0].initial_state.tolist() == [0, 0, 0]
    np.testing.assert_array_equal(result[0].exogenous["demand"], demand)


def test_generator_data_module_batches_statistical_generator_deterministically():
    def poisson_demand(*, rng, scenario_ids, horizon):
        return {
            "demand": rng.poisson(
                lam=10,
                size=(len(scenario_ids), horizon),
            )
        }

    data = GeneratorDataModule(
        poisson_demand,
        horizon=3,
        n_scenarios=5,
        initial_state=np.array([20, 21, 22, 23, 24]),
        batch_size=2,
        seed=17,
        scenario_ids=[100, 101, 102, 103, 104],
    )

    result = batches(data)

    rng = np.random.default_rng(17)
    expected_first = rng.poisson(lam=10, size=(2, 3))
    expected_second = rng.poisson(lam=10, size=(2, 3))
    expected_third = rng.poisson(lam=10, size=(1, 3))

    assert isinstance(data, GeneratorDataModule)
    assert [batch.batch_size for batch in result] == [2, 2, 1]
    assert result[0].scenario_ids == [100, 101]
    assert result[1].scenario_ids == [102, 103]
    assert result[2].scenario_ids == [104]
    assert result[0].initial_state.tolist() == [20, 21]
    assert result[2].initial_state.tolist() == [24]
    np.testing.assert_array_equal(result[0].exogenous["demand"], expected_first)
    np.testing.assert_array_equal(result[1].exogenous["demand"], expected_second)
    np.testing.assert_array_equal(result[2].exogenous["demand"], expected_third)


def test_generator_data_module_accepts_complete_scenario_batches():
    def complete_batch(*, rng, scenario_ids, horizon):
        del rng
        ids = np.asarray(scenario_ids)
        return ScenarioBatch(
            initial_state=np.full(len(scenario_ids), 7),
            exogenous={"signal": ids[:, None] + np.arange(horizon)},
            scenario_ids=scenario_ids,
        )

    result = batches(
        GeneratorDataModule(
            complete_batch,
            horizon=4,
            n_scenarios=3,
            batch_size=2,
            scenario_ids=[10, 20, 30],
        )
    )

    assert result[0].initial_state.tolist() == [7, 7]
    assert result[1].initial_state.tolist() == [7]
    np.testing.assert_array_equal(
        result[0].exogenous["signal"],
        np.array([[10, 11, 12, 13], [20, 21, 22, 23]]),
    )
    np.testing.assert_array_equal(
        result[1].exogenous["signal"],
        np.array([[30, 31, 32, 33]]),
    )


def test_generator_data_module_passes_only_requested_context():
    def simple_generator(*, rng, shape):
        return {"demand": rng.integers(1, 4, size=shape)}

    result = batches(
        GeneratorDataModule(
            simple_generator,
            horizon=2,
            n_scenarios=3,
            batch_size=2,
            seed=9,
        )
    )

    rng = np.random.default_rng(9)
    expected_first = rng.integers(1, 4, size=(2, 2))
    expected_second = rng.integers(1, 4, size=(1, 2))

    np.testing.assert_array_equal(result[0].exogenous["demand"], expected_first)
    np.testing.assert_array_equal(result[1].exogenous["demand"], expected_second)


def test_generator_data_module_accepts_kwargs_context():
    def contextual_generator(**context):
        return {"demand": np.full(context["shape"], context["start"])}

    result = batches(
        GeneratorDataModule(
            contextual_generator,
            horizon=2,
            n_scenarios=3,
            batch_size=2,
        )
    )

    np.testing.assert_array_equal(result[0].exogenous["demand"], np.full((2, 2), 0))
    np.testing.assert_array_equal(result[1].exogenous["demand"], np.full((1, 2), 2))


def test_generator_data_module_validates_generator_return_type():
    def invalid_generator(*, rng, scenario_ids, horizon):
        return [rng, scenario_ids, horizon]

    with pytest.raises(TypeError, match="mapping or ScenarioBatch"):
        batches(
            GeneratorDataModule(
                invalid_generator,
                horizon=2,
                n_scenarios=1,
            )
        )


def test_generator_data_module_validates_required_context_names():
    def unsupported_generator(required_name):
        return {"demand": required_name}

    with pytest.raises(TypeError, match="unsupported required parameter"):
        batches(
            GeneratorDataModule(
                unsupported_generator,
                horizon=2,
                n_scenarios=1,
            )
        )


def test_bootstrap_data_module_samples_individual_observations():
    data = BootstrapDataModule(
        history={
            "returns": np.arange(6),
            "volume": np.arange(6) + 100,
        },
        horizon=4,
        n_scenarios=3,
        initial_state=np.array([10, 11, 12]),
        batch_size=2,
        seed=23,
    )

    result = batches(data)

    rng = np.random.default_rng(23)
    expected_first = rng.integers(0, 6, size=(2, 4), dtype=np.int64)
    expected_second = rng.integers(0, 6, size=(1, 4), dtype=np.int64)

    assert [batch.batch_size for batch in result] == [2, 1]
    assert result[0].initial_state.tolist() == [10, 11]
    assert result[1].initial_state.tolist() == [12]
    np.testing.assert_array_equal(result[0].exogenous["returns"], expected_first)
    np.testing.assert_array_equal(result[0].exogenous["volume"], expected_first + 100)
    np.testing.assert_array_equal(result[1].exogenous["returns"], expected_second)


def test_bootstrap_data_module_wraps_circular_fixed_blocks():
    data = BootstrapDataModule(
        {"source_index": np.arange(4)},
        horizon=5,
        n_scenarios=2,
        initial_state=0,
        batch_size=2,
        method="circular",
        block_size=5,
        seed=7,
    )

    batch = batches(data)[0]

    assert batch.initial_state.tolist() == [0, 0]
    assert batch.exogenous["source_index"].shape == (2, 5)
    for sampled_path in batch.exogenous["source_index"]:
        np.testing.assert_array_equal(
            np.diff(sampled_path) % 4,
            np.ones(4, dtype=int),
        )


def test_bootstrap_data_module_uses_moving_non_wrapping_blocks():
    data = BootstrapDataModule(
        {"source_index": np.arange(6)},
        horizon=7,
        n_scenarios=2,
        initial_state=0,
        batch_size=2,
        method="moving_block",
        block_size=3,
        seed=11,
    )

    batch = batches(data)[0]

    expected = np.array(
        [
            [0, 1, 2, 0, 1, 2, 3],
            [1, 2, 3, 2, 3, 4, 2],
        ],
        dtype=np.int64,
    )
    np.testing.assert_array_equal(batch.exogenous["source_index"], expected)
    assert np.all(batch.exogenous["source_index"] < 6)


def test_bootstrap_data_module_uses_stationary_random_restarts():
    data = BootstrapDataModule(
        {"source_index": np.arange(7)},
        horizon=6,
        n_scenarios=2,
        initial_state=0,
        batch_size=2,
        method="stationary_block",
        average_block_size=3,
        seed=5,
    )

    batch = batches(data)[0]

    expected = np.array(
        [
            [4, 5, 6, 5, 1, 2],
            [2, 0, 1, 2, 0, 1],
        ],
        dtype=np.int64,
    )
    np.testing.assert_array_equal(batch.exogenous["source_index"], expected)


def test_bootstrap_data_module_accepts_stationary_alias_and_default_batch_size():
    data = BootstrapDataModule(
        {"source_index": np.arange(7)},
        horizon=6,
        n_scenarios=2,
        method="stationary",
        block_size=3,
        seed=5,
    )

    result = batches(data)

    expected = np.array(
        [
            [4, 5, 6, 5, 1, 2],
            [2, 0, 1, 2, 0, 1],
        ],
        dtype=np.int64,
    )
    assert len(result) == 1
    np.testing.assert_array_equal(result[0].exogenous["source_index"], expected)


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
def test_bootstrap_data_module_validates_method_parameters(kwargs, message):
    with pytest.raises(ValueError, match=message):
        BootstrapDataModule(
            {"source_index": np.arange(5)},
            horizon=3,
            n_scenarios=2,
            initial_state=0,
            batch_size=1,
            **kwargs,
        )
