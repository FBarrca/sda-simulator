import numpy as np
import pytest

from sda import (
    ArrayDataModule,
    BootstrapDataModule,
    GeneratorDataModule,
    ScenarioBatch,
    ScenarioSpec,
)


def batches(data):
    return list(data.batches())


def test_scenario_specs_validate_identity_and_time():
    scenario = ScenarioSpec(scenario_id=7, end_time=3.5, seed=11)
    batch = ScenarioBatch([scenario])

    assert batch.batch_size == 1
    assert batch.scenario_ids == [7]
    assert batch.end_time == pytest.approx(3.5)

    with pytest.raises(ValueError, match="scenario_id"):
        ScenarioSpec(scenario_id=True, end_time=1)
    with pytest.raises(ValueError, match="end_time"):
        ScenarioSpec(scenario_id=1, end_time=-1)
    with pytest.raises(ValueError, match="seed"):
        ScenarioSpec(scenario_id=1, end_time=1, seed=True)
    with pytest.raises(ValueError, match="at least one"):
        ScenarioBatch([])


def test_array_data_module_yields_per_scenario_specs():
    data = ArrayDataModule(
        {"demand": np.arange(12).reshape(3, 4)},
        initial_state={"inventory": np.array([10, 11, 12])},
        batch_size=2,
        scenario_ids=[100, 101, 102],
        seeds=[3, 4, 5],
    )

    result = batches(data)

    assert [batch.batch_size for batch in result] == [2, 1]
    assert result[0].scenario_ids == [100, 101]
    assert result[0].scenarios[0].end_time == pytest.approx(4)
    assert result[0].scenarios[0].initial_state == {"inventory": 10}
    assert result[0].scenarios[1].seed == 4
    np.testing.assert_array_equal(result[1].scenarios[0].data["demand"], [8, 9, 10, 11])


def test_array_data_module_broadcasts_scalar_initial_state():
    data = ArrayDataModule(
        {"demand": np.ones((3, 2))},
        initial_state=50,
        batch_size=2,
    )

    assert [
        [scenario.initial_state for scenario in batch.scenarios]
        for batch in batches(data)
    ] == [[50, 50], [50]]


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

    assert [batch.batch_size for batch in result] == [2, 2, 1]
    assert result[0].scenario_ids == [100, 101]
    assert [scenario.initial_state for scenario in result[0].scenarios] == [20, 21]
    np.testing.assert_array_equal(result[0].scenarios[0].data["demand"], expected_first[0])
    np.testing.assert_array_equal(result[1].scenarios[1].data["demand"], expected_second[1])
    np.testing.assert_array_equal(result[2].scenarios[0].data["demand"], expected_third[0])


def test_generator_data_module_accepts_complete_scenario_batches():
    def complete_batch(*, scenario_ids, horizon):
        return ScenarioBatch(
            [
                ScenarioSpec(
                    scenario_id=int(scenario_id),
                    end_time=float(horizon),
                    initial_state=7,
                    data={"signal": np.arange(horizon) + scenario_id},
                )
                for scenario_id in scenario_ids
            ]
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

    assert result[0].scenarios[0].initial_state == 7
    assert result[1].scenarios[0].scenario_id == 30
    np.testing.assert_array_equal(result[0].scenarios[1].data["signal"], [20, 21, 22, 23])


def test_generator_data_module_accepts_scenario_spec_iterators():
    def scenario_specs(*, scenario_ids, horizon):
        return (
            ScenarioSpec(
                scenario_id=int(scenario_id),
                end_time=float(horizon),
                data={"signal": np.asarray([scenario_id])},
            )
            for scenario_id in scenario_ids
        )

    result = batches(
        GeneratorDataModule(
            scenario_specs,
            horizon=2,
            n_scenarios=2,
            batch_size=2,
            scenario_ids=[5, 6],
        )
    )

    assert result[0].scenario_ids == [5, 6]
    np.testing.assert_array_equal(result[0].scenarios[1].data["signal"], [6])


def test_generator_data_module_validates_generator_contract():
    def invalid_generator(*, rng):
        return [rng]

    with pytest.raises(TypeError, match="mapping, ScenarioBatch"):
        batches(GeneratorDataModule(invalid_generator, horizon=2, n_scenarios=1))

    def unsupported_generator(required_name):
        return {"demand": required_name}

    with pytest.raises(TypeError, match="unsupported required parameter"):
        batches(GeneratorDataModule(unsupported_generator, horizon=2, n_scenarios=1))

    def wrong_batch_shape(*, horizon):
        return {"demand": np.ones((3, horizon))}

    with pytest.raises(ValueError, match="one entry per scenario"):
        batches(
            GeneratorDataModule(
                wrong_batch_shape,
                horizon=2,
                n_scenarios=2,
                batch_size=2,
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
    assert [scenario.initial_state for scenario in result[0].scenarios] == [10, 11]
    np.testing.assert_array_equal(result[0].scenarios[0].data["returns"], expected_first[0])
    np.testing.assert_array_equal(result[0].scenarios[1].data["volume"], expected_first[1] + 100)
    np.testing.assert_array_equal(result[1].scenarios[0].data["returns"], expected_second[0])


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

    assert [scenario.initial_state for scenario in batch.scenarios] == [0, 0]
    for scenario in batch.scenarios:
        sampled_path = scenario.data["source_index"]
        np.testing.assert_array_equal(np.diff(sampled_path) % 4, np.ones(4, dtype=int))
