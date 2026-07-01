import sda
import sda.core as core
import sda.data as data
import sda.data.array as data_array
import sda.data.bootstrap as data_bootstrap
import sda.data.generator as data_generator
import sda.metrics as metrics
import sda.simulation as simulation
import sda.tracking as tracking


def test_core_contract_import_identities():
    assert sda.Policy is core.Policy
    assert sda.SDAModel is core.SDAModel
    assert sda.EventRecord is core.EventRecord
    assert sda.ScenarioSpec is core.ScenarioSpec is data.ScenarioSpec
    assert sda.ScenarioBatch is core.ScenarioBatch is data.ScenarioBatch
    assert sda.DataModule is data.DataModule
    assert sda.ArrayDataModule is data.ArrayDataModule is data_array.ArrayDataModule
    assert sda.GeneratorDataModule is data.GeneratorDataModule
    assert sda.GeneratorDataModule is data_generator.GeneratorDataModule
    assert sda.BootstrapDataModule is data.BootstrapDataModule
    assert sda.BootstrapDataModule is data_bootstrap.BootstrapDataModule
    assert sda.MetricStore is metrics.MetricStore
    assert sda.MetricSeries is metrics.MetricSeries
    assert sda.Recorder is metrics.Recorder
    assert sda.Simulator is simulation.Simulator
    assert sda.evaluate is simulation.evaluate
    assert sda.MLflowTracker is tracking.MLflowTracker


def test_public_surfaces_are_small_and_current():
    assert sorted(sda.__all__) == [
        "ArrayDataModule",
        "BootstrapDataModule",
        "DataModule",
        "EventLevel",
        "EventRecord",
        "GeneratorDataModule",
        "MLflowTracker",
        "MetricSeries",
        "MetricStore",
        "Policy",
        "Recorder",
        "SDAModel",
        "ScenarioBatch",
        "ScenarioSpec",
        "SimulationResult",
        "Simulator",
        "evaluate",
    ]
    assert sorted(data.__all__) == [
        "ArrayDataModule",
        "BootstrapDataModule",
        "BootstrapMethod",
        "DataModule",
        "GeneratorDataModule",
        "ScenarioBatch",
        "ScenarioSpec",
    ]
