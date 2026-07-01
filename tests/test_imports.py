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
    assert sda.StepRecord is core.StepRecord
    assert sda.TrajectoryRecord is core.TrajectoryRecord
    assert sda.ScenarioBatch is core.ScenarioBatch is data.ScenarioBatch
    assert sda.DataModule is data.DataModule
    assert sda.ArrayDataModule is data.ArrayDataModule is data_array.ArrayDataModule
    assert sda.GeneratorDataModule is data.GeneratorDataModule
    assert sda.GeneratorDataModule is data_generator.GeneratorDataModule
    assert sda.BootstrapDataModule is data.BootstrapDataModule
    assert sda.BootstrapDataModule is data_bootstrap.BootstrapDataModule
    assert sda.evaluate is simulation.evaluate
    assert sda.StepMetric is metrics.StepMetric
    assert sda.TrajectoryMetric is metrics.TrajectoryMetric
    assert sda.InfoMetric is metrics.InfoMetric
    assert sda.step_metric is metrics.step_metric
    assert sda.trajectory_metric is metrics.trajectory_metric
    assert sda.info_metric is metrics.info_metric
    assert sda.MLflowTracker is tracking.MLflowTracker


def test_public_surfaces_are_small_and_current():
    assert sorted(sda.__all__) == [
        "ArrayDataModule",
        "BootstrapDataModule",
        "DataModule",
        "GeneratorDataModule",
        "InfoMetric",
        "MLflowTracker",
        "Metric",
        "MetricRow",
        "MetricSeries",
        "MetricSet",
        "MetricStore",
        "Policy",
        "SDAModel",
        "ScenarioBatch",
        "SimulationResult",
        "Simulator",
        "StepCostMetric",
        "StepMetric",
        "StepRecord",
        "TotalCostMetric",
        "TrajectoryMetric",
        "TrajectoryRecord",
        "evaluate",
        "info_metric",
        "step_metric",
        "trajectory_metric",
    ]
    assert sorted(data.__all__) == [
        "ArrayDataModule",
        "BootstrapDataModule",
        "BootstrapMethod",
        "DataModule",
        "GeneratorDataModule",
        "ScenarioBatch",
    ]
