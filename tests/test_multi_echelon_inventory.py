from types import SimpleNamespace
from xml.etree import ElementTree as ET

import numpy as np
import pytest

import examples.multi_echelon_inventory.main as main_module
from examples.multi_echelon_inventory.visualize import save_all_visualizations
from examples.multi_echelon_inventory import (
    MultiEchelonInventoryDataModule,
    REFERENCE_HORIZON,
    REFERENCE_INITIAL_GUESS,
    build_data,
    build_model,
    build_result,
    evaluate_reference_policy,
    policy_parameters_from_guess,
    reference_metric_names,
    summarize_reference_result,
)
from sda import evaluate


def test_reference_optimizer_guess_splits_into_rop_and_base_stock():
    parameters = policy_parameters_from_guess(REFERENCE_INITIAL_GUESS)

    np.testing.assert_allclose(
        parameters.reorder_point,
        [0, 1000, 250, 200, 150, 200],
    )
    np.testing.assert_allclose(
        parameters.base_stock,
        [10000, 3000, 600, 900, 300, 600],
    )
    np.testing.assert_allclose(
        parameters.initial_inventory,
        [9000, 2700, 540, 810, 270, 540],
    )


def test_example_supports_standard_data_model_evaluate_flow():
    data = build_data(n_scenarios=2, batch_size=1)
    model = build_model(initial_guess=REFERENCE_INITIAL_GUESS)

    result = evaluate(model, data)
    summary = summarize_reference_result(result)

    assert result["reference_average_on_hand"].count() == 2
    assert summary.objective == pytest.approx(2808.59555853753)


def test_build_result_matches_standard_evaluate_flow():
    result = build_result(replications=2, batch_size=1)
    summary = summarize_reference_result(result)

    assert result["total_cost"].count() == 2
    assert summary.average_on_hand == pytest.approx(2808.59555853753)
    assert "total_on_hand" not in result


def test_daily_diagnostics_are_opt_in():
    result = build_result(
        replications=1,
        batch_size=1,
        record_daily_metrics=True,
    )

    assert result["total_on_hand"].event_level().count() == REFERENCE_HORIZON
    assert result["demand_node_1"].event_level().count() == REFERENCE_HORIZON
    assert result["inventory_position_node_5"].event_level().count() == (
        REFERENCE_HORIZON
    )


def test_reference_metric_names_gate_daily_diagnostics():
    default_names = reference_metric_names()
    daily_names = reference_metric_names(include_daily_metrics=True)

    assert "reference_average_on_hand" in default_names
    assert "total_on_hand" not in default_names
    assert "demand_node_1" not in default_names
    assert "total_on_hand" in daily_names
    assert "demand_node_1" in daily_names


def test_cli_daily_metrics_flag_wires_through(monkeypatch, capsys):
    captured = {}

    def fake_build_evaluation(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            summary=SimpleNamespace(
                objective=1.0,
                average_on_hand=2.0,
                service_penalty=0.0,
                service_level=np.zeros(6),
            ),
            policy=SimpleNamespace(
                reorder_point=np.zeros(6),
                base_stock=np.ones(6),
            ),
        )

    monkeypatch.setattr(main_module, "build_evaluation", fake_build_evaluation)

    main_module.main(["--replications", "3", "--daily-metrics"])

    assert captured["replications"] == 3
    assert captured["record_daily_metrics"] is True
    assert "Objective: 1.000" in capsys.readouterr().out


def test_multi_echelon_visualizations_are_generated(tmp_path):
    paths = save_all_visualizations(tmp_path, replications=2, trace_seed=0)

    for path in paths.__dict__.values():
        assert path.exists()
        ET.parse(path)

    assert "Multi-echelon supply network" in paths.network.read_text()
    assert "Objective and service-level scorecard" in paths.objective.read_text()
    assert "Daily inventory dynamics" in paths.trace.read_text()


def test_lost_sales_reconstruction_matches_reference_two_replications():
    evaluation = evaluate_reference_policy(
        REFERENCE_INITIAL_GUESS,
        service_mode="lost_sales",
        replications=2,
        batch_size=1,
    )

    assert evaluation.summary.objective == pytest.approx(2808.59555853753)
    assert evaluation.summary.average_on_hand == pytest.approx(2808.59555853753)
    np.testing.assert_allclose(
        evaluation.summary.service_level,
        [0.0, 0.97719841, 0.99916860, 0.0, 1.0, 0.95806720],
        rtol=1e-7,
        atol=1e-7,
    )


def test_backorder_reconstruction_matches_reference_two_replications():
    evaluation = evaluate_reference_policy(
        REFERENCE_INITIAL_GUESS,
        service_mode="backorder",
        replications=2,
        batch_size=1,
    )

    assert evaluation.summary.objective == pytest.approx(2718.6934033582143)
    assert evaluation.summary.average_on_hand == pytest.approx(2718.6934033582143)
    np.testing.assert_allclose(
        evaluation.summary.service_level,
        [1.0, 0.97762852, 0.99991422, 1.0, 0.99870662, 0.95952097],
        rtol=1e-7,
        atol=1e-7,
    )


def test_lost_sales_reconstruction_matches_reference_seed_with_morning_delivery():
    data = MultiEchelonInventoryDataModule(
        n_scenarios=1,
        batch_size=1,
        scenario_seeds=[11],
    )

    evaluation = evaluate_reference_policy(
        REFERENCE_INITIAL_GUESS,
        service_mode="lost_sales",
        replications=1,
        batch_size=1,
        data=data,
    )

    assert evaluation.summary.objective == pytest.approx(2646.674550227423)
    np.testing.assert_allclose(
        evaluation.summary.service_level,
        [0.0, 0.95922582, 0.97962550, 0.0, 0.99923708, 0.97485655],
        rtol=1e-7,
        atol=1e-7,
    )
