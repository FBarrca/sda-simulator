from __future__ import annotations

import argparse
from dataclasses import dataclass
from html import escape
from pathlib import Path

import numpy as np

from examples.multi_echelon_inventory.data import MultiEchelonInventoryDataModule
from examples.multi_echelon_inventory.domain import (
    DEFAULT_LEAD_TIME,
    DEFAULT_NETWORK,
    DEFAULT_SERVICE_TARGET,
    REFERENCE_REPLICATIONS,
)
from examples.multi_echelon_inventory.main import (
    build_evaluation,
    build_policy,
    build_result,
)
from examples.multi_echelon_inventory.models import ServiceMode

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class VisualizationPaths:
    """Paths created by the multi-echelon visualization command."""

    network: Path
    policy_parameters: Path
    objective: Path
    trace: Path


@dataclass(frozen=True)
class ObjectiveCase:
    """Objective summary for one plotted policy vector."""

    label: str
    mode: ServiceMode
    objective: float
    average_on_hand: float
    service_penalty: float
    service_level: np.ndarray
    is_published: bool
    ci95: tuple[float, float] = (0.0, 0.0)


def save_all_visualizations(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    replications: int = REFERENCE_REPLICATIONS,
    trace_seed: int = 0,
) -> VisualizationPaths:
    """Create all SVG visualizations for the multi-echelon example."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return VisualizationPaths(
        network=save_network_plot(output_path / "multi_echelon_network.svg"),
        policy_parameters=save_policy_parameters_plot(
            output_path / "multi_echelon_policy_parameters.svg",
        ),
        objective=save_objective_plot(
            output_path / "multi_echelon_objective.svg",
            replications=replications,
        ),
        trace=save_inventory_trace_plot(
            output_path / "multi_echelon_inventory_trace.svg",
            trace_seed=trace_seed,
        ),
    )


def save_network_plot(
    output: str | Path = DEFAULT_OUTPUT_DIR / "multi_echelon_network.svg",
    *,
    network=DEFAULT_NETWORK,
    service_target=DEFAULT_SERVICE_TARGET,
    lead_time=DEFAULT_LEAD_TIME,
) -> Path:
    """Draw the six-node reference supply-chain network with lead times."""
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    network_array = np.asarray(network, dtype=int)
    target = np.asarray(service_target, dtype=float)
    lead_time_array = np.asarray(lead_time, dtype=float)
    positions = {
        0: (92, 270),
        1: (250, 270),
        2: (420, 172),
        3: (420, 368),
        4: (590, 298),
        5: (590, 434),
    }
    labels = {
        0: ("0", "Source", "unconstrained"),
        1: ("1", "Regional stock", "95% service"),
        2: ("2", "Customer node", "95% service"),
        3: ("3", "Transship", "diagnostic"),
        4: ("4", "Customer node", "95% service"),
        5: ("5", "Customer node", "95% service"),
    }

    edge_parts: list[str] = []
    for upstream, downstream in zip(*np.nonzero(network_array), strict=True):
        x1, y1 = positions[int(upstream)]
        x2, y2 = positions[int(downstream)]
        edge_parts.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            'class="edge" marker-end="url(#arrow)" />'
        )
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2 - 12
        lead_days = lead_time_array[int(downstream)]
        label = f"{lead_days:.0f}d lead time"
        label_width = 6.4 * len(label) + 10
        edge_parts.append(
            f'<rect x="{mid_x - label_width / 2:.1f}" y="{mid_y - 12:.1f}" '
            f'width="{label_width:.1f}" height="16" rx="4" class="edge-label-bg" />'
            f'<text x="{mid_x:.1f}" y="{mid_y:.1f}" class="edge-label">{label}</text>'
        )

    node_parts: list[str] = []
    for node, (x, y) in positions.items():
        node_id, title, subtitle = labels[node]
        target_class = "target" if target[node] > 0 else "neutral"
        node_parts.append(
            f'<g class="node {target_class}">'
            f'<circle cx="{x}" cy="{y}" r="42" />'
            f'<text x="{x}" y="{y - 10}" class="node-id">{node_id}</text>'
            f'<text x="{x}" y="{y + 10}" class="node-title">{escape(title)}</text>'
            f'<text x="{x}" y="{y + 28}" class="node-subtitle">{escape(subtitle)}</text>'
            "</g>"
        )

    caption = _wrap_text(
        "One source feeds five stocking nodes over lanes with empirical "
        "lead times; nodes 1, 2, 4, and 5 carry a 95% fill-rate target.",
        x=44,
        first_y=92,
        line_height=17,
        max_chars=64,
        css_class="caption",
    )

    body = f"""
    <defs>
      <marker id="arrow" markerWidth="12" markerHeight="12" refX="10" refY="6"
              orient="auto" markerUnits="strokeWidth">
        <path d="M2,2 L10,6 L2,10 Z" fill="#475569" />
      </marker>
    </defs>
    <rect class="panel" x="24" y="24" width="642" height="478" rx="16" />
    <text x="44" y="66" class="title">Multi-echelon supply network</text>
    {caption}
    {''.join(edge_parts)}
    {''.join(node_parts)}
    <g class="legend">
      <circle cx="48" cy="486" r="8" class="target-dot" />
      <text x="64" y="491">95% service target</text>
      <circle cx="218" cy="486" r="8" class="neutral-dot" />
      <text x="234" y="491">source or diagnostic node</text>
    </g>
    """
    return _write_svg(output_path, 690, 526, body, _network_style())


def save_policy_parameters_plot(
    output: str | Path = DEFAULT_OUTPUT_DIR / "multi_echelon_policy_parameters.svg",
    *,
    service_mode: ServiceMode = "lost_sales",
) -> Path:
    """Compare the tuned reorder-point and base-stock hyperparameters per node."""
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    initial_policy = build_policy(service_mode=service_mode, use_published_solution=False)
    published_policy = build_policy(service_mode=service_mode, use_published_solution=True)

    node_labels = {1: "Node 1", 2: "Node 2", 3: "Node 3", 4: "Node 4", 5: "Node 5"}
    nodes = list(node_labels)
    max_stock = float(
        max(
            initial_policy.base_stock[nodes].max(),
            published_policy.base_stock[nodes].max(),
        )
    )

    chart_left = 208
    chart_width = 508
    row_height = 24
    pair_gap = 10
    group_gap = 22
    top = 152

    rows: list[str] = []
    y = top
    for node in nodes:
        group_top = y
        for label, policy, css in (
            ("Initial guess", initial_policy, "initial"),
            ("Published", published_policy, "published"),
        ):
            rop = float(policy.reorder_point[node])
            stock = float(policy.base_stock[node])
            rop_width = chart_width * rop / max_stock
            stock_width = chart_width * stock / max_stock
            rows.append(
                f'<rect x="{chart_left}" y="{y}" width="{chart_width}" '
                f'height="{row_height}" class="bar-track" />'
                f'<rect x="{chart_left}" y="{y}" width="{rop_width:.1f}" '
                f'height="{row_height}" class="bar-rop {css}" />'
                f'<rect x="{chart_left + rop_width:.1f}" y="{y}" '
                f'width="{max(stock_width - rop_width, 0):.1f}" height="{row_height}" '
                f'class="bar-excess {css}" />'
                f'<text x="{chart_left + 8}" y="{y + row_height / 2 + 4:.1f}" '
                f'class="bar-caption">{label}</text>'
                f'<text x="{chart_left + chart_width + 12}" '
                f'y="{y + row_height / 2 + 4:.1f}" class="bar-value">'
                f"ROP {rop:.0f} &#8594; stock {stock:.0f}</text>"
            )
            y += row_height + pair_gap
        rows.append(
            f'<text x="{chart_left - 16}" y="{(group_top + y - pair_gap) / 2 + 4:.1f}" '
            f'class="node-label" text-anchor="end">{escape(node_labels[node])}</text>'
        )
        y += group_gap

    ticks: list[str] = []
    axis_bottom = y - group_gap + 6
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = chart_left + chart_width * fraction
        ticks.append(
            f'<line x1="{x:.1f}" y1="{top - 10}" x2="{x:.1f}" y2="{axis_bottom:.1f}" '
            'class="grid-v" />'
        )
        ticks.append(
            f'<text x="{x:.1f}" y="{top - 18}" class="tick" '
            f'text-anchor="middle">{max_stock * fraction:.0f}</text>'
        )

    legend_row_gap = 24
    legend_y = y + 12
    height = int(legend_y + legend_row_gap + 40)
    width = 1000
    caption = _wrap_text(
        f"Each pair compares the optimizer's initial guess to the published "
        f"{service_mode.replace('_', '-')} solution. Bar length is the base "
        "stock (order-up-to) level; the dark segment is the reorder point.",
        x=48,
        first_y=87,
        line_height=16,
        max_chars=92,
        css_class="caption",
    )
    body = f"""
    <rect class="panel" x="24" y="24" width="{width - 48}" height="{height - 48}" rx="16" />
    <text x="48" y="60" class="title">Reorder-point and base-stock hyperparameters</text>
    {caption}
    {''.join(ticks)}
    {''.join(rows)}
    <g class="legend" transform="translate({chart_left},{legend_y:.1f})">
      <rect width="14" height="14" class="bar-rop initial" />
      <text x="20" y="12">Initial reorder point</text>
      <rect x="240" width="14" height="14" class="bar-excess initial" />
      <text x="260" y="12">Initial order-up-to buffer</text>
      <rect x="0" y="{legend_row_gap}" width="14" height="14" class="bar-rop published" />
      <text x="20" y="{legend_row_gap + 12}">Published reorder point</text>
      <rect x="240" y="{legend_row_gap}" width="14" height="14" class="bar-excess published" />
      <text x="260" y="{legend_row_gap + 12}">Published order-up-to buffer</text>
    </g>
    """
    return _write_svg(output_path, width, height, body, _policy_style())


def save_objective_plot(
    output: str | Path = DEFAULT_OUTPUT_DIR / "multi_echelon_objective.svg",
    *,
    replications: int = REFERENCE_REPLICATIONS,
) -> Path:
    """Plot initial versus published objective values and service levels."""
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cases = _objective_cases(replications=replications)
    # Scale to the largest CI upper bound so the whiskers always fit inside the axis.
    axis_max = max(max(case.objective, case.ci95[1]) for case in cases)
    chart_left = 72
    chart_top = 112
    chart_width = 496
    chart_height = 238
    bar_width = 58
    gap = 42
    colors = {
        ("lost_sales", False): "#64748b",
        ("lost_sales", True): "#2563eb",
        ("backorder", False): "#94a3b8",
        ("backorder", True): "#16a34a",
    }

    def y_of(value: float) -> float:
        return chart_top + chart_height - chart_height * value / axis_max

    bars: list[str] = []
    for index, case in enumerate(cases):
        x = chart_left + index * (bar_width + gap)
        center = x + bar_width / 2
        y = y_of(case.objective)
        color = colors[(case.mode, case.is_published)]
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width}" '
            f'height="{chart_top + chart_height - y:.1f}" rx="5" fill="{color}" />'
            f'<text x="{center:.1f}" y="{y - 22:.1f}" '
            f'class="value">{case.objective:.0f}</text>'
            f'<text x="{center:.1f}" y="{chart_top + chart_height + 24}" '
            f'class="axis-label">{escape(case.label)}</text>'
        )
        # 95% confidence whisker over the bootstrap replications.
        low, high = case.ci95
        if high > low:
            y_hi, y_lo = y_of(high), y_of(low)
            cap = 9
            bars.append(
                f'<g class="whisker" stroke="#0f172a" stroke-width="1.8" fill="none">'
                f'<line x1="{center:.1f}" y1="{y_hi:.1f}" x2="{center:.1f}" y2="{y_lo:.1f}" />'
                f'<line x1="{center - cap:.1f}" y1="{y_hi:.1f}" x2="{center + cap:.1f}" y2="{y_hi:.1f}" />'
                f'<line x1="{center - cap:.1f}" y1="{y_lo:.1f}" x2="{center + cap:.1f}" y2="{y_lo:.1f}" />'
                f"</g>"
            )

    initial_lost, initial_back, published_lost, published_back = cases
    lost_improvement = _improvement(initial_lost.objective, published_lost.objective)
    backorder_improvement = _improvement(
        initial_back.objective,
        published_back.objective,
    )
    service_rows = _service_level_rows(published_lost, published_back)
    body = f"""
    <rect class="panel" x="24" y="24" width="920" height="562" rx="16" />
    <text x="48" y="64" class="title">Objective and service-level scorecard</text>
    <text x="48" y="91" class="caption">
      Objective = average on-hand inventory + 1e6 x service shortfall. Bars are means over 20 replications; whiskers are 95% CIs. Published vectors reduce inventory while meeting targets.
    </text>
    <g class="axis">
      <line x1="{chart_left}" y1="{chart_top + chart_height}" x2="{chart_left + chart_width}" y2="{chart_top + chart_height}" />
      <line x1="{chart_left}" y1="{chart_top}" x2="{chart_left}" y2="{chart_top + chart_height}" />
      <text x="{chart_left - 12}" y="{chart_top + 10}" class="tick">{axis_max:.0f}</text>
      <text x="{chart_left - 12}" y="{chart_top + chart_height}" class="tick">0</text>
    </g>
    {''.join(bars)}
    <g class="callout" transform="translate(620,122)">
      <text x="0" y="0" class="callout-title">Value from optimization</text>
      <text x="0" y="34">Lost sales: {lost_improvement:.1f}% less average inventory</text>
      <text x="0" y="62">Backorder: {backorder_improvement:.1f}% less average inventory</text>
      <text x="0" y="90">Service penalty: 0 in both published runs</text>
      <text x="0" y="118">20 seeded replications, horizon 360</text>
    </g>
    <g transform="translate(72,422)">
      <text x="0" y="0" class="section-title">
        Published solution service levels (black tick = 95% target)
      </text>
      {service_rows}
    </g>
    """
    return _write_svg(output_path, 968, 610, body, _scorecard_style())


def save_inventory_trace_plot(
    output: str | Path = DEFAULT_OUTPUT_DIR / "multi_echelon_inventory_trace.svg",
    *,
    service_mode: ServiceMode = "lost_sales",
    trace_seed: int = 0,
    focus_node: int | None = None,
    nodes: tuple[int, ...] | None = None,
    window_days: int = 120,
) -> Path:
    """Plot daily policy diagnostics for multiple stocking nodes.

    Each node gets its own row so the PFA is visible even when upstream and
    downstream inventory levels live on very different scales.
    """
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plotted_nodes = (
        tuple(range(1, DEFAULT_NETWORK.shape[0]))
        if nodes is None and focus_node is None
        else ((focus_node,) if nodes is None else tuple(nodes))
    )
    if any(node is None for node in plotted_nodes):
        raise ValueError("nodes must contain concrete node indexes")
    if not plotted_nodes:
        raise ValueError("at least one node must be plotted")

    data = MultiEchelonInventoryDataModule(
        n_scenarios=1,
        batch_size=1,
        scenario_seeds=[trace_seed],
    )
    policy = build_policy(service_mode=service_mode, use_published_solution=True)
    result = build_result(
        data=data,
        service_mode=service_mode,
        use_published_solution=True,
        record_daily_metrics=True,
    )
    _, times, _ = result[
        f"on_hand_inventory_node_{plotted_nodes[0]}"
    ].to_trajectory_matrix()

    full_horizon = int(np.nanmax(times))
    window = times <= window_days
    windowed_times = times[window]

    traces: list[dict[str, object]] = []
    for node in plotted_nodes:
        on_hand_values = result[f"on_hand_inventory_node_{node}"].to_trajectory_matrix()[
            2
        ][0]
        position_values = result[
            f"inventory_position_node_{node}"
        ].to_trajectory_matrix()[2][0]
        reorder_point = float(policy.reorder_point[node])
        traces.append(
            {
                "node": int(node),
                "on_hand": on_hand_values[window],
                "position": position_values[window],
                "trigger": policy.reorder_tolerance * reorder_point,
                "reorder_point": reorder_point,
                "base_stock": float(policy.base_stock[node]),
            }
        )

    chart = _multi_node_policy_trace_chart(
        windowed_times,
        traces,
        left=112,
        top=166,
        width=706,
        row_height=72,
        row_gap=18,
    )
    topology = _trace_topology_inset(
        DEFAULT_NETWORK,
        plotted_nodes,
        x=900,
        y=166,
        width=214,
        height=432,
    )
    caption = _wrap_text(
        f"Published {service_mode.replace('_', '-')} policy, seed {trace_seed}. First "
        f"{window_days} of {full_horizon} simulated days shown for legibility. "
        "Each row has its own y-scale so upstream and downstream stocking nodes "
        "can be compared; the topology inset maps row numbers to echelon position.",
        x=48,
        first_y=90,
        line_height=16,
        max_chars=98,
        css_class="caption",
    )
    height = 246 + len(traces) * 90
    body = f"""
    <rect class="panel" x="24" y="24" width="1132" height="{height - 48}" rx="16" />
    <text x="48" y="64" class="title">Daily inventory dynamics</text>
    {caption}
    {chart}
    {topology}
    """
    return _write_svg(output_path, 1180, height, body, _trace_style())


def _objective_cases(*, replications: int) -> list[ObjectiveCase]:
    specs = [
        ("Lost initial", "lost_sales", False),
        ("Backorder initial", "backorder", False),
        ("Lost published", "lost_sales", True),
        ("Backorder published", "backorder", True),
    ]
    cases: list[ObjectiveCase] = []
    for label, mode, use_published in specs:
        evaluation = build_evaluation(
            service_mode=mode,
            use_published_solution=use_published,
            replications=replications,
            batch_size=1,
        )
        cases.append(
            ObjectiveCase(
                label=label,
                mode=mode,
                objective=evaluation.summary.objective,
                average_on_hand=evaluation.summary.average_on_hand,
                service_penalty=evaluation.summary.service_penalty,
                service_level=evaluation.summary.service_level,
                is_published=use_published,
                ci95=evaluation.summary.average_on_hand_ci95,
            )
        )
    return cases


def _service_level_rows(lost: ObjectiveCase, backorder: ObjectiveCase) -> str:
    target = np.asarray(DEFAULT_SERVICE_TARGET, dtype=float)
    rows: list[str] = []
    row_y = 28
    for node in range(len(target)):
        if target[node] <= 0:
            continue
        lost_width = 180 * min(lost.service_level[node], 1.0)
        back_width = 180 * min(backorder.service_level[node], 1.0)
        target_x = 84 + 180 * target[node]
        rows.append(
            f'<text x="0" y="{row_y + 11}" class="node-label">Node {node}</text>'
            f'<rect x="84" y="{row_y}" width="180" height="10" class="bar-bg" />'
            f'<rect x="84" y="{row_y}" width="{lost_width:.1f}" height="10" class="lost-bar" />'
            f'<line x1="{target_x:.1f}" y1="{row_y - 4}" x2="{target_x:.1f}" y2="{row_y + 14}" class="target-line" />'
            f'<text x="276" y="{row_y + 11}" class="metric-label">lost {lost.service_level[node]:.1%}</text>'
            f'<rect x="400" y="{row_y}" width="180" height="10" class="bar-bg" />'
            f'<rect x="400" y="{row_y}" width="{back_width:.1f}" height="10" class="back-bar" />'
            f'<line x1="{400 + 180 * target[node]:.1f}" y1="{row_y - 4}" x2="{400 + 180 * target[node]:.1f}" y2="{row_y + 14}" class="target-line" />'
            f'<text x="592" y="{row_y + 11}" class="metric-label">backorder {backorder.service_level[node]:.1%}</text>'
        )
        row_y += 28
    return "".join(rows)


def _multi_node_policy_trace_chart(
    times: np.ndarray,
    traces: list[dict[str, object]],
    *,
    left: float,
    top: float,
    width: float,
    row_height: float,
    row_gap: float,
) -> str:
    x_min = float(np.nanmin(times))
    x_max = float(np.nanmax(times))

    def sx(value: float) -> float:
        return left + (float(value) - x_min) / (x_max - x_min) * width

    rows: list[str] = []
    for index, trace in enumerate(traces):
        row_top = top + index * (row_height + row_gap)
        row_bottom = row_top + row_height
        on_hand = np.asarray(trace["on_hand"], dtype=float)
        position = np.asarray(trace["position"], dtype=float)
        trigger = float(trace["trigger"])
        reorder_point = float(trace["reorder_point"])
        base_stock = float(trace["base_stock"])
        y_max = float(
            max(
                np.nanmax(on_hand),
                np.nanmax(position),
                base_stock * 1.08,
                trigger * 1.08,
            )
        )

        def sy(value: float) -> float:
            return row_bottom - float(value) / y_max * row_height

        grid = [
            f'<line x1="{left}" y1="{row_top + row_height * fraction:.1f}" '
            f'x2="{left + width}" y2="{row_top + row_height * fraction:.1f}" '
            'class="grid" />'
            for fraction in (0.0, 0.5, 1.0)
        ]
        lines = [
            f'<polyline class="position-line" points="{_points(times, position, sx, sy)}" />',
            f'<polyline class="onhand-line" points="{_points(times, on_hand, sx, sy)}" />',
        ]
        trigger_y = sy(trigger)
        target_y = sy(base_stock)
        references = [
            f'<line x1="{left}" y1="{trigger_y:.1f}" x2="{left + width}" '
            f'y2="{trigger_y:.1f}" class="reference-line trigger" />',
            f'<line x1="{left}" y1="{target_y:.1f}" x2="{left + width}" '
            f'y2="{target_y:.1f}" class="reference-line stock" />',
        ]

        order_markers: list[str] = []
        jumps = np.diff(position, prepend=position[0])
        jump_threshold = max(1.0, base_stock * 0.01)
        for day, value, jump in zip(times, position, jumps, strict=True):
            if jump <= jump_threshold:
                continue
            x = sx(day)
            y = sy(value)
            order_markers.append(
                f'<line x1="{x:.1f}" y1="{row_top}" x2="{x:.1f}" '
                f'y2="{row_bottom}" class="order-line" />'
            )
            order_markers.append(
                _order_marker(x, max(row_top + 2.0, y - 9.0), size=8.0)
            )

        node = int(trace["node"])
        rows.append(
            "".join(
                [
                    f'<text x="{left - 20}" y="{row_top + 18:.1f}" '
                    f'class="node-row-label" text-anchor="end">Node {node}</text>',
                    f'<text x="{left - 20}" y="{row_top + 38:.1f}" '
                    f'class="node-row-detail" text-anchor="end">R {reorder_point:.0f} / B {base_stock:.0f}</text>',
                    f'<text x="{left - 12}" y="{row_top + 6:.1f}" '
                    f'class="tick">{y_max:.0f}</text>',
                    *grid,
                    f'<line x1="{left}" y1="{row_bottom}" x2="{left + width}" y2="{row_bottom}" class="axis" />',
                    f'<line x1="{left}" y1="{row_top}" x2="{left}" y2="{row_bottom}" class="axis" />',
                    *references,
                    *order_markers,
                    *lines,
                    f'<text x="{left + width + 12}" y="{trigger_y + 4:.1f}" class="reference-label">trigger</text>',
                    f'<text x="{left + width + 12}" y="{target_y + 4:.1f}" class="reference-label">target</text>',
                ]
            )
        )

    final_bottom = top + (len(traces) - 1) * (row_height + row_gap) + row_height
    legend_y = final_bottom + 44
    legend = [
        f'<line x1="{left}" y1="{legend_y}" x2="{left + 24}" y2="{legend_y}" class="position-swatch" />'
        f'<text x="{left + 32}" y="{legend_y + 5}" class="legend">Inventory position</text>',
        f'<line x1="{left + 190}" y1="{legend_y}" x2="{left + 214}" y2="{legend_y}" class="onhand-swatch" />'
        f'<text x="{left + 222}" y="{legend_y + 5}" class="legend">On hand</text>',
        f'<polygon points="{left + 334},{legend_y - 7} {left + 344},{legend_y - 7} {left + 339},{legend_y + 3}" class="order-marker" />'
        f'<text x="{left + 354}" y="{legend_y + 5}" class="legend">Reorder fired</text>',
        f'<line x1="{left + 500}" y1="{legend_y}" x2="{left + 524}" y2="{legend_y}" class="reference-line stock" />'
        f'<text x="{left + 532}" y="{legend_y + 5}" class="legend">Target</text>',
        f'<line x1="{left + 610}" y1="{legend_y}" x2="{left + 634}" y2="{legend_y}" class="reference-line trigger" />'
        f'<text x="{left + 642}" y="{legend_y + 5}" class="legend">Trigger</text>',
    ]

    return "".join(
        [
            *rows,
            f'<text x="{left}" y="{final_bottom + 22}" class="tick start">day {int(x_min)}</text>',
            f'<text x="{left + width}" y="{final_bottom + 22}" class="tick end">day {int(x_max)}</text>',
            *legend,
        ]
    )


def _trace_topology_inset(
    network: np.ndarray,
    plotted_nodes: tuple[int, ...],
    *,
    x: float,
    y: float,
    width: float,
    height: float,
) -> str:
    network_array = np.asarray(network, dtype=int)
    highlighted = set(plotted_nodes)
    positions = {
        0: (x + 34, y + 92),
        1: (x + 94, y + 92),
        2: (x + 164, y + 48),
        3: (x + 164, y + 136),
        4: (x + 94, y + 238),
        5: (x + 164, y + 304),
    }
    labels = {
        0: "source",
        1: "regional",
        2: "customer",
        3: "transship",
        4: "customer",
        5: "customer",
    }

    edges: list[str] = [
        '<defs><marker id="trace-topology-arrow" markerWidth="8" markerHeight="8" '
        'refX="7" refY="4" orient="auto" markerUnits="strokeWidth">'
        '<path d="M1,1 L7,4 L1,7 Z" class="topology-arrow" /></marker></defs>'
    ]
    for upstream, downstream in zip(*np.nonzero(network_array), strict=True):
        if int(upstream) not in positions or int(downstream) not in positions:
            continue
        x1, y1 = positions[int(upstream)]
        x2, y2 = positions[int(downstream)]
        edges.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            'class="topology-edge" marker-end="url(#trace-topology-arrow)" />'
        )

    nodes: list[str] = []
    for node, (node_x, node_y) in positions.items():
        if node == 0:
            css_class = "source"
        elif node in highlighted:
            css_class = "plotted"
        else:
            css_class = "muted"
        nodes.append(
            f'<g class="topology-node {css_class}">'
            f'<circle cx="{node_x:.1f}" cy="{node_y:.1f}" r="16" />'
            f'<text x="{node_x:.1f}" y="{node_y + 4:.1f}" class="topology-node-id">{node}</text>'
            f'<text x="{node_x:.1f}" y="{node_y + 32:.1f}" class="topology-node-label">'
            f"{escape(labels[node])}</text>"
            "</g>"
        )

    return "".join(
        [
            '<g class="topology-inset">',
            f'<line x1="{x - 24}" y1="{y - 8}" x2="{x - 24}" '
            f'y2="{y + height - 22}" class="topology-divider" />',
            f'<text x="{x}" y="{y + 2}" class="topology-title">Network topology</text>',
            f'<text x="{x}" y="{y + 22}" class="topology-caption">'
            "node rows map to this flow</text>",
            "".join(edges),
            "".join(nodes),
            "</g>",
        ]
    )


def _points(xs, ys, sx, sy) -> str:
    return " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(xs, ys, strict=True))


def _order_marker(x: float, y: float, *, size: float = 10.0) -> str:
    half = size / 2.0
    return (
        f'<polygon points="{x - half:.1f},{y:.1f} {x + half:.1f},{y:.1f} '
        f'{x:.1f},{y + size:.1f}" class="order-marker" />'
    )


def _wrap_text(
    text: str,
    *,
    x: float,
    first_y: float,
    line_height: float,
    max_chars: int,
    css_class: str,
) -> str:
    """Break ``text`` into left-aligned ``<text>`` lines under ``max_chars``.

    SVG does not wrap text automatically, so long captions must be split into
    explicit lines to avoid overlapping other chart elements.
    """
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > max_chars and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return "".join(
        f'<text x="{x}" y="{first_y + index * line_height:.1f}" class="{css_class}">'
        f"{escape(line)}</text>"
        for index, line in enumerate(lines)
    )


def _improvement(before: float, after: float) -> float:
    return 100.0 * (before - after) / before


def _write_svg(output_path: Path, width: int, height: int, body: str, style: str) -> Path:
    output_path.write_text(
        "\n".join(
            [
                '<?xml version="1.0" encoding="UTF-8"?>',
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
                "<style>",
                style,
                "</style>",
                body,
                "</svg>",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return output_path


def _base_style() -> str:
    return """
    text { font-family: Arial, Helvetica, sans-serif; fill: #0f172a; }
    .panel { fill: #ffffff; stroke: #d8dee9; stroke-width: 1.2; }
    .title { font-size: 24px; font-weight: 700; }
    .caption { font-size: 13px; fill: #475569; }
    .section-title { font-size: 16px; font-weight: 700; }
    .tick { font-size: 11px; fill: #64748b; text-anchor: end; }
    .axis line, .axis { stroke: #94a3b8; stroke-width: 1.1; }
    .grid { stroke: #e2e8f0; stroke-width: 1; }
    """


def _network_style() -> str:
    return (
        _base_style()
        + """
    .edge { stroke: #475569; stroke-width: 3.2; opacity: 0.78; }
    .edge-label-bg { fill: #ffffff; opacity: 0.9; }
    .edge-label { font-size: 11px; font-weight: 700; fill: #334155; text-anchor: middle; }
    .node circle { stroke-width: 2; }
    .node.target circle { fill: #e0f2fe; stroke: #2563eb; }
    .node.neutral circle { fill: #f8fafc; stroke: #64748b; }
    .node-id { font-size: 18px; font-weight: 700; text-anchor: middle; }
    .node-title { font-size: 11px; font-weight: 700; text-anchor: middle; }
    .node-subtitle { font-size: 10px; fill: #475569; text-anchor: middle; }
    .legend text { font-size: 12px; fill: #475569; }
    .target-dot { fill: #e0f2fe; stroke: #2563eb; }
    .neutral-dot { fill: #f8fafc; stroke: #64748b; }
    """
    )


def _scorecard_style() -> str:
    return (
        _base_style()
        + """
    .value { font-size: 13px; font-weight: 700; text-anchor: middle; }
    .axis-label { font-size: 11px; fill: #475569; text-anchor: middle; }
    .callout-title { font-size: 17px; font-weight: 700; }
    .callout text { font-size: 14px; fill: #334155; }
    .bar-bg { fill: #e2e8f0; rx: 4; }
    .lost-bar { fill: #2563eb; rx: 4; }
    .back-bar { fill: #16a34a; rx: 4; }
    .target-line { stroke: #0f172a; stroke-width: 2; }
    .node-label { font-size: 12px; font-weight: 700; }
    .metric-label { font-size: 12px; fill: #475569; }
    """
    )


def _trace_style() -> str:
    return (
        _base_style()
        + """
    .position-line { fill: none; stroke: #15803d; stroke-width: 3.2; }
    .onhand-line { fill: none; stroke: #2563eb; stroke-width: 2.2; opacity: 0.92; }
    .position-swatch { stroke: #15803d; stroke-width: 3.2; }
    .onhand-swatch { stroke: #2563eb; stroke-width: 2.2; }
    .legend { font-size: 12px; fill: #475569; }
    .start { text-anchor: start; }
    .end { text-anchor: end; }
    .node-row-label { font-size: 13px; font-weight: 700; }
    .node-row-detail { font-size: 11px; fill: #64748b; }
    .reference-line { stroke-width: 1.6; stroke-dasharray: 6 5; }
    .reference-line.trigger { stroke: #dc2626; }
    .reference-line.stock { stroke: #0f172a; }
    .reference-label { font-size: 11px; font-weight: 700; fill: #334155; }
    .order-line { stroke: #94a3b8; stroke-width: 1; opacity: 0.26; }
    .order-marker { fill: #0f172a; opacity: 0.86; }
    .topology-divider { stroke: #d8dee9; stroke-width: 1.2; }
    .topology-title { font-size: 15px; font-weight: 700; }
    .topology-caption { font-size: 11px; fill: #64748b; }
    .topology-edge { stroke: #64748b; stroke-width: 2.1; opacity: 0.76; }
    .topology-arrow { fill: #64748b; }
    .topology-node circle { stroke-width: 1.8; }
    .topology-node.source circle { fill: #f8fafc; stroke: #64748b; }
    .topology-node.plotted circle { fill: #e0f2fe; stroke: #2563eb; }
    .topology-node.muted circle { fill: #f1f5f9; stroke: #94a3b8; }
    .topology-node-id { font-size: 13px; font-weight: 700; text-anchor: middle; }
    .topology-node-label { font-size: 9px; fill: #475569; text-anchor: middle; }
    """
    )


def _policy_style() -> str:
    return (
        _base_style()
        + """
    .bar-track { fill: #f1f5f9; }
    .bar-rop.initial { fill: #94a3b8; }
    .bar-excess.initial { fill: #cbd5e1; }
    .bar-rop.published { fill: #1d4ed8; }
    .bar-excess.published { fill: #93c5fd; }
    .bar-caption { font-size: 11px; font-weight: 700; fill: #1e293b; }
    .bar-value { font-size: 12px; fill: #334155; }
    .node-label { font-size: 13px; font-weight: 700; }
    .grid-v { stroke: #e2e8f0; stroke-width: 1; }
    .legend rect { rx: 3; }
    .legend text { font-size: 12px; fill: #475569; }
    """
    )


def main() -> None:
    """Create the multi-echelon example SVG visualizations."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="directory where SVG files are written",
    )
    parser.add_argument(
        "--replications",
        type=int,
        default=REFERENCE_REPLICATIONS,
        help="replications used for the objective scorecard",
    )
    parser.add_argument(
        "--trace-seed",
        type=int,
        default=0,
        help="scenario seed used for the daily inventory trace",
    )
    args = parser.parse_args()

    paths = save_all_visualizations(
        args.output_dir,
        replications=args.replications,
        trace_seed=args.trace_seed,
    )
    for path in paths.__dict__.values():
        print(f"Saved {path.resolve()}")


if __name__ == "__main__":
    main()
