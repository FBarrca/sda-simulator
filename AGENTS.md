# Agent Guide

This repository builds `sda`, a small Python library for Sequential Decision
Analytics simulation. Keep it simple, modular, and easy to explain.

The library should feel lightly inspired by PyTorch Lightning: clear lifecycle
hooks, reusable data modules, domain model objects, and one small trainer-like
evaluation entrypoint. Do not couple the project to Lightning or recreate a
large framework.

## North Star

The standard user flow is:

```python
policy = MyPolicy(...)
model = MyModel(policy)
data = MyDataModule(...)
result = evaluate(model, data)
```

Everything in the library should support this shape:

- `Policy` decides what action to take.
- `SDAModel` defines state, transition, cost, and domain diagnostics.
- `DataModule` creates scenario batches through `batches(stage)`.
- `evaluate` and `Simulator` run the data lifecycle, rollout, and metrics.

Prefer one obvious path. Do not add parallel concepts for the same job.

## Hard Boundaries

Keep model, data, policy, metrics, and orchestration separate.

| Piece | Owns | Must Not Own |
| --- | --- | --- |
| `Policy` | Decision logic in `act(state, t, history)` | Scenario generation, file loading, transition logic, metric storage |
| `SDAModel` | Initial state, transition, cost, `info(...)` diagnostics | Data loading, random future generation, experiment orchestration |
| `DataModule` | `prepare_data`, `setup(stage)`, `batches(stage)`, scenario construction | Policy decisions, transition/cost logic, metric storage |
| `Metric` | Reading records and logging observations | Model dynamics, data creation, policy choice |
| `Simulator` | Data lifecycle, rollout order, metric dispatch | Domain rules, source-specific data logic |

`SDAModel` holds a policy because the simulator needs one model object to run,
but the decision rule should remain swappable. Put policy-specific behavior in
`Policy` unless the model is only delegating or combining policies in a very
small, explicit way.

## DataModule Lifecycle

`DataModule` is the only scenario data abstraction.

- `prepare_data()` does one-time shared preparation.
- `setup(stage)` builds stage-specific state.
- `batches(stage)` yields `ScenarioBatch` objects.
- Stage-specific behavior belongs inside `setup(stage)` or `batches(stage)`.

Do not add alternate data-yielding base classes, stage-specific data hooks, or
factory helpers. Use one of the concrete data modules for simple sources:

- `ArrayDataModule(...)` for already-built futures.
- `GeneratorDataModule(...)` for lazy statistical, forecast, service, or
  domain-generated futures.
- `BootstrapDataModule(...)` for bootstrap scenarios from historical
  observations.

Use a custom `DataModule` only when scenario construction has configuration,
setup, fitted state, multiple stages, or source-specific batching logic.

## Simple Architecture Rules

- Do not overengineer.
- Keep implementations clean and minimal; avoid duplicate code when an
  existing method or small helper can express the behavior.
- Do not add registries, managers, plugin systems, dependency-injection
  containers, or broad abstraction layers unless the need is concrete in the
  current code.
- Before adding a module or base class, check whether `Policy`, `SDAModel`,
  `DataModule`, `Metric`, or a plain function is enough.
- Keep `sda.core` as the canonical contract and record layer.
- Keep `sda.data` about data modules, source adapters, and batching helpers.
- Keep `sda.simulation` about lifecycle and rollout orchestration.
- Keep `sda.metrics` about metric records, storage, and queries.
- Keep `sda.__init__` as the small public re-export surface.
- Keep examples as source-tree examples, not installed package API.

## Package Map

- `sda/core.py`: `ScenarioBatch`, `Policy`, `SDAModel`, `StepRecord`,
  `TrajectoryRecord`.
- `sda/data/module.py`: `DataModule`.
- `sda/data/array.py`: `ArrayDataModule`.
- `sda/data/generator.py`: `GeneratorDataModule`.
- `sda/data/bootstrap.py`: `BootstrapDataModule`.
- `sda/data/_state.py`: private initial-state slicing helper.
- `sda/simulation.py`: `evaluate`, `Simulator`, `SimulationResult`.
- `sda/metrics.py`: metrics, metric storage, and metric queries.
- `examples/`: runnable examples only.
- `docs/`: Sphinx documentation.
- `tests/`: pytest coverage.

Do not recreate legacy re-export layers or factory shortcuts; the package
should have one public data path.

Prefer public imports from `sda`:

```python
from sda import (
    ArrayDataModule,
    BootstrapDataModule,
    DataModule,
    GeneratorDataModule,
    Policy,
    SDAModel,
    evaluate,
)
```

## Data Invariants

- Exogenous arrays are batch-first and time-second:
  `[batch_size, horizon, ...]`.
- Full in-memory futures use `[n_scenarios, horizon, ...]`.
- `scenario_ids` align with the first dimension of every exogenous array.
- Step exogenous values preserve the batch dimension and drop only the time
  dimension.
- Costs are scalar or one-dimensional with one value per scenario.
- `initial_state` may be scalar, per-scenario, or a mapping of those.
- Seeded data modules should be deterministic across repeated `batches()`
  iterations.

## Data Source Guidance

Pick the simplest data module:

- `ArrayDataModule` for precomputed futures.
- `GeneratorDataModule` for lazy futures from a distribution, forecast model,
  external service, or domain simulator.
- `BootstrapDataModule` for empirical futures sampled from historical
  observations.
- Custom `DataModule` when the source has setup, stages, or domain-specific
  batching semantics.

Generator callables may request supported context names such as `rng`, `shape`,
`scenario_ids`, `horizon`, `batch_size`, `start`, `stop`, and `n_scenarios`.
Keep that signature-filtering behavior stable.

## Implementation Style

- Use standard library types plus NumPy by default.
- Add runtime dependencies only with a strong reason.
- Validate data at the boundary where users provide it.
- Raise `ValueError` for invalid values and `TypeError` for unsupported object
  shapes or call signatures.
- Write docstrings for public classes, methods, and functions.
- Keep changes focused; avoid broad refactors for narrow features.

## Tests And Docs

Run tests with:

```bash
uv run pytest
```

Direct equivalent:

```bash
python3 -m pytest
```

Add tests when changing public behavior, especially import surface, shape
validation, batch slicing, seeded generation, metric queries, and end-to-end
`evaluate` flows.

Update docs when public behavior changes:

- `docs/quickstart.rst` for first-use workflows.
- `docs/concepts.rst` for concepts.
- `docs/data.rst` for data behavior.
- `docs/api.rst` for API changes.
- `docs/architecture.rst` for package boundaries.
- `README.md` for the short project overview.

Build docs with:

```bash
uv run --group docs sphinx-build -b html docs docs/_build/html
```

Direct equivalent:

```bash
python3 -m sphinx -b html docs docs/_build/html
```

## Working Rules

- Respect existing user edits; the worktree may be dirty.
- Ask the user when unsure, or when the change appears to warrant a broader
  rework than the request explicitly covers.
- Use `rg` and `rg --files` for search.
- Prefer small patches.
- Keep generated artifacts, build outputs, and local environment files out of
  commits unless explicitly requested.
