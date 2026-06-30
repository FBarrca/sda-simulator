Data Loaders
============

Data loaders provide the exogenous futures that a simulator rolls through a
model. In ``sda`` the loader interface is called ``ScenarioLoader``. A loader
yields ``ScenarioBatch`` objects, and each batch contains:

``initial_state``
   The starting state for every scenario in the batch.

``exogenous``
   A mapping from input name to batch-first arrays shaped
   ``[batch_size, horizon, ...]``.

``scenario_ids``
   Stable identifiers for the scenarios in the batch.

The simulator does not care how a loader creates those futures. They may come
from precomputed arrays, a parametric distribution, historical resampling, a
forecasting model, or a domain-specific generator.

Array Loader
------------

Use ``ArrayScenarioLoader`` when futures are already available in memory. Each
exogenous array must be shaped ``[n_scenarios, horizon, ...]``.

.. code-block:: python

   import numpy as np
   from sda import ArrayScenarioLoader

   scenarios = ArrayScenarioLoader(
       initial_state=np.zeros(3),
       exogenous={
           "demand": np.array(
               [
                   [12, 10, 14, 9],
                   [8, 13, 11, 15],
                   [16, 12, 10, 13],
               ],
               dtype=float,
           )
       },
       batch_size=2,
   )

This loader slices the stored arrays into ``ScenarioBatch`` objects of at most
``batch_size`` scenarios. It does not resample or transform the data.

Bootstrap Loaders
-----------------

Bootstrap loaders generate new futures by resampling historical observations
with replacement. They are useful when you have one observed history and want
many plausible future paths without fitting a full parametric model.

The primary bootstrap class names mirror ``arch.bootstrap``:
``IIDBootstrap``, ``StationaryBootstrap``, ``CircularBlockBootstrap``, and
``MovingBlockBootstrap``. The longer ``*ScenarioLoader`` names remain available
for compatibility and for users who prefer explicit loader names.

Each historical input is supplied with shape ``[n_observations, ...]``. The
loader samples a matrix of history indexes shaped ``[batch_size, horizon]`` and
uses that same index matrix for every exogenous field. This keeps multivariate
observations aligned. For example, if ``returns`` and ``volume`` came from the
same historical day, a sampled future keeps that day's return and volume
together.

.. code-block:: python

   import numpy as np
   from sda import IIDBootstrap

   historical_returns = np.array([0.01, -0.02, 0.015, 0.004, -0.008])
   historical_volume = np.array([100, 140, 120, 130, 110])

   scenarios = IIDBootstrap(
       initial_state=1_000_000.0,
       history={
           "returns": historical_returns,
           "volume": historical_volume,
       },
       horizon=12,
       n_scenarios=1_000,
       batch_size=128,
       seed=42,
   )

The bootstrap loaders preserve the empirical values in the history. They do
not extrapolate new values beyond the observed sample.

IID Bootstrap
~~~~~~~~~~~~~

``IIDBootstrap`` samples individual historical observations independently with
replacement.

For each scenario and time period:

1. Draw one integer index uniformly from ``0`` to ``n_observations - 1``.
2. Use that historical observation as the exogenous value for the period.
3. Repeat independently until the future reaches ``horizon`` periods.

This method is appropriate when the historical observations can be treated as
independent and identically distributed. It preserves the empirical marginal
distribution, but it breaks serial dependence such as autocorrelation,
volatility clustering, weekday patterns, or runs of high demand.

Circular Block Bootstrap
~~~~~~~~~~~~~~~~~~~~~~~~

``CircularBlockBootstrap`` samples fixed-length contiguous blocks with
replacement. The history is treated as circular, so a block can wrap from the
final observation back to the first observation.

For each scenario:

1. Draw a block start uniformly from ``0`` to ``n_observations - 1``.
2. Copy ``block_size`` consecutive observations, wrapping around the end of the
   history if needed.
3. Draw another block start and append another block.
4. Stop when the sampled path reaches ``horizon`` periods.

.. code-block:: python

   from sda import CircularBlockBootstrap

   scenarios = CircularBlockBootstrap(
       initial_state=1_000_000.0,
       history={"returns": historical_returns},
       horizon=60,
       n_scenarios=5_000,
       batch_size=256,
       block_size=12,
       seed=42,
   )

Block bootstrap methods are useful for dependent time series because
observations inside each sampled block keep their original local order. The
fixed block length is a modeling choice: shorter blocks behave more like IID
resampling, while longer blocks preserve longer historical runs but create fewer
independent restarts.

Moving Block Bootstrap
~~~~~~~~~~~~~~~~~~~~~~

``MovingBlockBootstrap`` samples fixed-length contiguous blocks with
replacement, but it does not wrap blocks around the end of the history. Each
block start is drawn from the valid starts ``0`` to
``n_observations - block_size``.

For each scenario:

1. Draw a block start uniformly from the valid non-wrapping starts.
2. Copy ``block_size`` consecutive observations.
3. Draw another valid block start and append another block.
4. Stop when the sampled path reaches ``horizon`` periods.

.. code-block:: python

   from sda import MovingBlockBootstrap

   scenarios = MovingBlockBootstrap(
       initial_state=1_000_000.0,
       history={"returns": historical_returns},
       horizon=60,
       n_scenarios=5_000,
       batch_size=256,
       block_size=12,
       seed=42,
   )

``block_size`` must be less than or equal to the number of historical
observations. Use moving block bootstrap when you want fixed-length blocks but
do not want artificial wraparound from the end of the sample back to the
beginning.

Stationary Bootstrap
~~~~~~~~~~~~~~~~~~~~

``StationaryBootstrap`` samples circular blocks with random lengths. Instead
of choosing a fixed block length, it uses an average block length.

For each scenario:

1. Draw the first history index uniformly.
2. At each later period, restart at a new uniformly sampled history index with
   probability ``p = 1 / block_size``.
3. Otherwise continue to the next historical observation, wrapping around the
   end of the history.
4. Repeat until the sampled path reaches ``horizon`` periods.

This produces geometrically distributed block lengths with mean
``block_size``.

.. code-block:: python

   from sda import StationaryBootstrap

   scenarios = StationaryBootstrap(
       initial_state=1_000_000.0,
       history={"returns": historical_returns},
       horizon=60,
       n_scenarios=5_000,
       batch_size=256,
       block_size=12,
       seed=42,
   )

The stationary bootstrap avoids hard block boundaries of one fixed length while
still preserving local serial dependence. It is often a good default when the
data are time dependent and there is no natural fixed block size. The
backward-compatible ``StationaryBlockBootstrapScenarioLoader`` class exposes
the same value as ``average_block_size``.

Choosing a Bootstrap Method
---------------------------

Use IID bootstrap when observations are plausibly independent, or when you only
need the empirical one-period distribution.

Use circular block bootstrap when consecutive observations matter and you want
a clear fixed block length, such as seven days for weekly demand patterns or
twelve months for annual monthly data.

Use moving block bootstrap when consecutive observations matter, you want a
clear fixed block length, and wrapping the last historical observations back to
the first observation would be inappropriate.

Use stationary block bootstrap when consecutive observations matter but you
prefer random block lengths around an average length.

The current loaders require ``block_size`` for circular, moving, and
stationary bootstrap. The backward-compatible
``StationaryBlockBootstrapScenarioLoader`` uses ``average_block_size`` for the
stationary average block length. They do not estimate an optimal block length
automatically. Choose these values from domain knowledge, seasonality, the
decision horizon, or separate statistical analysis.

Implementation Notes
--------------------

Bootstrap sampling is done at iteration time, so iterating over a loader with
the same ``seed`` produces the same scenario batches again. Different seeds
produce different bootstrap futures.

All exogenous fields in a bootstrap loader must have the same number of
historical observations. Additional trailing dimensions are preserved. For
example, a historical array shaped ``[365, 4]`` becomes sampled futures shaped
``[batch_size, horizon, 4]``.

``initial_state`` follows the same rules as ``ArrayScenarioLoader``: it can be
a scalar, a vector with one entry per scenario, or a mapping whose values are
scalars or per-scenario vectors.
