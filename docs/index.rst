SDA Simulator V2
================

``sda`` is a small Python package for Sequential Decision Analytics
simulation. It helps you evaluate a decision rule by rolling it through many
sampled futures and collecting metric distributions.

Use it when a decision repeats over time, each decision changes the next state,
and uncertainty matters. Typical examples are inventory replenishment,
dispatching, pricing, staffing, preventive maintenance, and resource
allocation.

The core workflow is always the same:

.. code-block:: python

   policy = MyPolicy(...)
   model = MyModel(policy)
   data = MyDataModule(...)
   result = evaluate(model, data)

The installed library package is ``sda``. The repository also contains
``examples/`` as source-tree demonstration code, but examples are not part of
the installed library API.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   quickstart
   workflow
   concepts
   use_cases
   data
   metrics
   architecture
   api
   examples
   development

Start Here
----------

Read :doc:`quickstart` first. It walks through a complete tiny simulation and
shows exactly what gets written in a policy, model, data module, and result
query.

Then use:

* :doc:`workflow` when you are structuring your own project.
* :doc:`concepts` when the SDA vocabulary is new.
* :doc:`use_cases` when you want to map a real problem to state, decisions,
  uncertainty, and metrics.
* :doc:`data` when choosing between array, generated, bootstrap, or custom
  scenario data.
* :doc:`metrics` when reading logged results or adding domain metrics.
* :doc:`architecture` when changing the package itself.

Core Pieces
-----------

The framework separates the simulation lifecycle into small pieces:

``DataModule``
   Owns scenario setup and yields ``ScenarioBatch`` objects.

``Policy``
   Chooses decisions from observed state and completed history.

``SDAModel``
   Defines initial state, transitions, costs, and optional diagnostics.

``evaluate``
   Runs the standard data lifecycle, rollout, and default cost metrics.

``Simulator``
   The reusable configured runner for metric, history, and tracking settings.

``SimulationResult``
   Exposes logged metric distributions, records, percentiles, and risk
   measures.

The important timing rule is simple: a policy decides before the current
period's uncertainty is revealed. Put information known before the decision in
``state``. Put future uncertainty in the data module's exogenous paths.
