Inventory Example
=================

The inventory example models a lost-sales, single-item warehouse. It evaluates
an order-up-to policy against Poisson demand futures and reports cost,
inventory, stockout, and fill-rate metrics.

Run it from the repository root:

.. code-block:: bash

   uv run -m examples.inventory

or:

.. code-block:: bash

   uv run -m examples.inventory.main

If your environment already has the package requirements installed, the direct
Python form also works:

.. code-block:: bash

   python3 -m examples.inventory

Source Layout
-------------

The example is split by responsibility:

* ``examples/inventory/data.py`` defines the data module.
* ``examples/inventory/policies.py`` defines policies.
* ``examples/inventory/models.py`` defines the domain model.
* ``examples/inventory/metrics.py`` lists the domain metric names emitted by
  the model.
* ``examples/inventory/main.py`` wires the example together.

Data Module
-----------

``InventoryDataModule`` owns the scenario configuration and yields
``ScenarioBatch`` objects from ``batches(stage)``. Each ``ScenarioSpec``
contains one Poisson demand path and one initial inventory value:

.. code-block:: python

   data = InventoryDataModule(
       horizon=12,
       n_scenarios=1000,
       batch_size=128,
       initial_inventory=50,
       demand_lambda=20,
       seed=42,
   )

Use a fixed seed when you want repeatable policy comparisons.

Policy
------

``OrderUpToPolicy`` is a policy function approximation. It orders nothing while
inventory is at or above the reorder point. Once inventory falls below the
reorder point, it orders enough to reach the target level:

.. code-block:: python

   policy = OrderUpToPolicy(reorder_point=30, order_up_to=80)

The policy acts inside each SimPy scenario process, returning one order
quantity for the current inventory state.

Model
-----

``InventoryModel`` registers a daily SimPy process in ``build``. Each day it
asks the policy for an order, serves demand, updates inventory, logs cost, and
emits domain metrics with the scenario recorder.

.. code-block:: python

   model = InventoryModel(
       policy=policy,
       order_cost=1.0,
       holding_cost=0.1,
       stockout_cost=8.0,
   )

Metrics
-------

The model emits inventory-specific metrics directly:

.. code-block:: python

   from sda import evaluate

   result = evaluate(model, data)

``inventory`` logs ending inventory by period, ``stockout`` logs whether a
scenario lost sales in a period, and ``fill_rate`` logs the share of demand
served in each period. ``total_cost`` is logged when the recorder closes.

Interpreting Results
--------------------

The main module prints aggregate values such as:

.. code-block:: text

   Total cost mean: ...
   Total cost p95: ...
   Total cost CVaR 95: ...
   Inventory t=5 mean: ...
   Fill rate mean: ...
   Stockout rate: ...

Use ``mean`` for average performance, percentiles for distribution shape, and
``cvar(0.95)`` for the average cost in the worst tail of simulated outcomes.
