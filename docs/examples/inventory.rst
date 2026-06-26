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

* ``examples/inventory/dataloader.py`` defines scenario generation.
* ``examples/inventory/policies.py`` defines policies.
* ``examples/inventory/models.py`` defines the domain model.
* ``examples/inventory/metrics.py`` defines domain-specific metrics.
* ``examples/inventory/main.py`` wires the example together.

Scenario Loader
---------------

``InventoryScenarioLoader`` is a custom ``ScenarioLoader``. It generates
Poisson demand paths with shape ``[batch_size, horizon]`` and returns
``ScenarioBatch`` objects with one initial inventory value per scenario:

.. code-block:: python

   scenarios = InventoryScenarioLoader(
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

The policy operates on the full batch at once, returning one order quantity per
scenario.

Model
-----

``InventoryModel`` defines the domain dynamics:

* ``transition`` computes available inventory, sales, and ending inventory.
* ``cost`` combines order cost, holding cost, and lost-sales penalty.
* ``info`` exposes intermediate values used by custom metrics.

.. code-block:: python

   model = InventoryModel(
       policy=policy,
       order_cost=1.0,
       holding_cost=0.1,
       stockout_cost=8.0,
   )

Metrics
-------

The example combines built-in cost metrics with inventory-specific metrics:

.. code-block:: python

   simulator = Simulator(
       metrics=[
           StepCostMetric(),
           TotalCostMetric(),
           InventoryMetric(),
           StockoutMetric(),
           FillRateMetric(),
       ]
   )

``InventoryMetric`` logs ending inventory by period, ``StockoutMetric`` logs
whether a scenario lost sales in a period, and ``FillRateMetric`` logs the
share of demand served in each period.

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
