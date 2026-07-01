Examples
========

Examples live in the repository under ``examples/``. They are useful for
learning and testing, but they are not installed as part of the library package.

Run example commands from the repository root so both ``sda`` and
``examples`` are importable.

If you are deciding whether your own problem fits ``sda``, read
:doc:`use_cases` first. The pages below then show two concrete source-tree
applications.

.. toctree::
   :maxdepth: 2

   examples/inventory
   examples/logistics

Available Examples
------------------

Inventory
   A compact lost-sales inventory model that evaluates an order-up-to policy
   against Poisson demand futures.

Logistics Dispatch
   A fuller walkthrough of a Spanish road-freight dispatch problem with
   warehouses, vehicles, orders, stochastic events, policy comparison,
   tail-risk metrics, and generated visualizations.
