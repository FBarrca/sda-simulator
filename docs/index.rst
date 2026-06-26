SDA Simulator V2
================

SDA Simulator V2 is a minimal class-based framework for Sequential Decision
Analytics simulation. It helps you evaluate policies for sequential decision
problems by rolling many sampled futures through a model and collecting metric
distributions.

The installed library package is ``sda``. The repository also contains
``examples/`` as source-tree demonstration code, but examples are not part of
the installed library API.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   concepts
   quickstart
   api
   examples
   development

Core Pieces
-----------

The framework separates the simulation lifecycle into small pieces:

* ``ScenarioLoader`` produces batches of exogenous futures.
* ``SDAModel`` defines decisions, transitions, costs, and domain information.
* ``Simulator`` rolls trajectories forward.
* ``MetricStore`` stores raw metric observations.
* ``SimulationResult`` exposes distribution summaries, percentiles, and risk metrics.

Start with :doc:`concepts` if you want the modeling vocabulary, or jump to
:doc:`quickstart` if you want to run a minimal simulation first.
