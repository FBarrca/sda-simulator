Development
===========

This project uses uv for local development.

Run tests:

.. code-block:: bash

   uv run pytest

Build the documentation:

.. code-block:: bash

   uv run --group docs sphinx-build -b html docs docs/_build/html

If the docs dependencies are already installed in the active Python
environment, the equivalent direct command is:

.. code-block:: bash

   python3 -m sphinx -b html docs docs/_build/html

During documentation work, build with nitpicky mode and warnings as errors:

.. code-block:: bash

   python3 -m sphinx -b html docs docs/_build/html -n -W

Build the distribution wheel:

.. code-block:: bash

   uv build --wheel

The built wheel should contain the ``sda`` package only. The ``examples/``
directory is source-tree sample code, not installed package API.
