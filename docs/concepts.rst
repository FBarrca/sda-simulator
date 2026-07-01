Concepts
========

Sequential Decision Analytics evaluates decisions that unfold over time. A
decision changes system state, the new state affects later decisions, and
uncertainty enters through events.

In ``sda``, SimPy owns the event clock. The framework supplies a small
contract around SimPy so policies, models, data, and metrics stay separated.

The Small Mental Model
----------------------

``state``
   What the decision maker knows now. Examples: inventory, open orders,
   vehicle locations, machine health, or cash position.

``policy``
   The decision rule. ``Policy.act(state, env, history)`` returns an action
   from the current state, the current SimPy environment, and completed metric
   history.

``scenario``
   One independent future. A ``ScenarioSpec`` has a ``scenario_id``,
   ``end_time``, optional ``initial_state``, optional ``seed``, and arbitrary
   ``data`` used by the model's SimPy processes.

``model``
   The domain simulation. ``SDAModel.build(env, scenario, recorder)`` registers
   processes on a SimPy environment and returns model state.

``recorder``
   The metric logger. Processes call ``recorder.cost``, ``recorder.log``, or
   ``recorder.trajectory`` as events happen.

How The Pieces Run
------------------

.. code-block:: text

   data.prepare_data()
   data.setup(stage)
   for batch in data.batches(stage):
       for scenario in batch.scenarios:
           create simpy.Environment()
           call model.build(env, scenario, recorder)
           run env until scenario.end_time
           call model.finalize(state, scenario, recorder)
           close recorder

``env.now`` is the time stored on metric rows. There is no separate framework
time index.

Scenarios And Batches
---------------------

``ScenarioBatch`` is only a transport container for independent
``ScenarioSpec`` objects. Built-in data modules help create those specs:

* ``ArrayDataModule`` slices already-built paths.
* ``GeneratorDataModule`` calls a generator for each batch.
* ``BootstrapDataModule`` resamples historical observations.

For array-style sources, full paths are scenario-first:

.. code-block:: text

   [n_scenarios, horizon, ...]

The model receives one scenario at a time through ``scenario.data`` and decides
how to use that data inside SimPy processes.

Metrics
-------

Each metric row has:

* ``name``
* ``value``
* ``scenario_id``
* ``time``
* ``level``: ``"event"`` or ``"trajectory"``
* ``tags``

``recorder.cost(value)`` logs an event-level ``cost`` and accumulates
``total_cost``. ``recorder.close()`` writes the final trajectory-level
``total_cost`` row.

Policy Classes
--------------

Warren Powell's unified framework groups policies into broad classes such as
policy function approximations, cost function approximations, value function
approximations, and direct lookahead approximations. ``sda`` does not force
one class. Every policy is just an implementation of ``Policy.act(...)``.

When SDA Fits
-------------

SDA is a good fit when decisions repeat over time, state carries forward,
uncertainty changes outcomes, and you care about a distribution of possible
results rather than a single forecast.
