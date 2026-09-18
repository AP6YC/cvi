API Reference
=============

The classes below are the supported top-level interface. Import them directly
from ``cvi`` rather than from their implementation modules.

Base class
----------

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst

   cvi.CVI

Indices
-------

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst

   cvi.CH
   cvi.CONN
   cvi.cSIL
   cvi.DB
   cvi.GD43
   cvi.GD53
   cvi.PS
   cvi.rCIP
   cvi.WB
   cvi.XB

Common methods
--------------

All indices inherit the common update interface from :class:`cvi.CVI`.
``CONN`` does not currently implement ``remove``, ``merge``, or ``split``.

.. autosummary::

   cvi.CVI.get_cvi
   cvi.CVI.update_many
   cvi.CVI.remove
   cvi.CVI.merge
   cvi.CVI.split

Functional JAX batch interface
------------------------------

Install the optional ``jax`` extra and enable JAX x64 before using this module.
See :doc:`guide` for label encoding, precision, and supported operations.

.. py:module:: cvi.jax

.. py:function:: batch_state(data, labels, *, n_clusters)

   Return an immutable ``BatchState`` pytree of device-resident sufficient
   statistics. Labels must be dense and every cluster must be represented.
   ``n_clusters`` must be static under JIT.

.. py:function:: evaluate(state, *, index)

   Return a JAX scalar for ``index="CH"``, ``"WB"``, or ``"XB"``. The index
   name must be static under JIT.

.. py:function:: batch_cvi(data, labels, *, n_clusters, index)

   Compute batch statistics and evaluate the chosen index in one functional
   call. Suitable for composition with ``jit``, ``vmap``, and differentiation
   with fixed labels. No host scalar conversion is performed.

Functional JAX streaming interface
----------------------------------

.. py:function:: empty_stream(*, capacity, n_features, index)

   Allocate an immutable ``StreamingState`` of fixed-shape device arrays.
   Capacity and feature count must be positive static integers.

.. py:function:: stream_from_batch(state, *, capacity, index)

   Pad a valid ``BatchState`` into a stream without changing its statistics.
   Capacity must accommodate all existing clusters.

.. py:function:: stream_update(state, sample, slot, *, index)

   Return ``(new_state, score)`` after one incremental addition. Slots are
   integers in ``[0, capacity)``; they need not be contiguous.

.. py:function:: stream_chunk(state, data, slots, *, index, return_history=True)

   Return ``(new_state, history)`` using a compiled scan, or a final scalar
   when ``return_history=False``. Empty chunks are no-ops. Invalid slot values
   or nonfinite data return unchanged state and NaN output for the whole call.
   Shape/dtype errors raise ``ValueError``. Options must be static under JIT.

.. py:function:: evaluate_stream(state, *, index)

   Evaluate active clusters; zero/one-cluster states return zero. The index
   must match the state's distance layout (CH/WB versus XB).
