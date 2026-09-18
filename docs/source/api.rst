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
``CONN`` does not currently implement ``remove`` or ``merge``.

.. autosummary::

   cvi.CVI.get_cvi
   cvi.CVI.remove
   cvi.CVI.merge

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
