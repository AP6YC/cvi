Legacy v0 API
=============

The ``cvi.compat.v0`` namespace preserves the pre-1.0 incremental interface for
existing applications. New code should use the stateful top-level classes in
:doc:`api`, which provide a consistent ``get_cvi`` interface for batch and
incremental updates.

.. list-table:: Legacy-to-current names
   :header-rows: 1

   * - Legacy
     - Current
   * - ``iCH``
     - :class:`cvi.CH`
   * - ``iDB``
     - :class:`cvi.DB`
   * - ``iSIL``
     - :class:`cvi.cSIL`
   * - ``iGD43``
     - :class:`cvi.GD43`
   * - ``iGD53``
     - :class:`cvi.GD53`
   * - ``iPS``
     - :class:`cvi.PS`
   * - ``iXB``
     - :class:`cvi.XB`

Legacy objects use ``update`` and expose implementation-specific state. They
are retained for compatibility, but are not part of the primary API reference.

