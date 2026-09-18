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

   cvi.CVI.is_defined
   cvi.CVI.get_cvi
   cvi.CVI.remove
   cvi.CVI.merge
   cvi.CVI.split

Evaluation status
-----------------

Every index exposes the read-only :attr:`cvi.CVI.is_defined` property to
distinguish a computed score from the ``0.0`` fallback for undefined states.
