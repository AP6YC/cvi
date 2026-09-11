.. image:: https://github.com/AP6YC/FileStorage/blob/main/cvi/header.png?raw=true
   :alt: Cluster Validity Indices
   :align: center

``cvi``: Cluster Validity Indices in Python
===========================================

These pages serve as the official documentation for the ``cvi`` Python package, the Python implementation of the `ClusterValidityIndices.jl <https://github.com/AP6YC/ClusterValidityIndices.jl>`_ Julia package.

``cvi`` provides batch and incremental cluster validity indices (CVIs) for evaluating
hard partitions when reference labels are unavailable.
The package offers a shared stateful interface for processing complete datasets (in a batch) or monitoring a clustering stream over time (incrementally).

Cluster Validity Indices (CVIs) tackle the problem of judging the performance of an unsupervised/clustering algorithm without the availability of truth or supervisory labels, resulting in metrics of under- or over-partitioning.
Furthermore, Incremental CVIs (ICVIs) are variants of these ordinarily batch algorithms that enable an online and computationally tractable method of evaluating the performance of a clustering algorithm as it clusters while being numerically equivalent to their batch counterparts.

Start with :doc:`guide`, then use :doc:`choosing` to select an index.
``CONN`` has a specialized prototype-based interface covered in :doc:`conn`.


.. toctree::
   :maxdepth: 1
   :caption: Contents:

   background
   guide
   choosing
   conn
   api
   legacy

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
