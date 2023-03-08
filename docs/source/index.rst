.. cluster_validit_indices documentation master file, created by
   sphinx-quickstart on Thu Aug 18 11:37:46 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to cvi's documentation!
====================================================


These pages serve as the official documentation for the `cvi` Python package, the Python implementation of the `ClusterValidityIndices.jl <https://github.com/AP6YC/ClusterValidityIndices.jl>`_ Julia package.

Cluster Validity Indices (CVI) tackle the problem of judging the performance of an unsupervised/clustering algorithm without the availability of truth or supervisory labels, resulting in metrics of under- or over-partitioning.
Furthermore, Incremental CVIs (ICVI) are variants of these ordinarily batch algorithms that enable an online and computationally tractable method of evaluating the performance of a clustering algorithm as it clusters while being numerically equivalent to their batch counterparts.

The purpose of this package is to provide a home for the development and use of these CVIs and ICVIs.

..
   For a list of all CVIs available from the package, see the [Implemented CVI List](@ref cvi-list-page) page.

..
   See the [Index](@ref main-index) for the complete list of documented functions and types.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   background
   usage
   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
