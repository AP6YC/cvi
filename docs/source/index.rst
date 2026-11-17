.. image:: https://github.com/AP6YC/FileStorage/blob/main/cvi/header.png?raw=true
   :alt: Cluster Validity Indices
   :align: center

``cvi``: Cluster Validity Indices in Python
===========================================

``cvi`` scores hard partitions when reference labels are unavailable. It
evaluates a complete labeled dataset at once, or keeps the score up to date as
samples arrive, with the same result either way. It is the Python counterpart
of the `ClusterValidityIndices.jl <https://github.com/AP6YC/ClusterValidityIndices.jl>`_
Julia package.

The animations below show CH, DB, and cSIL updating as samples arrive (solid
lines) and landing on the batch value for the full dataset (dashed lines). The
top row streams synthetic Gaussian blobs and the bottom row the Iris dataset,
first in class order and then shuffled.

.. image:: _static/readme/incremental-cvi-synthetic-class-order.gif
   :alt: Incremental CVIs on class-ordered synthetic Gaussian blobs
   :width: 49%
.. image:: _static/readme/incremental-cvi-synthetic-shuffled.gif
   :alt: Incremental CVIs on shuffled synthetic Gaussian blobs
   :width: 49%
.. image:: _static/readme/incremental-cvi-iris-class-order.gif
   :alt: Incremental CVIs on the class-ordered Iris dataset
   :width: 49%
.. image:: _static/readme/incremental-cvi-iris-shuffled.gif
   :alt: Incremental CVIs on the shuffled Iris dataset
   :width: 49%

New to CVIs? Read :doc:`background`. Otherwise, start with :doc:`guide`, then
use :doc:`choosing` to pick an index.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   background
   guide
   batch_incremental
   operations
   choosing
   backends
   conn
   api
   legacy

Acknowledgements
----------------

The implementations are largely derived from
`ClusterValidityIndices.jl <https://github.com/AP6YC/ClusterValidityIndices.jl>`_,
by the same authors:

* Sasha Petrenko - petrenkos@mst.edu - `github.com/AP6YC <https://github.com/AP6YC>`_
* Nik Melton - nmmz76@mst.edu - `github.com/NiklasMelton <https://github.com/NiklasMelton>`_

Related Python packages: `validclust <https://github.com/crew102/validclust>`_
and `clusterval <https://github.com/Nuno09/clusterval>`_.

The logo uses the `Ethnocentric <https://www.1001fonts.com/ethnocentric-font.html>`_
font and the `cluster computing icon <https://www.flaticon.com/free-icon/cluster_5464694>`_
by IconBaandar on Flaticon.

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
