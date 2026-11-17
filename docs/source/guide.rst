Getting Started
===============

Installation
------------

Install the latest release from PyPI:

.. code-block:: console

   python -m pip install cvi

or the development version from GitHub:

.. code-block:: console

   python -m pip install git+https://github.com/AP6YC/cvi.git

The ``numba`` and ``jax`` extras enable optional numerical backends
(:doc:`backends`), and the ``art`` extra enables incremental ``CONN``
(:doc:`conn`).

Batch evaluation
----------------

A batch is a two-dimensional NumPy array with one sample per row and one feature
per column. Labels are a one-dimensional array with one integer per sample.

.. code-block:: python

   import numpy as np
   import cvi

   samples = np.array([
       [0.0, 0.1],
       [0.2, 0.0],
       [2.8, 3.0],
       [3.1, 2.9],
   ])
   labels = np.array([0, 1, 2, 2])

   index = cvi.CH()
   value = index.get_cvi(samples, labels)

The call updates ``index`` in place and returns the criterion value.

Streaming evaluation
--------------------

Pass a one-dimensional sample and a scalar label to update an index one sample
at a time:

.. code-block:: python

   index = cvi.CH()
   values = np.empty(len(labels))

   for i, (sample, label) in enumerate(zip(samples, labels)):
       values[i] = index.get_cvi(sample, int(label))

An object may also receive samples after its initial batch. To add, remove,
move, or merge in bulk or by trial, see :doc:`operations`.

State and input rules
---------------------

CVI objects are stateful accumulators:

* Batch data have shape ``(n_samples, n_features)`` and incremental samples have
  shape ``(n_features,)``. The feature count is fixed for the object's lifetime.
* Labels are arbitrary integers. They need not be consecutive or start at zero.
* Batch initialization needs at least two distinct labels, and an object accepts
  only one batch initialization.
* Scores are ``numpy.nan`` while an index is undefined, such as before enough
  clusters have been seen. Test with ``numpy.isnan``; a score of zero is valid.
  An undefined one-shot batch evaluation emits a ``RuntimeWarning``; incremental
  and mini-batch updates stay silent.
* Use a fresh object for each independent dataset or partition.

The object also exposes its state and lifecycle:

.. code-block:: python

   index.n_clusters  # number of tracked clusters
   index.n_samples   # number of represented samples
   index.labels      # known labels, in first-seen order

   fresh = index.clone()  # new, unfitted object with the same configuration
   index.reset()          # clear all data, keeping configuration and backend

Selecting an index by name
--------------------------

Use :func:`cvi.create_cvi` when the index name comes from configuration or a
loop. Each call creates a fresh object:

.. code-block:: python

   indices = [cvi.create_cvi(name) for name in ["CH", "cSIL", "rCIP"]]

Names are case-insensitive, so ``"csil"`` selects ``cSIL``. Constructor options
pass through: ``cvi.create_cvi("CONN", model_type="KMeans")`` is the same as
``cvi.CONN(model_type="KMeans")``.

Index metadata
--------------

Every index has an ``info`` class attribute with its name, range, optimization
direction, and numerical backends:

.. doctest::

   >>> import cvi
   >>> cvi.CH.info
   CVIInfo(name='Calinski-Harabasz', name_short='CH', index_min=0.0, index_max=inf, optimality='max', backends=('numpy', 'numba', 'jax'))

Use ``optimality`` rather than assuming larger is better. ``backends`` lists
implemented backends even when their optional dependencies are not installed.

An instance's ``capabilities`` property reports which operations its
configuration supports:

.. doctest::

   >>> cvi.CONN(model_type="KMeans").capabilities
   CVICapabilities(batch=True, incremental=False, merge=True, remove=False, split=True, mini_batch=False, trial=False, move=False)

Capabilities describe the selected numerical and prototype backends, not
whether the object is initialized or an optional dependency is installed.
``batch`` is one-shot ``get_cvi(samples, labels)``; ``mini_batch`` is
``update_batch``; ``trial`` is scoring with ``commit=False``; ``move`` is
``move``. See :doc:`operations` for each.
