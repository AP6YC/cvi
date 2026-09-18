Getting Started
===============

Installation
------------

Install the latest release from PyPI:

.. code-block:: console

   python -m pip install cvi

To install the current development version directly from GitHub:

.. code-block:: console

   python -m pip install git+https://github.com/AP6YC/cvi.git

Batch evaluation
----------------

A batch is a two-dimensional NumPy array with one sample per row and one feature per column.
Labels are a one-dimensional array with one integer label per sample.

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

The call updates ``index`` in place and returns the resulting criterion value.
A CVI object accepts only one batch initialization.
Create a new object to evaluate an independent partition.

Optional CPU acceleration
-------------------------

Install the Numba extra to enable compiled numerical kernels:

.. code-block:: console

   python -m pip install "cvi[numba]"

For a development checkout, use ``python -m pip install -e ".[numba]"``.
Select the backend when constructing an index:

.. code-block:: python

   index = cvi.XB(backend="numba")
   value = index.get_cvi(samples, labels)
   assert index.backend == "numba"

NumPy remains the default. Numba is supported by ``CH``, ``WB``, ``DB``, ``XB``,
``GD43``, ``GD53``, ``PS``, and ``cSIL``. ``CONN`` and ``rCIP`` reject
``backend="numba"``. CONN's numerical backend is separate from ``model_type``.
Selection is per object and remains fixed through removal of all samples and
reinitialization. The batch, incremental, remove, and merge APIs are unchanged.

Compiled kernels accelerate grouping, compactness, centroid distances, and
cSIL batch dissimilarities. Dtype-sensitive means and raw-moment reductions
remain in NumPy. Unsupported array types (including float16 and non-native
byte order) use NumPy for the affected operations. Floating-point rounding may
differ; bitwise equivalence is not guaranteed. Existing undefined NaN/inf scores
retain their meaning. Unchanged paths, including CH/WB streaming updates and
cSIL incremental updates, are not accelerated.

The first invocation for a new input type/layout incurs compilation. Later
calls reuse compiled code, with a disk cache across processes. Include that
initial latency when measuring short tasks, and warm up kernels before measuring
steady-state throughput. Numba is imported for the numerical backend only when
selected; CONN's existing ART dependency can independently install and use it.
Selecting Numba without the dependency installed raises an installation hint
rather than silently selecting another backend.

Streaming evaluation
--------------------

Pass a one-dimensional sample and a scalar label to update an index
incrementally:

.. code-block:: python

   index = cvi.CH()
   values = np.empty(len(labels))

   for i, (sample, label) in enumerate(zip(samples, labels)):
       values[i] = index.get_cvi(sample, int(label))

The same object may also receive incremental samples after its initial batch
call. Feature dimensionality must remain constant throughout the object's
lifetime.

State and Input Rules
---------------------

All CVI implementations are stateful accumulators.
Keep the following rules in mind:

* Batch data have shape ``(n_samples, n_features)`` and incremental samples have shape ``(n_features,)``.
* Labels are arbitrary integer identifiers.
   They need not be consecutive or start at zero.
* Batch initialization requires at least two distinct labels, and a second
  batch call on the same object is rejected.
* Criterion values are ``0.0`` while an index is not defined, such as before enough clusters have been observed.
   Do not interpret that sentinel as an optimal clustering result.
* Use a fresh instance when comparing independent datasets or partitions.

See :doc:`choosing` for differences between indices.
In particular, ``CONN`` has additional preprocessing and backend requirements described in :doc:`conn`.

Updating a partition
--------------------

After batch or incremental initialization, a sample can be added with ``get_cvi`` and (except for ``CONN``) an existing sample can be removed or two clusters can be merged:

.. code-block:: python

   value = index.get_cvi(new_sample, new_label)
   value = index.remove(existing_sample, existing_label)
   value = index.merge(target_label=20, source_label=10)

These operations update the object in place and return its new criterion value.
``merge`` retains the target label and deletes the source label.
Removing a cluster's final sample deletes that label; removing the final sample in the whole index returns the object to its initial empty state.

The package stores sufficient statistics rather than the original dataset.
Consequently, the caller must ensure that a sample passed to ``remove`` really belongs to the supplied label.
Invalid labels, inconsistent samples, changed feature dimensions, and attempts to merge a label with itself raise an error.

Index metadata
--------------

Every implementation exposes an ``info`` class attribute describing its name,
range, and optimization direction:

.. doctest::

   >>> import cvi
   >>> cvi.CH.info
   CVIInfo(name='Calinski-Harabasz', name_short='CH', index_min=0.0, index_max=inf, optimality='max')

Use ``optimality`` rather than assuming that a larger value is always better.

Acknowledgements
----------------

Derivation
^^^^^^^^^^

The incremental and batch CVI implementations in this package are largely derived from the following Julia language implementations by the same authors of this package:

* `ClusterValidityIndices.jl <https://github.com/AP6YC/ClusterValidityIndices.jl>`_

Authors
^^^^^^^

The principal authors of the `cvi` pacakge are:

* Sasha Petrenko - petrenkos@mst.edu - `github.com/AP6YC <https://github.com/AP6YC>`_
* Nik Melton - nmmz76@mst.edu - `github.com/NiklasMelton <https://github.com/NiklasMelton>`_

Related Projects
^^^^^^^^^^^^^^^^

If this package is missing something that you need, feel free to check out some related Python cluster validity packages:

* `validclust <https://github.com/crew102/validclust>`_
* `clusterval <https://github.com/Nuno09/clusterval>`_
