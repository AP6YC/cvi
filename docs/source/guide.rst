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
   Check the read-only ``is_defined`` property rather than interpreting that sentinel as an optimal clustering result; a valid score may also be zero.
   An undefined batch evaluation emits a ``RuntimeWarning``; incremental updates remain silent while an index is not yet defined.
* Use a fresh instance when comparing independent datasets or partitions.

See :doc:`choosing` for differences between indices.
In particular, ``CONN`` has additional preprocessing and backend requirements described in :doc:`conn`.

Updating a partition
--------------------

After batch or incremental initialization, a sample can be added with ``get_cvi`` and (except for ``CONN``) samples or tracked sufficient statistics can be removed, merged, or split:

.. code-block:: python

   value = index.get_cvi(new_sample, new_label)
   value = index.remove(existing_sample, existing_label)
   value = index.merge(target_label=20, source_label=10)
   value = index.split(
       retained_label=20,
       new_label=30,
       count=prototype_count,
       centroid=prototype_centroid,
       compactness=prototype_compactness,
       covariance=prototype_covariance,
   )

These operations update the object in place and return its new criterion value.
``merge`` retains the target label and deletes the source label.
``split`` retains the existing label for the residual cluster and assigns the split-off subset to an unused new label without changing the total sample count.
Removing a cluster's final sample deletes that label; removing the final sample in the whole index returns the object to its initial empty state.

Every split requires the subset's sample count and centroid.
For non-singletons, compactness-based indices require the centered sum of squared distances, while ``rCIP`` requires the unregularized unbiased sample covariance; ``PS`` needs neither additional statistic.
Singleton compactness and covariance are inferred as zero.

The package stores sufficient statistics rather than the original dataset.
Consequently, the caller must ensure that a sample passed to ``remove`` really belongs to the supplied label.
Similarly, split statistics must describe a true subset of the retained cluster.
Invalid labels, inconsistent statistics, changed feature dimensions, and attempts to merge a label with itself raise an error.

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
