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

Mini-batch updates
------------------

Use ``update_batch`` to append labeled chunks and score the complete accumulated
partition after each chunk:

.. code-block:: python

   index = cvi.CH()
   assert index.capabilities.mini_batch
   for start in range(0, len(samples), 1024):
       value = index.update_batch(samples[start:start + 1024],
                                  labels[start:start + 1024])

``CH``, ``WB``, ``DB``, ``XB``, ``GD43``, ``GD53``, ``PS``, and ``cSIL`` support
this operation with NumPy and Numba. ``rCIP`` supports it with NumPy only.
``CONN(model_type="Fuzzy")`` provides a sequential convenience wrapper;
its KMeans backends and all JAX configurations raise ``NotImplementedError``.
Numba uses the shared NumPy aggregation kernels and compiled
distance/dissimilarity kernels for the final rebuild.

Each chunk contains only new observations; previous labels and observations
remain in the accumulated partition. The method can initialize a fresh object
or continue after batch initialization, sample updates, other chunks, or
structural operations. New integer labels are appended in first-seen order.
The feature count must remain constant. An empty ``(0, n_features)`` chunk
is a no-op, including on a fresh object. A chunk containing only one label is
allowed; an undefined cumulative score returns NaN without warning.

Inputs must contain finite real numbers and one integer label per row. The
complete update is staged before committing, so a failed call leaves the
object unchanged. Except for FuzzyART CONN, chunks are summarized in float64
using centered moments, then merged into the existing cluster statistics.
Floating-point reduction order differs from batch and sample updates; bitwise equivalence is not
guaranteed. Legacy float32 batch initialization retains its original reduction
precision, which later chunks cannot recover.

FuzzyART CONN calls the existing incremental update for every row in order,
including all learning and scoring steps, and returns only the final score.
Chunk inputs must already use a fixed ART input scale (normally ``[0, 1]``);
``normalize_batch`` does not normalize chunks. The complete ART state is
copied for each nonempty chunk to preserve it on failure, so this wrapper
does not promise a speedup. See :doc:`conn`.

``cSIL`` keeps centered compactness and residual sums through batch, sample,
chunk, remove, merge, and split operations. Its dissimilarities are computed
from these centered summaries, avoiding subtraction of large raw moments.
Older serialized objects can recover available centered state from their
stored dissimilarities, but cannot recover precision already lost.

``rCIP`` combines unregularized sample covariances through centered scatter and
applies its existing regularization once to the merged covariance. Its one-shot
batch calculation also uses centered float64 covariance reductions. Covariance
memory remains proportional to the number of clusters times the square of the
feature count. Large coordinate offsets still limit centroid precision, so
different chunk boundaries can introduce small numerical differences.

No intermediate per-sample scores are produced. Use sample updates or JAX's
sequential ``update_many`` when those scores are needed. Chunking amortizes
grouping and score evaluation, with input working memory bounded by the chunk
size. Pairwise indices still retain quadratic storage in the cluster count.
Larger chunks generally improve throughput at the cost of memory and less
frequent scores. One-shot batch evaluation remains useful when all data fit
in memory and only one score is needed.

Selecting an index by name
--------------------------

Use :func:`cvi.create_cvi` when the index name comes from configuration or a
loop. Each call creates a fresh index object:

.. code-block:: python

   names = ["CH", "cSIL", "rCIP"]
   indices = [cvi.create_cvi(name) for name in names]

Names are case-insensitive, so ``cvi.create_cvi("csil")`` selects ``cSIL``.
Constructor options pass through unchanged. For example,
``cvi.create_cvi("CONN", model_type="KMeans")`` configures CONN just like
``cvi.CONN(model_type="KMeans")``.

State and Input Rules
---------------------

All CVI implementations are stateful accumulators.
Keep the following rules in mind:

* Batch data have shape ``(n_samples, n_features)`` and incremental samples have shape ``(n_features,)``.
* Labels are arbitrary integer identifiers.
   They need not be consecutive or start at zero.
* Batch initialization requires at least two distinct labels, and a second
  batch call on the same object is rejected.
* Criterion values are ``numpy.nan`` while an index is not defined, such as before enough clusters have been observed.
   Use ``numpy.isnan`` before consuming a result; a valid score may be zero.
   An undefined one-shot batch evaluation emits a ``RuntimeWarning``; incremental and mini-batch updates remain silent while an index is not yet defined.
* Use a fresh instance when comparing independent datasets or partitions.

See :doc:`choosing` for differences between indices.
In particular, ``CONN`` has additional preprocessing and backend requirements described in :doc:`conn`.

Updating a partition
--------------------

After batch or incremental initialization, a sample can be added with ``get_cvi``.
Most indices also support removing samples, merging clusters, and splitting
tracked sufficient statistics:

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

For the other indices, every split requires the subset's sample count and centroid.
For non-singletons, compactness-based indices require the centered sum of squared distances, while ``rCIP`` requires the unregularized unbiased sample covariance; ``PS`` needs neither additional statistic.
Singleton compactness and covariance are inferred as zero.
``CONN`` instead merges by moving all source prototypes and splits by passing
global ``prototype_ids``; see :doc:`conn`.

The package stores sufficient statistics rather than the original dataset.
Consequently, the caller must ensure that a sample passed to ``remove`` really belongs to the supplied label.
Similarly, split statistics must describe a true subset of the retained cluster.
Invalid labels, inconsistent statistics, changed feature dimensions, and attempts to merge a label with itself raise an error.

Index metadata
--------------

Every implementation exposes an ``info`` class attribute describing its name,
range, optimization direction, and numerical backends:

.. doctest::

   >>> import cvi
   >>> cvi.CH.info
   CVIInfo(name='Calinski-Harabasz', name_short='CH', index_min=0.0, index_max=inf, optimality='max', backends=('numpy', 'numba', 'jax'))

Use ``optimality`` rather than assuming that a larger value is always better.
``backends`` lists implemented numerical backends, even when their optional
dependencies are not installed.

Use an instance's ``capabilities`` property for configuration-specific operation
support:

.. doctest::

   >>> cvi.CONN(model_type="KMeans").capabilities
   CVICapabilities(batch=True, incremental=False, merge=True, remove=False, split=True, mini_batch=False)

Capabilities describe the selected numerical and prototype backends. They do
not indicate whether the object has been initialized or whether an optional
dependency is installed.
``batch`` indicates one-shot initialization with ``get_cvi(samples, labels)``;
``mini_batch`` indicates cumulative updates with ``update_batch``, including
the sequential FuzzyART CONN wrapper.

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
