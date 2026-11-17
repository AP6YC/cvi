Updating a Partition
====================

An initialized index can change its partition without replaying the data. Each
operation below updates the object in place and returns the new criterion value.
Check ``index.capabilities`` for which operations a given configuration
supports.

Remove, merge, and split
------------------------

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

``merge`` keeps the target label and deletes the source label. ``split`` keeps
the existing label for the residual cluster and gives the split-off subset an
unused new label without changing the total sample count. Removing a cluster's
last sample deletes that label; removing the last sample in the index returns
the object to its initial empty state.

Every split needs the subset's sample count and centroid. For non-singletons,
compactness-based indices also need the centered sum of squared distances, and
``rCIP`` needs the unregularized unbiased sample covariance; ``PS`` needs
neither. Singleton compactness and covariance are inferred as zero. ``CONN``
merges by moving all source prototypes and splits by passing global
``prototype_ids``, and it does not support ``remove``; see :doc:`conn`.

The package stores sufficient statistics, not the original data. The caller
must make sure that a sample passed to ``remove`` belongs to the supplied
label and that split statistics describe a true subset of the retained
cluster. Unknown labels, inconsistent statistics, changed feature dimensions,
and merging a label with itself raise an error.

Trial scoring
-------------

``get_cvi``, ``remove``, ``merge``, ``split``, and ``move`` take a keyword-only
``commit`` argument, ``True`` by default. With ``commit=False`` the call returns
the score the operation would produce and leaves the object unchanged. A label
that has not been seen yet means "create a new cluster". Scores are NaN while
the index is undefined.

.. code-block:: python

   # Score one candidate without changing the index.
   score = index.get_cvi(sample, label, commit=False)

   # Score several candidate labels at once: one score per label.
   scores = index.get_cvi(sample, np.array([10, 20, 30, 99]), commit=False)

An array of candidate labels is accepted only with ``commit=False`` and a
one-dimensional sample, so the batch form of ``get_cvi`` stays unambiguous.
Committing always needs a scalar label. Scoring candidates together shares
work: CH and WB score ``K`` candidates in ``O(K·d)``. Indices built on pairwise
centroid statistics (DB, XB, GD43, GD53, PS, cSIL, rCIP) need each candidate's
new row, so they cost ``O(K²·d)``, but they still avoid rebuilding the full
pairwise matrix for each candidate.

**Committing after a trial does not recompute.** Each object keeps the state
from its most recent trial. If the next committing call is the same operation
with the same arguments, it applies that state directly. For ``get_cvi`` and
``move``, this means the same sample (and old label), with the committed label
among the scored candidates. Any other change discards the stored state, which
is private and is not pickled. A typical accept-first-improvement loop is one
vectorized trial plus one cheap commit:

.. code-block:: python

   scores = index.get_cvi(sample, candidates, commit=False)
   for label, score in zip(candidates, scores):
       if index.is_better(score, index.criterion_value):
           index.get_cvi(sample, label)  # reuses the trial's state
           break

``is_better(new, old, tol=0.0)`` compares two scores in the index's preferred
direction (``info.optimality``). Ties are accepted, and a NaN on either side
counts as acceptance.

Moves
-----

.. code-block:: python

   scores = index.move(sample, old_label, np.array([10, 20, 99]), commit=False)
   value = index.move(sample, old_label, 20)

A committed move leaves exactly the state of ``remove(sample, old_label)``
followed by ``get_cvi(sample, new_label)``. It is atomic: when it empties a
singleton cluster, no intermediate state is visible, and a failed move leaves
the object unchanged. Moving a sample to its own label is a no-op that returns
the current score.

``CONN`` and JAX configurations report ``trial=False`` and ``move=False`` and
raise ``NotImplementedError`` for ``commit=False`` and ``move``.

Mini-batch updates
------------------

``update_batch`` appends a labeled chunk and scores the whole accumulated
partition:

.. code-block:: python

   index = cvi.CH()
   assert index.capabilities.mini_batch
   for start in range(0, len(samples), 1024):
       value = index.update_batch(samples[start:start + 1024],
                                  labels[start:start + 1024])

``CH``, ``WB``, ``DB``, ``XB``, ``GD43``, ``GD53``, ``PS``, and ``cSIL`` support
this with NumPy and Numba, and ``rCIP`` with NumPy only.
``CONN(model_type="Fuzzy")`` provides a sequential convenience wrapper; its
KMeans backends and all JAX configurations raise ``NotImplementedError``.

Each chunk contains only new observations. The method can start a fresh object
or continue after a batch, sample updates, other chunks, or structural
operations. New integer labels are appended in first-seen order and the feature
count must stay fixed. An empty ``(0, n_features)`` chunk is a no-op. A chunk
with only one label is allowed; an undefined cumulative score returns NaN
without a warning. Inputs must be finite real numbers with one integer label per
row. The update is staged before committing, so a failed call leaves the object
unchanged.

Except for FuzzyART CONN, chunks are summarized in float64 using centered
moments and merged into the existing statistics. Reduction order differs from
batch and sample updates, so results are not bitwise identical. Legacy float32
batch initialization keeps its original precision, which later chunks cannot
recover. ``cSIL`` keeps centered compactness and residual sums through every
operation to avoid subtracting large raw moments; older pickled objects recover
what centered state they can from their stored dissimilarities. ``rCIP`` combines
unregularized covariances through centered scatter and regularizes the merged
covariance once.

FuzzyART CONN runs its incremental update for every row in order and returns
only the final score. Chunks must already be on the ART input scale (normally
``[0, 1]``); ``normalize_batch`` does not apply. The ART state is copied per
chunk to preserve it on failure, so this wrapper is a convenience, not a
speedup.

No per-sample scores are produced; use sample updates or JAX's ``update_many``
when you need them. Larger chunks generally improve throughput at the cost of
memory and less frequent scores. Pairwise indices still store a matrix that
grows with the square of the cluster count.
