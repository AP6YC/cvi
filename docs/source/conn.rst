Using CONN
==========

``CONN`` evaluates connectivity between prototypes rather than relying only on centroid distances.
Its batch and incremental modes therefore require an explicit choice of prototype backend.

Batch mode
----------

The default backend is ``MiniBatchKMeans`` and supports batch input only:

.. code-block:: python

   import cvi

   index = cvi.CONN(
       model_type="MiniBatchKMeans",
       kmeans_k=8,
       kmeans_kwargs={"random_state": 0, "n_init": 10},
   )
   value = index.get_cvi(samples, labels)

``model_type="KMeans"`` selects ordinary scikit-learn KMeans.
For either KMeans backend, ``kmeans_k`` may be a positive integer applied to every input label or a dictionary keyed by the original integer labels.
Counts larger than a label's sample count are capped automatically.
Do not pass ``n_clusters`` in ``kmeans_kwargs``; configure it through ``kmeans_k``.

With the default ``normalize_batch=True``, each feature is min-max normalized over the full batch before prototypes are fitted.
Disable this only when the data are already on the intended scale.

Incremental mode
----------------

Incremental updates require the optional FuzzyART backend. Install its extra
before selecting it:

.. code-block:: console

   python -m pip install "cvi[art]"

Then construct a Fuzzy-backed index:

.. code-block:: python

   index = cvi.CONN(model_type="Fuzzy")

   for sample, label in stream:
       value = index.get_cvi(sample, int(label))

Incremental samples are not normalized by the class because future feature
bounds are unknown. They must normally already lie in ``[0, 1]``.
The default ``check_incremental_normalized=True`` validates that assumption; disabling the
check does not normalize the samples.

The FuzzyART parameters ``rho``, ``alpha``, ``beta``, and ``match_tracking``
control prototype formation and are passed to the underlying ART model. They
are optional and initialized only for ``model_type="Fuzzy"``. Conversely,
``kmeans_k`` and ``kmeans_kwargs`` are initialized only for a KMeans model.
Stream order and those parameters can therefore affect the learned prototypes
and the resulting criterion trajectory.

Chunk updates with FuzzyART
------------------------------

``update_batch`` is a convenience wrapper for the FuzzyART incremental path.
It calls the existing single-sample update for each row in order, including
the special first-two-sample initialization, prototype learning, and scoring.
It returns only the final score. Changing chunk boundaries preserves the
same learning trajectory as individual updates in the same order:

.. doctest:: conn_fuzzy
   :skipif: __import__("importlib.util").util.find_spec("artlib") is None

   >>> import numpy as np
   >>> import cvi
   >>> samples = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
   >>> labels = np.array([10, 10, 20, 20])
   >>> index = cvi.CONN(model_type="Fuzzy")
   >>> index.capabilities.mini_batch
   True
   >>> first = index.update_batch(samples[:1], labels[:1])
   >>> np.isnan(first).item()
   True
   >>> index.update_batch(samples[1:], labels[1:])
   0.125

Chunks may initialize an empty index or continue after batch initialization,
individual samples, other chunks, or prototype merge/split operations. New
integer labels are accepted. Empty chunks leave the state unchanged; an
undefined score returns NaN without warning.

As with incremental input, chunks must already use the ART input scale,
normally ``[0, 1]``. ``normalize_batch=True`` applies only to one-shot
``get_cvi`` batch initialization, never to ``update_batch``. When continuing
from a normalized batch, apply the same fixed scaling to subsequent samples.

The wrapper validates the entire chunk and stages updates on a copy of the
complete state, including ART, so a failed call leaves the index unchanged.
Copying adds time and memory per chunk; this API does not aggregate learning
or promise a speedup. KMeans and MiniBatchKMeans keep
``capabilities.mini_batch=False`` and reject ``update_batch``.

Moving prototypes
-----------------

All CONN prototype models can merge clusters or split off whole prototypes
after initialization. Merge retains the target label and moves every source
prototype into it. Split retains the original label for the remaining
prototypes and assigns the selected prototypes to a new label:

.. code-block:: python

   value = index.merge(target_label=20, source_label=10)
   value = index.split(
       retained_label=20,
       new_label=30,
       prototype_ids=[1, 3],
   )

Use ``index.get_prototype_ids(label)`` to find the global prototype IDs owned
by a cluster. Each selected ID must belong to the retained cluster, and at
least one prototype must remain there. For KMeans models, both resulting
clusters must contain assigned samples; an unused center alone cannot form
a cluster. Samples assigned to moved prototypes follow their new label.
Prototype weights or centers and connectivity counts are preserved, while
CONN's cluster-level score is recalculated. KMeans models are not refitted.

Limitations
-----------

``CONN`` does not implement :meth:`cvi.CVI.remove`.
The KMeans and MiniBatchKMeans models reject incremental samples and chunk
updates. Use a new object when changing backend or evaluating another
independent partition.
