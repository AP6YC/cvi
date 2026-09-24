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

Moving prototypes
-----------------

FuzzyART-backed CONN can merge clusters or split off whole prototypes after
batch or incremental initialization. Merge retains the target label and moves
every source prototype into it. Split retains the original label for the
remaining prototypes and assigns the selected prototypes to a new label:

.. code-block:: python

   value = index.merge(target_label=20, source_label=10)
   value = index.split(
       retained_label=20,
       new_label=30,
       prototype_ids=[1, 3],
   )

Use ``index.get_prototype_ids(label)`` to find the global prototype IDs owned
by a cluster. Each selected ID must belong to the retained cluster, and at
least one prototype must remain there. Samples recorded under moved prototypes
follow their new label.
Prototype weights and connectivity counts are preserved, while CONN's
cluster-level score is recalculated.

Limitations
-----------

``CONN`` does not implement :meth:`cvi.CVI.remove`.
The KMeans and MiniBatchKMeans models do not support merge or split and reject
incremental samples. Use a new object when changing backend or evaluating
another independent partition.
