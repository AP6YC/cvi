Choosing an Index
=================

Reading a score
---------------

A CVI turns a partition into a single number that rates its quality by that
index's own definition of a good partition. Some indices are maximized and
some are minimized: a larger Calinski–Harabasz value is better, while a
smaller Davies–Bouldin value is better. Each object's ``info.optimality``
says which (``"max"`` or ``"min"``), and ``index.is_better(new, old)``
compares two scores in the right direction. Pushing a score toward its optimum,
for example by choosing between candidate partitions, gives a better partition
in the sense that index measures.

Because each index defines quality differently, two indices can disagree about
which partition is better, and their values are not on a common scale. Compare
partitions with the same index, and choose the index whose idea of quality fits
your data.

What each index measures
------------------------

Most indices balance **compactness** (how tightly each cluster's samples sit
around its centroid) against **separation** (how far apart clusters are). They
differ in how they summarize each one, which changes what they reward.

``CH`` (Calinski–Harabasz), maximized
   Ratio of between-cluster to within-cluster sum of squares, scaled by the
   sample and cluster counts. It looks at the partition as a whole and favors
   compact, well-separated, roughly spherical clusters.

``WB`` (Within/Between), minimized
   Within-cluster over between-cluster sum of squares, multiplied by the number
   of clusters. It uses the same statistics as CH, but the multiplier penalizes
   adding clusters more directly.

``XB`` (Xie–Beni), minimized
   Total within-cluster sum of squares divided by the smallest distance between
   any two centroids. A single pair of nearby clusters makes the score worse,
   so it penalizes partitions that split one cluster in two.

``DB`` (Davies–Bouldin), minimized
   For each cluster, finds the other cluster it is most easily confused with,
   using the ratio of their combined spread to their centroid distance, then
   averages those worst cases. It rewards every cluster being distinct from its
   nearest neighbor, not just the partition on average.

``GD43`` and ``GD53`` (Generalized Dunn), maximized
   Smallest separation between any two clusters divided by the spread of the
   widest cluster. Both are governed by the worst cases. GD43 measures
   separation as centroid distance; GD53 measures it from the pooled spread of
   each pair of clusters around their own centroids.

``PS`` (Partition Separation), maximized
   Sums a per-cluster term that rises with the cluster's size relative to the
   largest cluster and with its distance to the nearest other centroid. It
   favors balanced cluster sizes as well as separation.

``cSIL`` (Centroid-based Silhouette), maximized
   A silhouette computed with centroids instead of every pair of samples,
   comparing each cluster's spread around its own centroid with how close the
   nearest other cluster comes to it. Its bounded range makes values easy to
   read on their own.

``rCIP`` (Representative Cross Information Potential), minimized
   Adds up the overlap between Gaussian models of each pair of clusters, built
   from their centroids and covariances. Because it uses covariances, it
   accounts for cluster shape and orientation, not only distance.

``CONN`` (Connectivity), maximized
   Learns prototypes of the data and measures how strongly neighboring
   prototypes connect within clusters versus across them. Unlike the others, it
   can recognize clusters that are not convex. It has its own setup
   requirements; see :doc:`conn`.

Summary
-------

Every index supports batch evaluation. The table below describes the default
NumPy backend; see :doc:`backends` for Numba and JAX coverage and
``index.capabilities`` for the exact operations a configuration supports.

.. list-table:: Implemented indices
   :header-rows: 1
   :widths: 12 25 13 13 14 23

   * - Index
     - Name
     - Prefer
     - Range
     - Incremental
     - Remove/merge/split
   * - ``CH``
     - Calinski–Harabasz
     - Larger
     - ``[0, inf)``
     - Yes
     - Yes
   * - ``CONN``
     - Connectivity
     - Larger
     - ``[0, 1]``
     - FuzzyART backend only
     - Merge/split only
   * - ``cSIL``
     - Centroid-based Silhouette
     - Larger
     - ``[-1, 1]``
     - Yes
     - Yes
   * - ``DB``
     - Davies–Bouldin
     - Smaller
     - ``[0, inf)``
     - Yes
     - Yes
   * - ``GD43``
     - Generalized Dunn 43
     - Larger
     - ``[0, inf)``
     - Yes
     - Yes
   * - ``GD53``
     - Generalized Dunn 53
     - Larger
     - ``[0, inf)``
     - Yes
     - Yes
   * - ``PS``
     - Partition Separation
     - Larger
     - ``[0, 1]``
     - Yes
     - Yes
   * - ``rCIP``
     - Representative Cross Information Potential
     - Smaller
     - ``[0, inf)``
     - Yes
     - Yes
   * - ``WB``
     - Within/Between
     - Smaller
     - ``[0, inf)``
     - Yes
     - Yes
   * - ``XB``
     - Xie–Beni
     - Smaller
     - ``[0, inf)``
     - Yes
     - Yes

When is a score defined?
------------------------

The value ``numpy.nan`` is used when an index is not yet defined, including
many one-cluster states. When monitoring a stream, consider the trajectory
only after enough clusters and samples have been observed. An undefined batch
evaluation also emits a ``RuntimeWarning``; incremental startup and structural
operations remain silent. Use ``numpy.isnan`` to test whether a result is
undefined. A computed score of ``0.0`` remains a valid result.

The conditions for a defined value are:

.. list-table:: Definition conditions
   :header-rows: 1

   * - Index
     - The score is defined when
   * - ``CH``
     - There are at least two clusters and within-cluster sum of squares is positive.
   * - ``CONN``
     - At least two prototypes or ART categories have been learned.
   * - ``cSIL``
     - There are at least two clusters. A local term with equal zero compactness and separation contributes ``0.0``.
   * - ``DB``
     - There are at least two clusters and every pair of centroids has positive separation.
   * - ``GD43`` and ``GD53``
     - There are at least two clusters and at least one cluster has positive dispersion.
   * - ``PS``
     - There are at least two clusters and the cluster centroids have positive dispersion.
   * - ``rCIP``
     - There are at least two clusters.
   * - ``WB``
     - There are at least two clusters and between-cluster sum of squares is positive.
   * - ``XB``
     - There are at least two clusters and minimum centroid separation is positive.

The checks use exact zero comparisons. No small value is added to a
denominator, so a very small nonzero denominator remains part of the metric's
result.

For example, exclude undefined values when consuming a streaming score:

.. code-block:: python

   import cvi
   import numpy as np

   index = cvi.CH()
   for sample, label in zip(samples, labels):
       value = index.get_cvi(sample, label)
       if not np.isnan(value):
           print(value)
