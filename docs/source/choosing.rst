Choosing an Index
=================

Each cluster validity index emphasizes different properties of a partition.
Values from different index families are not on a common scale, so compare
candidate partitions with the same index rather than comparing one index's
number directly with another's.

The following operation support describes the default NumPy backend. Every
listed index supports batch evaluation.

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
     - FuzzyART Backend Only
     - No
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

.. include:: _backend_coverage.rstinc

See :doc:`backends` for installation, accelerated kernels, precision rules,
and compilation costs. Backend availability does not guarantee a speedup.

``CH``, ``DB``, ``GD43``, ``GD53``, ``WB``, and ``XB`` summarize variants of
within-cluster compactness and between-cluster separation. ``cSIL`` offers a
bounded, centroid-based silhouette measure. ``PS`` measures partition
separation, while ``rCIP`` uses distributional information. ``CONN`` is the
specialized choice when connectivity between learned prototypes is important;
see :doc:`conn` before using it.

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
