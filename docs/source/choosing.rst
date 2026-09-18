Choosing an Index
=================

Each cluster validity index emphasizes different properties of a partition.
Values from different index families are not on a common scale, so compare
candidate partitions with the same index rather than comparing one index's
number directly with another's.

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
     - Fuzzy backend
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

``CH``, ``DB``, ``GD43``, ``GD53``, ``WB``, and ``XB`` summarize variants of
within-cluster compactness and between-cluster separation. ``cSIL`` offers a
bounded, centroid-based silhouette measure. ``PS`` measures partition
separation, while ``rCIP`` uses distributional information. ``CONN`` is the
specialized choice when connectivity between learned prototypes is important;
see :doc:`conn` before using it.

The value ``0.0`` is used when an index is not yet defined, including many
one-cluster states. When monitoring a stream, consider the trajectory only
after enough clusters and samples have been observed.

Every index provides a read-only ``is_defined`` property. It is ``False``
before evaluation and is updated after batch or incremental evaluation and
after remove, merge, or split operations. Undefined states return ``0.0``.
An undefined batch evaluation also emits a ``RuntimeWarning``; incremental
startup and structural operations remain silent. A computed score of ``0.0``
can still have ``is_defined == True``; the score alone does not identify an
undefined state.

The conditions for a defined value are:

.. list-table:: Definition conditions
   :header-rows: 1

   * - Index
     - ``is_defined`` is ``True`` when
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

For example, check the property before consuming a streaming score:

.. code-block:: python

   import cvi

   index = cvi.CH()
   for sample, label in zip(samples, labels):
       value = index.get_cvi(sample, label)
       if index.is_defined:
           print(value)
