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
     - Remove/merge
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

