[![cvi-header](https://github.com/AP6YC/FileStorage/blob/main/cvi/header.png?raw=true)][docs-dev-url]

Batch and incremental cluster validity indices (CVIs) for hard partitions in Python.

| **Stable Docs** | **Dev Docs** | **Build Status** | **Coverage** |
|:---------------:|:------------:|:----------------:|:------------:|
| [![Stable][docs-stable-img]][docs-stable-url] | [![Dev][docs-dev-img]][docs-dev-url] | [![Build Status][ci-img]][ci-url] | [![Codecov][codecov-img]][codecov-url] |
| **Version** | **Issues** | **Downloads** | **Zenodo DOI** |
| [![version][version-img]][version-url] | [![issues][issues-img]][issues-url] | [![Downloads][downloads-img]][downloads-url] | [![DOI][zenodo-img]][zenodo-url] |

[downloads-img]: https://static.pepy.tech/badge/cvi
[downloads-url]: https://pepy.tech/project/cvi
[zenodo-img]: https://zenodo.org/badge/526280198.svg
[zenodo-url]: https://zenodo.org/badge/latestdoi/526280198
[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://AP6YC.github.io/cvi/main
[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://AP6YC.github.io/cvi/develop
[ci-img]: https://github.com/AP6YC/cvi/actions/workflows/Test.yml/badge.svg
[ci-url]: https://github.com/AP6YC/cvi/actions/workflows/Test.yml
[codecov-img]: https://codecov.io/gh/AP6YC/cvi/branch/main/graph/badge.svg
[codecov-url]: https://codecov.io/gh/AP6YC/cvi
[version-img]: https://img.shields.io/pypi/v/cvi.svg
[version-url]: https://pypi.org/project/cvi
[issues-img]: https://img.shields.io/github/issues/AP6YC/cvi?style=flat
[issues-url]: https://github.com/AP6YC/cvi/issues

You clustered your data. Without true labels, how do you know whether the result is any good?
A cluster validity index scores a partition using only the samples and the labels your algorithm assigned.
The score rates the partition's quality by that index's definition: some indices are better when larger and others when smaller, but moving a score toward its optimum always means a better partition in the sense that index measures.
Indices look for different things, such as tight clusters, wide gaps between them, or balanced sizes. See [Choosing an index](https://AP6YC.github.io/cvi/main/choosing.html) for what each one rewards.

`cvi` computes these scores for a whole dataset at once, or keeps them up to date one sample at a time as a stream is clustered, with the same result either way.

The animations below show CH, DB, and cSIL updating as samples arrive (solid lines) and landing on the batch value for the full dataset (dashed lines).
The top row streams synthetic Gaussian blobs and the bottom row the Iris dataset, first in class order and then shuffled.

<p align="center">
  <img src="https://media.githubusercontent.com/media/AP6YC/cvi/develop/docs/source/_static/readme/incremental-cvi-synthetic-class-order.gif" alt="Incremental CVIs on class-ordered synthetic Gaussian blobs" width="49%">
  <img src="https://media.githubusercontent.com/media/AP6YC/cvi/develop/docs/source/_static/readme/incremental-cvi-synthetic-shuffled.gif" alt="Incremental CVIs on shuffled synthetic Gaussian blobs" width="49%">
</p>
<p align="center">
  <img src="https://media.githubusercontent.com/media/AP6YC/cvi/develop/docs/source/_static/readme/incremental-cvi-iris-class-order.gif" alt="Incremental CVIs on the class-ordered Iris dataset" width="49%">
  <img src="https://media.githubusercontent.com/media/AP6YC/cvi/develop/docs/source/_static/readme/incremental-cvi-iris-shuffled.gif" alt="Incremental CVIs on the shuffled Iris dataset" width="49%">
</p>

## Installation

```console
python -m pip install cvi
```

Optional extras: `cvi[numba]` for compiled kernels, `cvi[jax]` for JAX, and `cvi[art]` for incremental `CONN`.

## Quickstart

Samples are rows and features are columns.

```python
import numpy as np
import cvi

samples = np.array([[0.0, 0.1], [0.2, 0.0], [2.8, 3.0], [3.1, 2.9]])
labels = np.array([0, 1, 2, 2])

# Score a whole dataset at once
value = cvi.CH().get_cvi(samples, labels)

# Or update the score as each sample arrives
index = cvi.CH()
for sample, label in zip(samples, labels):
    value = index.get_cvi(sample, int(label))
```

Besides batch evaluation, every index has an incremental version that keeps running statistics for each cluster instead of the data.
Because it holds state, an index can also `remove` samples, `move` them between clusters, and `merge` or `split` clusters without replaying the data, and `commit=False` scores any of these without applying it.
A batch call leaves the same state behind, so these work after a batch too.
Use a new object for each partition you evaluate.
See [Batch and incremental evaluation](https://AP6YC.github.io/cvi/main/batch_incremental.html) for when to use each.

## Indices

| Index | Name | Prefer | Range |
|---|---|---|---|
| `CH` | Calinski–Harabasz | Larger | `[0, ∞)` |
| `CONN` | Connectivity | Larger | `[0, 1]` |
| `cSIL` | Centroid-based Silhouette | Larger | `[-1, 1]` |
| `DB` | Davies–Bouldin | Smaller | `[0, ∞)` |
| `GD43` | Generalized Dunn 43 | Larger | `[0, ∞)` |
| `GD53` | Generalized Dunn 53 | Larger | `[0, ∞)` |
| `PS` | Partition Separation | Larger | `[0, 1]` |
| `rCIP` | Representative Cross Information Potential | Smaller | `[0, ∞)` |
| `WB` | Within/Between | Smaller | `[0, ∞)` |
| `XB` | Xie–Beni | Smaller | `[0, ∞)` |

## Documentation

The [documentation][docs-stable-url] covers:

- [Getting started](https://AP6YC.github.io/cvi/main/guide.html): input rules and index metadata
- [Batch and incremental evaluation](https://AP6YC.github.io/cvi/main/batch_incremental.html): how the two compare and when to use each
- [Updating a partition](https://AP6YC.github.io/cvi/main/operations.html): remove, merge, split, trial scoring, moves, and mini-batches
- [Choosing an index](https://AP6YC.github.io/cvi/main/choosing.html): how to read a score and what each index rewards
- [Backends](https://AP6YC.github.io/cvi/main/backends.html): Numba and JAX, with benchmarks
- [Using CONN](https://AP6YC.github.io/cvi/main/conn.html), which has its own setup requirements

## Acknowledgements

`cvi` is derived from [ClusterValidityIndices.jl](https://github.com/AP6YC/ClusterValidityIndices.jl), by the same authors:

- Sasha Petrenko <petrenkos@mst.edu>
- Nik Melton <nmmz76@mst.edu>

Related Python packages: [validclust](https://github.com/crew102/validclust) and [clusterval](https://github.com/Nuno09/clusterval).
