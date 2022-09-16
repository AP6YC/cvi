# cvi

A Python package implementing both batch and incremental cluster validity indices (CVIs).

| **Stable Docs**  | **Dev Docs** | **Build Status** | **Coverage** |
|:----------------:|:------------:|:----------------:|:------------:|
| [![Stable][docs-stable-img]][docs-stable-url] | [![Dev][docs-dev-img]][docs-dev-url]| [![Build Status][ci-img]][ci-url] | [![Codecov][codecov-img]][codecov-url] |
| **Version** | **Issues** | **Commits** | **Commits Since Release**
| [![version][version-img]][version-url] | [![issues][issues-img]][issues-url] | [![commits][commits-img]][commits-url] | [![compare][compare-img]][compare-url] |

<!-- | **Zenodo DOI** |
| :------------: |
| [![DOI][zenodo-img]][zenodo-url] | -->

[docs-stable-img]: https://readthedocs.org/projects/cluster-validity-indices/badge/?version=latest
[docs-stable-url]: https://cluster-validity-indices.readthedocs.io/en/latest/?badge=latest

[docs-dev-img]: https://readthedocs.org/projects/cluster-validity-indices/badge/?version=develop
[docs-dev-url]: https://cluster-validity-indices.readthedocs.io/en/develop/?badge=develop

[ci-img]: https://github.com/AP6YC/cvi/actions/workflows/Test.yml/badge.svg
[ci-url]: https://github.com/AP6YC/cvi/actions/workflows/Test.yml

[codecov-img]: https://codecov.io/gh/AP6YC/cvi/branch/main/graph/badge.svg
[codecov-url]: https://codecov.io/gh/AP6YC/cvi

[version-img]: https://img.shields.io/pypi/v/cvi.svg
[version-url]: https://pypi.org/project/cvi

[issues-img]: https://img.shields.io/github/issues/AP6YC/cvi?style=flat
[issues-url]: https://github.com/AP6YC/cvi/issues

[commits-img]: https://img.shields.io/github/commit-activity/m/AP6YC/cvi?style=flat
[commits-url]: https://github.com/AP6YC/cvi/commits/main

[compare-img]: https://img.shields.io/github/commits-since/AP6YC/cvi/latest/develop
[compare-url]: https://github.com/AP6YC/cvi/compare/v0.1.0-alpha.4...develop

## Table of Contents

- [cvi](#cvi)
  - [Table of Contents](#table-of-contents)
  - [Cluster Validity Indices](#cluster-validity-indices)
  - [Installation](#installation)
  - [Usage](#usage)
  - [History](#history)
  - [Acknowledgements](#acknowledgements)
  - [Related Projects](#related-projects)

## Cluster Validity Indices

Say you have a clustering algorithm that clusters a set of samples containing features of some kind and some dimensionality.
Great!
That was a lot of work, and you should feel accomplished.
But how do you know that the algorithm performed _well_?
By definition, you wouldn't have the _true_ label belonging to each sample (if one could even exist in your context), just the label prescribed by your clustering algorithm.

**Enter Cluster Validity Indices (CVIs)**.

CVIs are metrics of cluster partitioning when true cluster labels are unavailable.
Each operates on only the information available (i.e., the provided samples of features and the labels prescribed by the clustering algorithm) and produces a _metric_, a number that goes up or down according to how well the CVI believes the clustering algorithm appears to, well, _cluster_.
Clustering well in this context means correctly partitioning (i.e., separating) the data rather than prescribing too many different clusters (over partitioning) or too few (under partitioning)
Every CVI itself also behaves differently in terms of the range and scale of their numbers.
**Furthermore, each CVI has an original batch implementation and incremental implementation that are equivalent**.

The `cvi` Python package contains a variety of these batch and incremental CVIs.

## Installation

The `cvi` package is listed on PyPi, so you may install the latest version with

```python
pip install cvi
```

You can also specify a version to install in the usual way with

```python
pip install cvi==v0.1.0-alpha.2
```

Alternatively, you can manually install a release from the [releases page](https://github.com/AP6YC/cvi/releases) on GitHub.

## Usage

TODO

## History

- 8/18/2022: Initialize project.
- 9/8/2022: First release on PyPi and initiate GitFlow.

## Acknowledgements

The incremental and batch CVI implementations in this package are largely derived from the following Julia language implementations:

- https://github.com/AP6YC/ClusterValidityIndices.jl

The principal authors of the `cvi` pacakge are:

- Sasha Petrenko <sap625@mst.edu>
- Nik Melton <nmmz76@mst.edu>

## Related Projects

If this package is missing something that you need, feel free to check out some related Python cluster validity packages:

- [validclust](https://github.com/crew102/validclust)
- [clusterval](https://github.com/Nuno09/clusterval)
