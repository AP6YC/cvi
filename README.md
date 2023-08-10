# cvi

A Python package implementing both batch and incremental cluster validity indices (CVIs).

| **Stable Docs**  | **Dev Docs** | **Build Status** | **Coverage** |
|:----------------:|:------------:|:----------------:|:------------:|
| [![Stable][docs-stable-img]][docs-stable-url] | [![Dev][docs-dev-img]][docs-dev-url]| [![Build Status][ci-img]][ci-url] | [![Codecov][codecov-img]][codecov-url] |
| **Version** | **Issues** | **Downloads** | **Zenodo DOI** |
| [![version][version-img]][version-url] | [![issues][issues-img]][issues-url] | [![Downloads][downloads-img]][downloads-url] |  [![DOI][zenodo-img]][zenodo-url] |

[downloads-img]: https://static.pepy.tech/badge/cvi
[downloads-url]: https://pepy.tech/project/cvi

[zenodo-img]: https://zenodo.org/badge/526280198.svg
[zenodo-url]: https://zenodo.org/badge/latestdoi/526280198

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

## Table of Contents

- [cvi](#cvi)
  - [Table of Contents](#table-of-contents)
  - [Cluster Validity Indices](#cluster-validity-indices)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Quickstart](#quickstart)
    - [Detailed Usage](#detailed-usage)
  - [Implemented CVIs](#implemented-cvis)
  - [History](#history)
  - [Acknowledgements](#acknowledgements)
    - [Derivation](#derivation)
    - [Authors](#authors)
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
Clustering well in this context means correctly partitioning (i.e., separating) the data rather than prescribing too many different clusters (over partitioning) or too few (under partitioning).
Every CVI itself also behaves differently in terms of the range and scale of their numbers.
**Furthermore, each CVI has an original batch implementation and incremental implementation that are equivalent**.

The `cvi` Python package contains a variety of these batch and incremental CVIs.

## Installation

The `cvi` package is listed on PyPI, so you may install the latest version with

```python
pip install cvi
```

You can also specify a version to install in the usual way with

```python
pip install cvi==v0.4.0
```

Alternatively, you can manually install a release from the [releases page](https://github.com/AP6YC/cvi/releases) on GitHub.

## Usage

### Quickstart

Create a CVI object and compute the criterion value in batch with `get_cvi`:

```python
# Import the library
import cvi
# Create a Calinski-Harabasz (CH) CVI object
my_cvi = cvi.CH()
# Load some data from some clustering algorithm
samples, labels = load_some_clustering_data()
# Compute the final criterion value in batch
criterion_value = my_cvi.get_cvi(samples, labels)
```

or do it incrementally, also with `get_cvi`:

```python
# Datasets are numpy arrays
import numpy as np
# Create a container for criterion values
n_samples = len(labels)
criterion_values = np.zeros(n_samples)
# Iterate over the data
for ix in range(n_samples):
    criterion_values = my_cvi.get_cvi(samples[ix, :], labels[ix])
```

### Detailed Usage

The `cvi` package contains a set of implemented CVIs with batch and incremental update methods.
Each CVI is a standalone stateful object inheriting from a base class `CVI`, and all `CVI` functions are object methods, such as those that update parameters and return the criterion value.

Instantiate a CVI of you choice with the default constructor:

```python
# Import the package
import cvi
# Import numpy for some data handling
import numpy as np

# Instantiate a Calinski-Harabasz (CH) CVI object
my_cvi = cvi.CH()
```

CVIs are instantiated with their acronyms, with a list of all implemented CVIS being found in the [Implemented CVIs](#implemented-cvis) section.

A batch of data is assumed to be a numpy array of samples and a numpy vector of integer labels.

```python
# Load some data
samples, labels = my_clustering_alg(some_data)
```

> **NOTE**:
>
> The `cvi` package assumes the Numpy **row-major** convention where rows are individual samples and columns are features.
> A batch dataset is then `[n_samples, n_features]` large, and their corresponding labels are '[n_samples]` large.

You may compute the final criterion value with a batch update all at once with `CVI.get_cvi`

```python
# Get the final criterion value in batch mode
criterion_value = my_cvi.get_cvi(samples, labels)
```

or you may get them incrementally with the same method, where you pass instead just a single numpy vector of features and a single integer label.
The incremental methods are used automatically based upon the dimensions of the data that is passed.

```python
# Create a container for the criterion value after each sample
n_samples = len(labels)
criterion_values = np.zeros(n_samples)

# Iterate across the data and store the criterion value over time
for ix in range(n_samples):
    sample = samples[ix, :]
    label = labels[ix]
    criterion_values[ix] = my_cvi.get_cvi(sample, label)
```

> **NOTE**:
>
> Currently only using _either_ batch _or_ incremental methods is supported; switching from batch to incremental updates with the same is not yet implemented.

## Implemented CVIs

The following CVIs have been implemented as of the latest version of `cvi`:

- **CH**: Calinski-Harabasz
- **cSIL**: Centroid-based Silhouette
- **DB**: Davies-Bouldin
- **GD43**: Generalized Dunn's Index 43.
- **GD53**: Generalized Dunn's Index 53.
- **PS**: Partition Separation.
- **rCIP**: (Renyi's) representative Cross Information Potential.
- **WB**: WB-index.
- **XB**: Xie-Beni.

## History

- 8/18/2022: Initialize project.
- 9/8/2022: First release on PyPi and initiate GitFlow.

## Acknowledgements

### Derivation

The incremental and batch CVI implementations in this package are largely derived from the following Julia language implementations:

- [ClusterValidityIndices.jl](https://github.com/AP6YC/ClusterValidityIndices.jl)

### Authors

The principal authors of the `cvi` pacakge are:

- Sasha Petrenko <petrenkos@mst.edu>
- Nik Melton <nmmz76@mst.edu>

### Related Projects

If this package is missing something that you need, feel free to check out some related Python cluster validity packages:

- [validclust](https://github.com/crew102/validclust)
- [clusterval](https://github.com/Nuno09/clusterval)
