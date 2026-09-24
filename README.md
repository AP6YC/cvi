[![cvi-header](https://github.com/AP6YC/FileStorage/blob/main/cvi/header.png?raw=true)][docs-dev-url]

A Python package implementing batch and incremental cluster validity indices (CVIs) for hard partitions.

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

Cluster validity indices measure properties such as compactness, separation, and connectivity when ground-truth labels are unavailable.
This package uses a shared, stateful interface for evaluating a complete labeled partition or tracking its criterion value as samples arrive.

Please see the [documentation][docs-stable-url] for detailed usage.

## Table of Contents

- [Table of Contents](#table-of-contents)
- [What Are Cluster Validity Indices?](#what-are-cluster-validity-indices)
- [Incremental CVIs in Action](#incremental-cvis-in-action)
  - [Synthetic Gaussian Blobs](#synthetic-gaussian-blobs)
  - [Iris Dataset](#iris-dataset)
- [Installation](#installation)
- [Quickstart](#quickstart)
  - [Updating an Existing Partition](#updating-an-existing-partition)
  - [Implemented Indices](#implemented-indices)
- [Optimizations](#optimizations)
  - [Optional FuzzyART CONN Backend](#optional-fuzzyart-conn-backend)
  - [Optional Numba Acceleration](#optional-numba-acceleration)
  - [Optional JAX Batch Backend](#optional-jax-batch-backend)
  - [Fixed-capacity JAX Streaming](#fixed-capacity-jax-streaming)
- [Acknowledgements](#acknowledgements)
  - [Derivation](#derivation)
  - [Authors](#authors)
  - [Related Projects](#related-projects)
  - [Assets](#assets)
    - [Fonts](#fonts)
    - [Icons](#icons)

## What Are Cluster Validity Indices?

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

## Incremental CVIs in Action

These animations track three familiar CVIs as labeled samples arrive: Calinski–Harabasz (CH), Davies–Bouldin (DB), and centroid-based Silhouette (cSIL). Faded points have not arrived yet; opaque points are already part of the incremental partition. The solid traces show each incremental value, while the dashed lines show the value obtained by evaluating the complete dataset as one batch.

### Synthetic Gaussian Blobs

The same three well-separated clusters are streamed first in class order and then in a fixed random order.

<p align="center">
  <img src="https://media.githubusercontent.com/media/AP6YC/cvi/develop/docs/source/_static/readme/incremental-cvi-synthetic-class-order.gif" alt="Incremental CVIs on class-ordered synthetic Gaussian blobs" width="49%">
  <img src="https://media.githubusercontent.com/media/AP6YC/cvi/develop/docs/source/_static/readme/incremental-cvi-synthetic-shuffled.gif" alt="Incremental CVIs on shuffled synthetic Gaussian blobs" width="49%">
</p>

### Iris Dataset

The CVIs use all four standardized Iris features. PCA is used only to make the samples visible in the two-dimensional scatter plot.

<p align="center">
  <img src="https://media.githubusercontent.com/media/AP6YC/cvi/develop/docs/source/_static/readme/incremental-cvi-iris-class-order.gif" alt="Incremental CVIs on the class-ordered Iris dataset" width="49%">
  <img src="https://media.githubusercontent.com/media/AP6YC/cvi/develop/docs/source/_static/readme/incremental-cvi-iris-shuffled.gif" alt="Incremental CVIs on the shuffled Iris dataset" width="49%">
</p>

## Installation

The `cvi` package is listed on PyPI, so you may install the latest version with

```console
python -m pip install cvi
```

You can also specify a version to install in the usual way with

```console
pip install cvi==0.7.2
```

Alternatively, you can manually install a release from any of the builds on the [releases page](https://github.com/AP6YC/cvi/releases) on GitHub.

## Quickstart

```python
import numpy as np
import cvi

samples = np.array([
    [0.0, 0.1],
    [0.2, 0.0],
    [2.8, 3.0],
    [3.1, 2.9],
])
labels = np.array([0, 1, 2, 2])

# Batch evaluation
batch_index = cvi.CH()
batch_value = batch_index.get_cvi(samples, labels)

# Incremental evaluation
incremental_index = cvi.CH()
values = np.empty(len(labels))
for i, (sample, label) in enumerate(zip(samples, labels)):
    values[i] = incremental_index.get_cvi(sample, int(label))
```

CVI objects accumulate state.
Use a fresh object for each independent dataset or partition.
A batch call may be followed by incremental samples, but the same object cannot be initialized with a second batch.

> [!NOTE] NOTE
> The `cvi` package assumes the Numpy **row-major** convention where rows are individual samples and columns are features.
> A batch dataset is then `[n_samples, n_features]` large, and their corresponding labels are `[n_samples]` large.

Users can also query the `.info` property of the CVI objects to obtain scaling, naming, supported operations, and implemented numerical backends.

```
>>> print(my_cvi.info)
CVIInfo(name='Calinski-Harabasz', name_short='CH', index_min=0.0, index_max=inf, optimality='max', batch=True, incremental=True, merge=True, remove=True, split=True, backends=('numpy', 'numba', 'jax'))
```

The `batch`, `incremental`, `merge`, `remove`, and `split` flags describe support
in at least one configuration of the index. `backends` includes optional backends
regardless of whether their dependencies are installed. JAX incremental updates
require `capacity`; JAX does not support merge, remove, or split. CONN incremental
updates require an ART model.

### Updating an Existing Partition

Except for `CONN`, initialized indices support adding samples, removing samples, and merging clusters without replaying the full dataset via `remove` and `merge`:

```python
value = index.get_cvi(new_sample, new_label)
value = index.remove(existing_sample, existing_label)
value = index.merge(target_label=20, source_label=10)
```

Both methods update the object in place and return its new criterion value.
Removing the final sample of a cluster deletes that cluster, while `merge` retains `target_label` and deletes `source_label`.
The caller is responsible for ensuring that a removed sample belongs to the supplied label.

For input rules, index-selection guidance, references, legacy API information, and the complete API, see the [documentation][docs-stable-url].

[conn-guide]: https://AP6YC.github.io/cvi/main/conn.html

### Implemented Indices

The operation support below describes the default NumPy backend. Optional
backend coverage is listed separately.

| Index | Prefer | Range | Batch | Incremental | Remove/merge/split |
|---|---|---|---|---|---|
| `CH` | Larger | `[0, ∞)` | Yes | Yes | Yes |
| `CONN` | Larger | `[0, 1]` | Yes | FuzzyART backend only | No |
| `cSIL` | Larger | `[-1, 1]` | Yes | Yes | Yes |
| `DB` | Smaller | `[0, ∞)` | Yes | Yes | Yes |
| `GD43` | Larger | `[0, ∞)` | Yes | Yes | Yes |
| `GD53` | Larger | `[0, ∞)` | Yes | Yes | Yes |
| `PS` | Larger | `[0, 1]` | Yes | Yes | Yes |
| `rCIP` | Smaller | `[0, ∞)` | Yes | Yes | Yes |
| `WB` | Smaller | `[0, ∞)` | Yes | Yes | Yes |
| `XB` | Smaller | `[0, ∞)` | Yes | Yes | Yes |

**Optional optimization coverage**

|Index | Numba batch | Numba sample updates | Numba remove/merge | JAX batch | JAX streaming |
|---|---|---|---|---|---|
| `CH` | Yes | Unchanged | Unchanged | Yes | Capacity |
| `CONN` | Unavailable | Unavailable | Unavailable | Unavailable | Unavailable |
| `cSIL` | Yes | Unchanged | Unchanged | Unavailable | Unavailable |
| `DB` | Yes | Distances | Distances | Unavailable | Unavailable |
| `GD43` | Yes | Distances | Distances | Unavailable | Unavailable |
| `GD53` | Yes | Unchanged | Unchanged | Unavailable | Unavailable |
| `PS` | Yes | Distances | Distances | Unavailable | Unavailable |
| `rCIP` | Unavailable | Unavailable | Unavailable | Unavailable | Unavailable |
| `WB` | Yes | Unchanged | Unchanged | Yes | Capacity |
| `XB` | Yes | Distances | Distances | Yes | Capacity |

**Yes** means compiled batch kernels are available, not that every operation is
compiled. **Distances** means centroid-distance kernels are compiled; other
update logic remains in Python/NumPy. **Unchanged** means the operation is
supported with `backend="numba"` but uses its existing NumPy implementation.
**Unavailable** means that index rejects the selected numerical backend.

**Capacity** means JAX sample updates and `update_many` chunks require a positive
`capacity` at construction. CH, WB, and XB also expose functional JAX batch and
streaming APIs. All JAX modes require x64; JAX remove/merge is unsupported.
Numba retains remove/merge support for all eight supported indices, including
operations marked **Unchanged**. Backend selection does not guarantee a speedup;
see [Numba](#optional-numba-acceleration) and [JAX](#optional-jax-batch-backend)
for compilation, precision, and fallback details.

`CONN` uses prototype connectivity and has additional backend and normalization requirements.
See the [CONN guide][conn-guide] before using it.

## Optimizations

Except for `CONN`, initialized indices support adding samples, removing samples, merging clusters, and splitting clusters from tracked sufficient statistics without replaying the full dataset:

```python
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
```

These methods update the object in place and return its new criterion value.
Removing the final sample of a cluster deletes that cluster, while `merge` retains `target_label` and deletes `source_label`.
`split` retains `retained_label` for the residual cluster and assigns the split-off statistics to the unused `new_label`.
The caller is responsible for ensuring that a removed sample belongs to the supplied label.
`cvi` comes with some optimizations in the form of various backends that you can switch between for faster performance depending on your use-case.

### Optional FuzzyART CONN Backend

CONN uses KMeans prototypes by default and does not require ART dependencies.
Install the ART extra to use its FuzzyART prototype model for incremental or
batch evaluation:

```console
python -m pip install "cvi[art]"
```

```python
import cvi

index = cvi.CONN(model_type="Fuzzy")
value = index.get_cvi(sample, label)
```

The ART model is imported and initialized only when the Fuzzy-backed index is
first used. ART parameters and KMeans parameters are optional and only apply to
their corresponding prototype model.

### Optional Numba Acceleration

Install the extra and select the backend per index:

```console
python -m pip install "cvi[numba]"
```

```python
import cvi

index = cvi.XB(backend="numba")
value = index.get_cvi(samples, labels)
```

For a development checkout, install with `python -m pip install -e ".[numba]"`.
`backend="numpy"` remains the default. Numba acceleration is available for
`CH`, `WB`, `DB`, `XB`, `GD43`, `GD53`, `PS`, and `cSIL`, with the same batch,
incremental, remove, and merge API. `CONN` and `rCIP` currently support only the
NumPy numerical backend. CONN's `model_type` separately selects its clustering
algorithm.

The backend compiles grouping, compactness, centroid distances, and cSIL batch
distance assembly on the CPU. It retains NumPy's dtype-sensitive means and raw
moments. Float16, non-native byte-order arrays, and other unsupported array types
use NumPy for the affected operations. Results agree within floating-point
tolerances, rather than necessarily bit for bit. Unchanged operations, such as
CH/WB streaming updates, are not accelerated by this selection.

The first use of a kernel for an input type/layout incurs compilation; subsequent
calls reuse compiled code, with a disk cache across processes. Measure warmed
performance for your workload; small workloads may not repay compilation cost.
Numba is imported by the numerical backend only when selected. The optional
ART dependency may also install/use Numba independently when using Fuzzy CONN.
See [backend benchmarks](benchmarks/README.md) for timings and reproduction steps.

### Optional JAX Batch Backend

Install `cvi[jax]` (or `python -m pip install -e ".[jax]"` in a checkout), then
enable JAX's 64-bit mode explicitly:

```python
import jax
import cvi

jax.config.update("jax_enable_x64", True)
index = cvi.XB(backend="jax")
value = index.get_cvi(samples, labels)
```

JAX supports **batch CH, WB, and XB**, plus optional fixed-capacity streaming
for these indices. Remove and merge remain unsupported. Other indices
continue to support their existing backends. Importing CVI does not import JAX
or change its configuration.

The batch object interface returns a Python float and retains host-side state. It
preserves NumPy's input-dtype mean reductions, including float32 behavior, then
uses JAX for compactness, separation, and evaluation. For device-resident work
and composition with `jit`, `vmap`, or `grad`, use the functional interface:

```python
from functools import partial
from cvi.jax import batch_cvi

# Dense labels must be integers 0..2, with every cluster represented.
score = jax.jit(partial(batch_cvi, n_clusters=3, index="XB"))
device_value = score(device_samples, dense_labels)
```

The functional API evaluates real input data in float64 and uses shifted means
for numerical stability. `batch_state(...)` returns an immutable pytree that
`evaluate(state, index="CH")` can reuse for another supported index. Results
agree mathematically, with floating-point reduction differences. Cluster count
and index name are static under JIT; new input shapes can require recompilation.
Small CPU workloads and the compatibility object may be slower than NumPy.
See [benchmarks](benchmarks/README.md) for synchronized timing instructions.

Undefined CH, WB, and XB scores are `NaN` when fewer than two clusters are
present or their denominator is exactly zero: WGSS for CH, BGSS for WB, and
minimum centroid separation for XB. Denominators use exact zero checks, with no
epsilon adjustment. An undefined object batch evaluation emits a
`RuntimeWarning`; functional JAX and streaming calls return `NaN` silently.

### Fixed-capacity JAX Streaming

Reserve space for the maximum number of distinct clusters to enable streaming:

```python
index = cvi.XB(backend="jax", capacity=32)
value = index.get_cvi(sample, label)
history = index.update_many(next_samples, next_labels)
value = index.update_many(more_samples, more_labels, return_history=False)
```

`capacity` limits clusters, not samples. Arbitrary integer labels map to slots
in first-seen order. Exceeding capacity raises before any part of the call is
applied. An initial batch can also be followed by samples or chunks. Without
`capacity`, JAX objects retain their batch-only behavior.

Statistics stay on-device as a fixed-shape `index.stream_state`; scalar calls
return a Python float and chunks return NumPy histories (or a final float).
New clusters within capacity do not change array shapes. Streaming uses float64
and the existing incremental formulas, with unused slots excluded from scores.
The functional `empty_stream`, `stream_update`, `stream_chunk`, and
`evaluate_stream` functions support device-resident use inside JIT; see the
[guide](docs/source/guide.rst) for their slot and validation contracts.
Choose capacity near the expected maximum: XB reserves a capacity-by-capacity
distance matrix. Chunk processing amortizes Python and device synchronization
costs; per-sample JAX calls can be slower than NumPy.

## Acknowledgements

### Derivation

The incremental and batch CVI implementations in this package are largely derived from the following Julia language implementations by the same authors of this package:

- [ClusterValidityIndices.jl](https://github.com/AP6YC/ClusterValidityIndices.jl)

### Authors

The principal authors of the `cvi` pacakge are:

- Sasha Petrenko <petrenkos@mst.edu>
- Nik Melton <nmmz76@mst.edu>

### Related Projects

If this package is missing something that you need, feel free to check out some related Python cluster validity packages:

- [validclust](https://github.com/crew102/validclust)
- [clusterval](https://github.com/Nuno09/clusterval)

### Assets

#### Fonts

The following font is used in the logo:

- [Ethnocentric Font Family](https://www.1001fonts.com/ethnocentric-font.html)

#### Icons

The icon for the project is taken from:

- [Cluster computing icons created by IconBaandar - Flaticon](https://www.flaticon.com/free-icons/cluster-computing) ([cluster-5464694](https://www.flaticon.com/free-icon/cluster_5464694))
