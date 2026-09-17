# Numerical kernel benchmarks

Run from the repository root with the project's dependencies installed:

```sh
python -m benchmarks.benchmark_kernels --samples 2000 --clusters 24 --features 8 --repeats 5
```

To compare a saved version, pass `--baseline /path/to/cvi`, where that directory
contains the earlier package's `__init__.py` and modules. Both versions run on
the same seeded data in one process. `--modes batch stream` limits the operations.
Batch and stream timings include object construction and score evaluation.
Single-operation timings exclude batch initialization of their fixtures. Each
trial gets a fresh object; each case is warmed once before taking the median.
These timings are descriptive and are not CI thresholds.

## Initial NumPy refactor

Measured against commit `8cfa922` on the local macOS machine with Python 3.14.7
and NumPy 2.5.2, using the command above. Times are milliseconds, median of five
trials, with no concurrent test run. These synthetic results depend on workload
and hardware; they do not predict Numba or JAX performance.

| Index | Original batch | Shared kernels | Speedup |
| --- | ---: | ---: | ---: |
| CH | 2.829 | 0.650 | 4.35x |
| WB | 2.804 | 0.635 | 4.41x |
| DB | 3.087 | 0.723 | 4.27x |
| XB | 3.021 | 0.657 | 4.60x |
| GD43 | 3.129 | 0.682 | 4.59x |
| GD53 | 2.742 | 0.670 | 4.09x |
| PS | 3.024 | 0.704 | 4.30x |
| cSIL | 59.038 | 0.982 | 60.13x |

Streaming through all 2,000 samples improved by 1.51x for DB, 2.30x for XB,
2.69x for GD43, and 1.32x for PS. Their remove/merge operations improved by
roughly 3–5x. Unmodified streaming paths (CH, WB, GD53, cSIL) were essentially
unchanged. Microsecond-scale single-operation timings are particularly noisy.

## Numerical compatibility

The NumPy implementation groups dense internal labels once and preserves the
original order of samples within each cluster. Means retain NumPy's original
input-dtype reduction followed by assignment into float64 centroid storage.
Centered compactness, appendable count/compactness lists, and first-seen external
label order are preserved. Distances use coordinate differences, avoiding Gram
matrix cancellation, and only require one row's temporary distance data.

cSIL batch mode retains its raw moments for subsequent add/remove/merge operations
but computes its dissimilarity matrix using centered statistics. For cluster
`i` and centroid `j`, it uses:

```text
delta = centroid[i] - centroid[j]
S[i, j] = centered_compactness[i] / count[i]
          + dot(delta, delta)
          + 2 * dot(sum(x - centroid[i]), delta) / count[i]
```

The final residual term preserves the direct-distance definition even when
floating-point means have nonzero residuals. The batch calculation no longer
allocates a sample-by-cluster distance matrix or scans all labels for every
cluster pair. It retains the existing orientation `S[cluster, centroid]` and
column-based silhouette evaluation. Floating-point reductions can differ in
their last bits; bitwise score identity is not promised.

The regression tests compare these calculations to direct sample definitions
for float64, float32, integer, non-contiguous, large-offset, singleton, and
coincident-centroid inputs. Existing undefined NaN/inf scores remain unchanged.
Operation tests cover batch-to-stream transitions, new labels, deletion, merging,
and state equivalence. At this stage all 337 tests pass; the new kernels have
100% statement coverage. Numba/JAX dependencies and backend selection are deferred
to the next approved implementation steps.
