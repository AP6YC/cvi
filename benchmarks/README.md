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
and state equivalence. At the initial refactor stage all 337 tests passed; the
NumPy kernels had 100% statement coverage.

## Optional Numba backend

Install the extra with `python -m pip install -e ".[numba]"`, then compare the
two current backends (not the pre-refactor implementation):

```sh
python -m benchmarks.benchmark_kernels --backend numba --compare-backend numpy --samples 2000 --clusters 24 --features 8 --repeats 5
python -m benchmarks.benchmark_kernels --backend numba --compare-backend numpy --samples 4000 --clusters 128 --features 16 --repeats 5 --indices XB GD43 cSIL --modes batch stream remove merge
```

On the same local macOS arm64 environment, with Numba 0.67.0, Python 3.14.7, and
NumPy 2.5.2, warmed batch timings were:

| Index | N / K / d | NumPy (ms) | Numba (ms) | Speedup |
| --- | --- | ---: | ---: | ---: |
| CH | 2000 / 24 / 8 | 0.662 | 0.573 | 1.16x |
| WB | 2000 / 24 / 8 | 0.697 | 0.585 | 1.19x |
| DB | 2000 / 24 / 8 | 0.759 | 0.619 | 1.23x |
| XB | 2000 / 24 / 8 | 0.737 | 0.585 | 1.26x |
| GD43 | 2000 / 24 / 8 | 0.740 | 0.589 | 1.26x |
| GD53 | 2000 / 24 / 8 | 0.718 | 0.619 | 1.16x |
| PS | 2000 / 24 / 8 | 0.727 | 0.621 | 1.17x |
| cSIL | 2000 / 24 / 8 | 1.006 | 0.803 | 1.25x |
| XB | 4000 / 128 / 16 | 2.162 | 1.405 | 1.54x |
| GD43 | 4000 / 128 / 16 | 2.120 | 1.406 | 1.51x |
| cSIL | 4000 / 128 / 16 | 3.979 | 2.404 | 1.66x |

For 128 clusters, XB remove/merge improved by 5.17x/4.82x and GD43 by
7.04x/6.71x because their full pairwise distance rebuilds are compiled. Streaming
through 4,000 samples improved by only 1.03x for XB and 1.12x for GD43; Python
state updates and evaluations remain significant. cSIL streaming and structural
operations are unchanged and measured approximately equal throughput. Numba is
therefore most useful here for repeated batch calculations or distance rebuilds,
not as a blanket speedup for every index and operation.

To measure uncached first-use latency separately:

```sh
python -m benchmarks.benchmark_kernels --backend numba --compare-backend numpy --samples 2000 --clusters 24 --features 8 --repeats 5 --indices XB cSIL --modes batch --cold
```

Each cold measurement uses a fresh process and empty temporary Numba cache. It
includes backend import, compilation, construction, and the first batch call;
process startup and initial CVI/NumPy imports are outside the timer. The observed
single-run first-call latencies were 489 ms for XB and 567 ms for cSIL, compared
with warmed sub-millisecond calls. The warm columns remain medians of five
trials. First use of another dtype/layout can require another specialization.

Numba kernels use `fastmath=False` and serial loops. Grouping is a stable
counting pass; compactness and distances avoid large intermediate arrays. NumPy
still computes means and raw moments to preserve input-dtype reduction behavior.
For unsupported array types, the affected operation uses the NumPy reference.
No performance thresholds are imposed by the test suite.

## JAX batch support

The first JAX implementation covers CH, WB, and XB batch initialization. Install
with `python -m pip install -e ".[jax]"` and enable x64 explicitly:

```sh
JAX_ENABLE_X64=1 python -m benchmarks.benchmark_jax --samples 2000 --clusters 24 --features 8 --repeats 7
JAX_ENABLE_X64=1 python -m benchmarks.benchmark_jax --samples 20000 --clusters 64 --features 16 --repeats 7
```

Measured on the local macOS arm64 CPU with JAX/jaxlib 0.11.2, Python 3.14.7,
and NumPy 2.5.2. Warm columns are medians of seven synchronized calls, in ms:

| Index | N / K / d | Functional, resident | Functional, host input | JAX object | NumPy object |
| --- | --- | ---: | ---: | ---: | ---: |
| CH | 2000 / 24 / 8 | 0.046 | 0.040 | 0.476 | 0.647 |
| WB | 2000 / 24 / 8 | 0.049 | 0.051 | 0.438 | 0.633 |
| XB | 2000 / 24 / 8 | 0.056 | 0.060 | 0.449 | 0.709 |
| CH | 20000 / 64 / 16 | 0.352 | 0.383 | 2.982 | 6.040 |
| WB | 20000 / 64 / 16 | 0.376 | 0.375 | 2.943 | 6.035 |
| XB | 20000 / 64 / 16 | 0.335 | 0.327 | 2.991 | 6.332 |

The first functional call took approximately 40–68 ms including compilation,
but excluding imports and initial device placement. Those are single first-call
measurements, not medians. Each workload command ran in a fresh process. CPU
timings do not establish GPU performance; differences of a few microseconds
between host and resident inputs are measurement noise.

The functional resident timing includes computing statistics and the score,
with inputs already on the device and `block_until_ready()` on every result.
Host-input timings pass NumPy arrays to the compiled function. The functional
path returns a JAX scalar and allows compiler optimization across the complete
calculation. The object paths also construct an object and retain its summary
state; the JAX object additionally encodes external labels and preserves NumPy's
mean reductions on the host, then copies the JAX results back. These are distinct
interfaces, not interchangeable performance claims. The object adapter is useful
for compatibility; the functional interface is intended for JAX applications.

For the object-only comparison, including a fresh-process first batch with an
empty compilation cache, use:

```sh
JAX_ENABLE_X64=1 python -m benchmarks.benchmark_kernels --backend jax --compare-backend numpy --cold
```

This defaults to supported batch indices and rejects unsupported operations.
`cvi.jax.batch_state` and `evaluate` can reuse an immutable device-resident
summary for several supported indices. They require dense labels and static
cluster counts under JIT. Fixed-capacity streaming is covered below; JAX remove
and merge remain unsupported. The regression suite checks NumPy equivalence,
degenerate scores, float32 compatibility, large offsets, `jit`, `vmap`, gradients,
dependency isolation, and failure atomicity. No global JAX settings are modified
by library imports or constructors.

## Fixed-capacity JAX streaming

```bash
JAX_ENABLE_X64=1 python -m benchmarks.benchmark_jax_stream --samples 2000 --clusters 24 --capacity 32
```

This verifies complete score histories and final statistics against NumPy before
reporting synchronized timings. First calls include compilation. Warm timings
separate device-resident scans (history and final-only), object chunks including
host validation/transfers, and Python per-sample calls for JAX and NumPy. Device
placement is outside resident timings. Vary `--capacity` independently of
`--clusters` to measure padding costs, especially XB's square distance matrix.
Single-sample dispatch may be slower than NumPy even when compiled chunks win.
