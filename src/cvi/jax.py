"""Functional, device-resident JAX batch CVIs (CH, WB, and XB).

Import this module explicitly after installing ``cvi[jax]``. Enable JAX x64
in the application before calling these functions; importing CVI never changes
JAX configuration. Labels here must be dense integers in [0, n_clusters), with
every cluster represented. The object API handles arbitrary external labels.
"""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp


INDICES = ("CH", "WB", "XB")


class BatchState(NamedTuple):
    """Immutable pytree of sufficient statistics, stored on the JAX device."""

    counts: jax.Array
    centroids: jax.Array
    compactness: jax.Array
    mean: jax.Array
    n_samples: jax.Array


def _require_x64():
    if not jax.config.x64_enabled:
        raise ValueError(
            "CVI's JAX backend requires 64-bit arithmetic. Set JAX_ENABLE_X64=1 "
            "or call jax.config.update('jax_enable_x64', True) before use."
        )


def _check_index(index):
    if index not in INDICES:
        raise ValueError("JAX batch indices are 'CH', 'WB', and 'XB'")


def _summarize(data, labels, centroids, mean):
    """Centered statistics, also used with host-computed legacy centroids."""
    counts = jax.ops.segment_sum(
        jnp.ones(data.shape[0], dtype=jnp.int64), labels,
        num_segments=centroids.shape[0],
    )
    residuals = data - centroids[labels]
    compactness = jax.ops.segment_sum(
        jnp.sum(residuals ** 2, axis=1), labels,
        num_segments=centroids.shape[0],
    )
    return BatchState(counts, centroids, compactness, mean,
                      jnp.asarray(data.shape[0], dtype=jnp.int64))


@partial(jax.jit, static_argnames=("n_clusters",))
def _batch_state(data, labels, n_clusters):
    counts = jax.ops.segment_sum(
        jnp.ones(data.shape[0], dtype=jnp.int64), labels, n_clusters,
    )
    # Shift each cluster before summation to avoid summing a large common
    # coordinate offset N times. This is the same mean in exact arithmetic.
    first = jax.ops.segment_min(jnp.arange(data.shape[0]), labels, n_clusters)
    anchors = data[jnp.minimum(first, data.shape[0] - 1)]
    centered_sums = jax.ops.segment_sum(data - anchors[labels], labels, n_clusters)
    centroids = anchors + centered_sums / counts[:, None]
    mean = data[0] + jnp.mean(data - data[0], axis=0)
    return _summarize(data, labels, centroids, mean)


def batch_state(data, labels, *, n_clusters):
    """Compute a device-resident state from a batch and dense integer labels.

    ``n_clusters`` must be a static integer >= 2 under ``jax.jit``. Every label
    must be in [0, n_clusters), and every cluster must be nonempty. Shapes and
    dtypes are checked here; value-dependent label validation belongs to the
    caller when composing this function under JAX transformations.

    Real integer/float inputs are evaluated in float64. Shifted mean reductions
    can differ in their last bits from NumPy's unshifted reductions. No Python
    scalar conversion or host transfer occurs. The returned named tuple is a
    pytree suitable for ``jit``, ``vmap``, and differentiation with fixed labels.
    """
    _require_x64()
    data, labels = jnp.asarray(data), jnp.asarray(labels)
    if data.ndim != 2 or data.shape[0] == 0 or data.shape[1] == 0:
        raise ValueError("JAX batch data must be a nonempty two-dimensional array")
    if labels.ndim != 1 or labels.shape[0] != data.shape[0]:
        raise ValueError("JAX batch labels must contain one label per sample")
    if not jnp.issubdtype(labels.dtype, jnp.integer):
        raise ValueError("JAX batch labels must be integers")
    if not (jnp.issubdtype(data.dtype, jnp.floating)
            or jnp.issubdtype(data.dtype, jnp.integer)):
        raise ValueError("JAX batch data must contain real numbers")
    if not isinstance(n_clusters, int) or not 2 <= n_clusters <= data.shape[0]:
        raise ValueError("n_clusters must be a static integer between 2 and N")
    return _batch_state(data.astype(jnp.float64), labels, n_clusters)


def _derived(state, index):
    """Index-specific arrays and score, with the existing formulas/NaN policy."""
    counts, centers, cp, mean, n_samples = state
    k = centers.shape[0]
    within = jnp.sum(cp)
    if index in ("CH", "WB"):
        separation = counts * jnp.sum((centers - mean) ** 2, axis=1)
        between = jnp.sum(separation)
        if index == "CH":
            value = (between / within) * ((n_samples - k) / (k - 1))
        else:
            value = (within / between) * k
        derived = {"_SEP": separation, "_BGSS": between}
    else:
        # Map one row at a time: avoid a K-by-K-by-d intermediate and avoid
        # cancellation from a Gram-matrix distance identity.
        distances = jax.lax.map(
            lambda center: jnp.sum((centers - center) ** 2, axis=1), centers,
        )
        distances = distances.at[jnp.diag_indices(k)].set(0.0)
        separation = jnp.min(jnp.where(jnp.eye(k, dtype=bool), jnp.inf, distances))
        value = within / (n_samples * separation)
        derived = {"_D": distances, "_SEP": separation}
    # Invalid dense partitions must not produce a plausible score merely
    # because segment reductions drop an out-of-range label.
    valid = jnp.all(counts > 0) & (jnp.sum(counts) == n_samples)
    derived.update(_WGSS=within, criterion_value=jnp.where(valid, value, jnp.nan))
    return derived


@partial(jax.jit, static_argnames=("index",))
def _evaluate(state, index):
    return _derived(state, index)["criterion_value"]


def evaluate(state, *, index):
    """Evaluate CH, WB, or XB on a BatchState, returning a JAX scalar."""
    _require_x64()
    _check_index(index)
    return _evaluate(state, index)


def batch_cvi(data, labels, *, n_clusters, index):
    """Functional batch score; bind n_clusters/index statically under jit."""
    _check_index(index)
    return evaluate(batch_state(data, labels, n_clusters=n_clusters), index=index)


@partial(jax.jit, static_argnames=("index",))
def _host_batch(data, labels, centroids, mean, index):
    state = _summarize(data, labels, centroids, mean)
    return state, _derived(state, index)


__all__ = ["BatchState", "batch_state", "evaluate", "batch_cvi"]
