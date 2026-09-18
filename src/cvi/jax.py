"""Functional, device-resident JAX CVIs (CH, WB, and XB).

Import this module explicitly after installing ``cvi[jax]``. Enable JAX x64
in the application before calling these functions; importing CVI never changes
JAX configuration. Batch labels are dense integers in [0, n_clusters), with
every cluster represented. Streaming labels are slots in [0, capacity) and may
be sparse. The object API handles arbitrary external labels.
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
        raise ValueError("JAX indices are 'CH', 'WB', and 'XB'")


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


class StreamingState(NamedTuple):
    """Immutable fixed-shape statistics; inactive slots are excluded from scores.

    ``distances`` is capacity-by-capacity for XB and empty for CH/WB. Retain
    ``residuals`` to preserve the existing compactness update recurrence.
    """

    counts: jax.Array
    centroids: jax.Array
    compactness: jax.Array
    residuals: jax.Array
    mean: jax.Array
    n_samples: jax.Array
    active: jax.Array
    distances: jax.Array


def _positive_size(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive static integer")


def empty_stream(*, capacity, n_features, index):
    """Allocate an empty float64 stream; sizes and index must be static under JIT.

    Capacity limits distinct clusters, not samples. Zero/one-cluster scores
    are zero, matching the object API. No global JAX configuration is changed.
    """
    _require_x64()
    _check_index(index)
    _positive_size(capacity, "capacity")
    _positive_size(n_features, "n_features")
    return StreamingState(
        jnp.zeros(capacity, dtype=jnp.int64),
        jnp.zeros((capacity, n_features), dtype=jnp.float64),
        jnp.zeros(capacity, dtype=jnp.float64),
        jnp.zeros((capacity, n_features), dtype=jnp.float64),
        jnp.zeros(n_features, dtype=jnp.float64),
        jnp.asarray(0, dtype=jnp.int64), jnp.zeros(capacity, dtype=bool),
        jnp.zeros((capacity, capacity) if index == "XB" else (0, 0),
                  dtype=jnp.float64),
    )


def stream_from_batch(state, *, capacity, index):
    """Pad a BatchState into a stream, preserving its statistics and slot order.

    As in object batch initialization, residual corrections start at zero.
    The input must be a valid BatchState with every cluster represented.
    """
    result = empty_stream(capacity=capacity, n_features=state.centroids.shape[1],
                          index=index)
    k = state.centroids.shape[0]
    if k > capacity:
        raise ValueError("Cluster count exceeds capacity")
    distances = result.distances
    if index == "XB":
        distances = distances.at[:k, :k].set(_derived(state, index)["_D"])
    return result._replace(
        counts=result.counts.at[:k].set(state.counts),
        centroids=result.centroids.at[:k].set(state.centroids),
        compactness=result.compactness.at[:k].set(state.compactness),
        mean=state.mean, n_samples=state.n_samples,
        active=result.active.at[:k].set(True), distances=distances,
    )


def _check_stream(state, index):
    _require_x64()
    _check_index(index)
    capacity, dim = state.centroids.shape
    expected = (capacity, capacity) if index == "XB" else (0, 0)
    if state.distances.shape != expected:
        raise ValueError("Streaming state does not match the requested index")
    return capacity, dim


@partial(jax.jit, static_argnames=("index",))
def _stream_derived(state, index):
    k = jnp.sum(state.active)
    within = jnp.sum(jnp.where(state.active, state.compactness, 0.0))
    if index in ("CH", "WB"):
        separation = jnp.where(
            state.active,
            state.counts * jnp.sum((state.centroids - state.mean) ** 2, axis=1),
            0.0,
        )
        between = jnp.sum(separation)
        if index == "CH":
            value = (between / within) * ((state.n_samples - k) / (k - 1))
        else:
            value = (within / between) * k
        derived = {"_SEP": separation, "_BGSS": between}
    else:
        pairs = state.active[:, None] & state.active[None, :]
        pairs &= ~jnp.eye(state.counts.shape[0], dtype=bool)
        separation = jnp.min(jnp.where(pairs, state.distances, jnp.inf))
        value = within / (state.n_samples * separation)
        derived = {"_D": state.distances, "_SEP": separation}
    derived.update(_WGSS=within, criterion_value=jnp.where(k > 1, value, 0.0))
    return derived


def evaluate_stream(state, *, index):
    """Evaluate active clusters, returning a device scalar (zero for <2 clusters)."""
    _check_stream(state, index)
    return _stream_derived(state, index)["criterion_value"]


def _advance(state, sample, slot, index):
    """Existing NumPy incremental recurrence, including residual corrections."""
    n_old = state.counts[slot]
    n_new = n_old + 1
    old_center = state.centroids[slot]
    center = (1.0 - 1.0 / n_new) * old_center + (1.0 / n_new) * sample
    delta = old_center - center
    residual = sample - center
    cp = (state.compactness[slot] + jnp.dot(residual, residual)
          + n_old * jnp.dot(delta, delta)
          + 2.0 * jnp.dot(delta, state.residuals[slot]))
    # A new cluster has exactly zero compactness, even when squaring its
    # finite coordinates would overflow (0 * inf must not contaminate it).
    cp = jnp.where(n_old == 0, 0.0, cp)
    correction = state.residuals[slot] + residual + n_old * delta
    total = state.n_samples + 1
    mean = (1.0 - 1.0 / total) * state.mean + (1.0 / total) * sample
    active = state.active.at[slot].set(True)
    centers = state.centroids.at[slot].set(center)
    distances = state.distances
    if index == "XB":
        row = jnp.where(active, jnp.sum((centers - center) ** 2, axis=1), 0.0)
        row = row.at[slot].set(0.0)
        distances = distances.at[slot, :].set(row).at[:, slot].set(row)
    return StreamingState(
        state.counts.at[slot].set(n_new), centers,
        state.compactness.at[slot].set(cp),
        state.residuals.at[slot].set(correction), mean, total, active, distances,
    )


def _stream_inputs(state, data, slots, index):
    _, dim = _check_stream(state, index)
    data, slots = jnp.asarray(data), jnp.asarray(slots)
    if data.ndim != 2 or data.shape[1] != dim:
        raise ValueError(f"Streaming data must have shape (N, {dim})")
    if slots.ndim != 1 or slots.shape[0] != data.shape[0]:
        raise ValueError("Streaming slots must contain one slot per sample")
    if not jnp.issubdtype(slots.dtype, jnp.integer):
        raise ValueError("Streaming slots must be integers")
    if not (jnp.issubdtype(data.dtype, jnp.floating)
            or jnp.issubdtype(data.dtype, jnp.integer)):
        raise ValueError("Streaming data must contain real numbers")
    return data.astype(jnp.float64), slots


@partial(jax.jit, static_argnames=("index",))
def _stream_update(state, sample, slot, index):
    valid = (slot >= 0) & (slot < state.counts.shape[0]) & jnp.all(jnp.isfinite(sample))

    def update(_):
        result = _advance(state, sample, slot, index)
        return result, _stream_derived(result, index)["criterion_value"]
    return jax.lax.cond(valid, update, lambda _: (state, jnp.asarray(jnp.nan)), None)


def stream_update(state, sample, slot, *, index):
    """Add one sample to a slot and return (new_state, device_score).

    Slots are integers in [0, capacity); unused slots need not be contiguous.
    Invalid slot values or nonfinite samples return the unchanged state and NaN,
    including under JIT. Shape/dtype errors raise ValueError before dispatch.
    """
    sample, slot = jnp.asarray(sample), jnp.asarray(slot)
    if sample.ndim != 1 or slot.ndim != 0:
        raise ValueError("Expected a sample vector and scalar slot")
    data, slots = _stream_inputs(state, sample[None, :], slot[None], index)
    return _stream_update(state, data[0], slots[0], index)


@partial(jax.jit, static_argnames=("index", "return_history"))
def _stream_chunk(state, data, slots, index, return_history):
    valid = (jnp.all((slots >= 0) & (slots < state.counts.shape[0]))
             & jnp.all(jnp.isfinite(data)))

    def update(_):
        def step(current, inputs):
            result = _advance(current, *inputs, index)
            score = (_stream_derived(result, index)["criterion_value"]
                     if return_history else None)
            return result, score
        result, history = jax.lax.scan(step, state, (data, slots))
        return result, (history if return_history else
                        _stream_derived(result, index)["criterion_value"])
    shape = (data.shape[0],) if return_history else ()
    return jax.lax.cond(valid, update,
                        lambda _: (state, jnp.full(shape, jnp.nan)), None)


def stream_chunk(state, data, slots, *, index, return_history=True):
    """Scan a chunk, returning (new_state, score_history_or_final_score).

    All inputs are validated before any update: an out-of-range slot or nonfinite
    sample returns the original state and NaN output for the entire chunk.
    Empty chunks are no-ops. Bind index/return_history statically under JIT;
    changing chunk length can compile another specialization. Final-only mode
    avoids both per-sample score evaluation and allocating a history array.
    """
    if not isinstance(return_history, bool):
        raise ValueError("return_history must be a static boolean")
    data, slots = _stream_inputs(state, data, slots, index)
    return _stream_chunk(state, data, slots, index, return_history)


__all__ += ["StreamingState", "empty_stream", "stream_from_batch",
            "evaluate_stream", "stream_update", "stream_chunk"]
