Backends
========

`cvi` comes with some optimizations in the form of various backends that you can switch between for faster performance depending on your use-case.

Optional CPU acceleration
-------------------------

Install the Numba extra to enable compiled numerical kernels:

.. code-block:: console

   python -m pip install "cvi[numba]"

For a development checkout, use ``python -m pip install -e ".[numba]"``.
Select the backend when constructing an index:

.. code-block:: python

   index = cvi.XB(backend="numba")
   value = index.get_cvi(samples, labels)
   assert index.backend == "numba"

NumPy remains the default. Numba is supported by ``CH``, ``WB``, ``DB``, ``XB``,
``GD43``, ``GD53``, ``PS``, and ``cSIL``. ``CONN`` and ``rCIP`` reject
``backend="numba"``. CONN's numerical backend is separate from ``model_type``.
Selection is per object and remains fixed through removal of all samples and
reinitialization. The batch, incremental, remove, and merge APIs are unchanged.

Compiled kernels accelerate grouping, compactness, centroid distances, and
cSIL batch dissimilarities. Dtype-sensitive means and raw-moment reductions
remain in NumPy. Unsupported array types (including float16 and non-native
byte order) use NumPy for the affected operations. Floating-point rounding may
differ; bitwise equivalence is not guaranteed. Existing undefined NaN/inf scores
retain their meaning. Unchanged paths, including CH/WB streaming updates and
cSIL incremental updates, are not accelerated.

The first invocation for a new input type/layout incurs compilation. Later
calls reuse compiled code, with a disk cache across processes. Include that
initial latency when measuring short tasks, and warm up kernels before measuring
steady-state throughput. Numba is imported for the numerical backend only when
selected; CONN's existing ART dependency can independently install and use it.
Selecting Numba without the dependency installed raises an installation hint
rather than silently selecting another backend.

JAX batch evaluation
--------------------

Install ``cvi[jax]`` or, from a checkout, ``python -m pip install -e ".[jax]"``.
JAX supports batch CH, WB, and XB:

.. code-block:: python

   import jax

   jax.config.update("jax_enable_x64", True)
   index = cvi.XB(backend="jax")
   value = index.get_cvi(samples, labels)

Alternatively, set ``JAX_ENABLE_X64=1`` before starting Python. CVI never changes
JAX's global configuration and reports an error if 64-bit mode is disabled.
NumPy remains the default and does not load JAX. The class adapter accepts real
integer/float inputs up to 64 bits and arbitrary integer labels. It validates
the batch before changing state and retains first-seen label ordering.

The adapter computes input-dtype means with NumPy to preserve existing float32
and large-offset behavior. Compactness, separation, and scores are computed by
JAX, then transferred back to the object's NumPy state and Python scalar result.
This adapter synchronizes with the device; it is not intended for use inside
``jax.jit``. Without ``capacity``, incremental updates raise
``NotImplementedError``. Remove and merge are unsupported for all JAX objects.
Other indices do not yet support ``backend="jax"``.

For device-resident calculations, use the functional interface:

.. code-block:: python

   from functools import partial
   import jax.numpy as jnp
   from cvi.jax import batch_cvi, batch_state, evaluate

   x = jnp.array([[0., 0.], [1., 1.], [4., 4.], [5., 5.]])
   dense_labels = jnp.array([0, 0, 1, 1])
   score = jax.jit(partial(batch_cvi, n_clusters=2, index="XB"))
   device_value = score(x, dense_labels)
   state = batch_state(x, dense_labels, n_clusters=2)
   ch_value = evaluate(state, index="CH")

Functional labels must be dense integers in ``[0, n_clusters)`` with every
cluster represented. Unlike the class adapter, this interface leaves
value-dependent label validation to the caller so it can run under JAX
transformations. Invalid partitions produce undefined scores. ``n_clusters``
and ``index`` must be static under JIT, and new array shapes can recompile.
``BatchState`` is an immutable pytree of counts, centroids, centered compactness,
global mean, and sample count. It contains JAX arrays and remains on-device.

The functional interface computes in float64, using shifted means to reduce
cancellation for large offsets. Reduction order differs from NumPy and may vary
by device; bitwise identity is not promised. It supports ``vmap`` across batches
with compatible shapes and ``grad`` with respect to data for fixed labels where
the score is differentiable. Degenerate NaN/inf scores retain the index formulas.
Time JAX results with ``block_until_ready()`` and separate first-use compilation
from warmed execution. Host transfers and small CPU workloads can outweigh the
benefit of compilation.

Fixed-capacity JAX streaming
----------------------------

Opt into streaming for CH, WB, or XB by reserving cluster slots:

.. code-block:: python

   index = cvi.XB(backend="jax", capacity=32)
   value = index.get_cvi(samples[0], int(labels[0]))
   history = index.update_many(samples[1:100], labels[1:100])
   final_value = index.update_many(samples[100:], labels[100:],
                                   return_history=False)

``capacity`` is a positive integer limiting the number of distinct labels,
not the number of samples. Feature dimension is inferred on the first nonempty
call and then fixed. Arbitrary integer labels map to slots in first-seen order.
Unused slots do not contribute to the score. Capacity cannot be resized; choose
a suitable bound or create a new object. It is only accepted with JAX.

``get_cvi`` accepts a sample, or a one-time initial batch with at least two
clusters. An initial batch can be followed by samples or ``update_many`` chunks.
Chunk processing uses the incremental recurrence in input order; it is distinct
from batch initialization. Empty chunks are no-ops and return an empty history
or the current score. Zero/one-cluster streams return zero; otherwise existing
NaN/inf score behavior is preserved.

The adapter checks the entire input before dispatch: data must be finite real
numbers up to 64 bits, labels must be integers, dimensions must match, and all
new labels must fit. A rejected sample, batch, or chunk leaves both statistics
and label mapping unchanged. A full stream still accepts existing labels.
Removal and merging remain unsupported.

Streaming arithmetic is float64, including float32 input conversion. Batch
initialization retains the batch adapter's input-dtype means; subsequent samples
use the float64 incremental recurrence, including its residual correction.
Floating-point operation ordering can differ across devices, so results are
numerically equivalent rather than bitwise identical.

``index.stream_state`` exposes immutable, padded JAX arrays; it is ``None``
until the first nonempty call. Numerical state stays on-device. Object methods
synchronize to return a Python scalar or NumPy score history. For calculations
inside JIT, use the functional API directly:

.. code-block:: python

   from cvi.jax import empty_stream, stream_chunk, stream_update

   state = empty_stream(capacity=32, n_features=2, index="XB")
   state, score = stream_update(state, jnp.array([1., 2.]), 7, index="XB")
   state, history = stream_chunk(
       state, jnp.array([[2., 3.], [5., 6.]]), jnp.array([7, 12]), index="XB",
   )

Functional labels are slot indices in ``[0, capacity)``. Slots may be sparse;
this interface does not map external labels. ``StreamingState`` contains counts,
centroids, compactness, residual corrections, global mean, sample count, active
flags, and XB distances. CH/WB use an empty distance array. Use
``stream_from_batch(batch_state, capacity=32, index="XB")`` to pad an existing
valid ``BatchState`` and ``evaluate_stream(state, index="XB")`` to evaluate it.
CH/WB share a state layout; XB requires its distance-matrix layout.

Functional shape/dtype errors raise ``ValueError``. To work under JIT,
out-of-range slot values or nonfinite samples return the unchanged state and
NaN output; one invalid row rejects an entire chunk. Check outputs as appropriate
for your application. This differs from the object API's host-side exceptions.

Bind ``index`` and ``return_history`` statically under JIT. Capacity and feature
dimension fix the state shapes, so adding a cluster within capacity does not
cause recompilation. Changing chunk length can compile a new specialization.
``return_history=False`` avoids allocating a history and evaluates the score
only after the scan. CH/WB storage scales as capacity times feature dimension;
XB additionally stores a square distance matrix. A large capacity increases
work on padded arrays. Prefer chunks for throughput; individual Python calls
can cost more than NumPy. See ``benchmarks/benchmark_jax_stream.py`` for
synchronized compilation, resident chunk, and object timings.
