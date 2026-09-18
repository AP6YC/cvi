"""Synchronized fixed-capacity streaming timings, including compile and host costs.

Run with JAX_ENABLE_X64=1 python -m benchmarks.benchmark_jax_stream.
"""

import argparse
from functools import partial
from statistics import median
from time import perf_counter

import jax
import numpy as np

import src.cvi as cvi
from src.cvi.jax import empty_stream, stream_chunk


def milliseconds(run, repeats):
    times = []
    for _ in range(repeats):
        start = perf_counter()
        jax.block_until_ready(run())
        times.append(1000 * (perf_counter() - start))
    return median(times)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--clusters", type=int, default=24)
    parser.add_argument("--capacity", type=int, default=32)
    parser.add_argument("--features", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if not jax.config.x64_enabled:
        parser.error("Set JAX_ENABLE_X64=1")
    if not (2 <= args.clusters <= min(args.samples, args.capacity)
            and args.features > 0 and args.repeats > 0):
        parser.error("Require 2 <= clusters <= min(samples, capacity), features/repeats > 0")
    rng = np.random.default_rng(318)
    labels = np.arange(args.samples, dtype=np.int64) % args.clusters
    centers = rng.normal(size=(args.clusters, args.features)) * 3
    data = centers[labels] + rng.normal(size=(args.samples, args.features))
    device_data, device_labels = jax.device_put((data, labels))
    jax.block_until_ready((device_data, device_labels))
    print(f"JAX {jax.__version__}, {jax.devices()}; N={args.samples}, K={args.clusters}, "
          f"capacity={args.capacity}, d={args.features}; median ms, {args.repeats} repeats")
    print("index,compile_history,compile_final,resident_history,resident_final,"
          "object_chunk,object_samples,numpy_samples")
    for name in ("CH", "WB", "XB"):
        index_type = getattr(cvi, name)
        state = empty_stream(capacity=args.capacity, n_features=args.features, index=name)
        jax.block_until_ready(state)
        history_fn = jax.jit(partial(stream_chunk, index=name))
        final_fn = jax.jit(partial(stream_chunk, index=name, return_history=False))

        def resident_history():
            return history_fn(state, device_data, device_labels)

        def resident_final():
            return final_fn(state, device_data, device_labels)

        first_history = milliseconds(resident_history, 1)
        first_final = milliseconds(resident_final, 1)

        def sequential(backend):
            kwargs = {"capacity": args.capacity} if backend == "jax" else {}
            obj = index_type(backend=backend, **kwargs)
            with np.errstate(divide="ignore", invalid="ignore"):
                values = np.array([obj.get_cvi(x, int(y)) for x, y in zip(data, labels)])
            return obj, values

        expected, reference = sequential("numpy")
        actual, values = resident_history()
        np.testing.assert_allclose(values, reference, rtol=1e-9, atol=1e-10, equal_nan=True)
        np.testing.assert_allclose(actual.centroids[:args.clusters], expected._v,
                                   rtol=1e-9, atol=1e-10)
        np.testing.assert_allclose(actual.compactness[:args.clusters], expected._CP,
                                   rtol=1e-9, atol=1e-10)
        np.testing.assert_allclose(resident_final()[1], reference[-1], rtol=1e-9)
        _, object_values = sequential("jax")  # warm and verify scalar adapter
        np.testing.assert_allclose(object_values, reference, rtol=1e-9, equal_nan=True)

        def object_chunk():
            return index_type(backend="jax", capacity=args.capacity).update_many(data, labels)

        np.testing.assert_allclose(object_chunk(), reference, rtol=1e-9, equal_nan=True)
        timings = [milliseconds(fn, args.repeats) for fn in
                   [resident_history, resident_final, object_chunk,
                    lambda: sequential("jax")[1], lambda: sequential("numpy")[1]]]
        print(name + "," + ",".join(f"{x:.3f}" for x in
                                    [first_history, first_final, *timings]))


if __name__ == "__main__":
    main()
