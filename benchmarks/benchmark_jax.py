"""Separate first-call, device-resident, and host-input JAX batch timings.

Run with JAX_ENABLE_X64=1 python -m benchmarks.benchmark_jax. Synchronization
is included in every timed call; device placement is outside resident timings.
"""

import argparse
from functools import partial
from statistics import median
from time import perf_counter

import jax
import numpy as np

import src.cvi as cvi
from src.cvi.jax import batch_cvi


def milliseconds(run, repeats):
    times = []
    for _ in range(repeats):
        start = perf_counter()
        value = run()
        if hasattr(value, "block_until_ready"):
            value.block_until_ready()
        times.append((perf_counter() - start) * 1000)
    return median(times)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--clusters", type=int, default=24)
    parser.add_argument("--features", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if not jax.config.x64_enabled:
        parser.error("Set JAX_ENABLE_X64=1 before running this benchmark")
    if not (2 <= args.clusters <= args.samples
            and args.features > 0 and args.repeats > 0):
        parser.error("Require 2 <= K <= N, d > 0, and repeats > 0")
    rng = np.random.default_rng(318)
    labels = np.arange(args.samples, dtype=np.int64) % args.clusters
    centers = rng.normal(size=(args.clusters, args.features)) * 3
    data = centers[labels] + rng.normal(size=(args.samples, args.features))
    device_data, device_labels = jax.device_put((data, labels))
    jax.block_until_ready((device_data, device_labels))
    print(f"JAX {jax.__version__}, {jax.devices()}; N={args.samples}, "
          f"K={args.clusters}, d={args.features}; median of {args.repeats}, ms")
    print("index,first_resident_call_ms,resident_ms,host_input_ms,object_ms,numpy_ms")
    for name in ("CH", "WB", "XB"):
        compiled = jax.jit(partial(batch_cvi, n_clusters=args.clusters, index=name))

        def resident():
            return compiled(device_data, device_labels)

        first = milliseconds(resident, 1)
        expected = getattr(cvi, name)().get_cvi(data, labels)
        np.testing.assert_allclose(resident(), expected, rtol=1e-10, atol=1e-12)
        resident_ms = milliseconds(resident, args.repeats)
        host_ms = milliseconds(lambda: compiled(data, labels), args.repeats)

        def object_call():
            return getattr(cvi, name)(backend="jax").get_cvi(data, labels)

        object_call()  # Compile the compatibility adapter before timing it.
        object_ms = milliseconds(object_call, args.repeats)
        numpy_ms = milliseconds(lambda: getattr(cvi, name)().get_cvi(data, labels),
                                args.repeats)
        print(f"{name},{first:.3f},{resident_ms:.6f},{host_ms:.6f},"
              f"{object_ms:.6f},{numpy_ms:.6f}")


if __name__ == "__main__":
    main()
