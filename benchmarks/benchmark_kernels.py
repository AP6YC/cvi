"""Reproducible timings for batch, streaming, and structural CVI operations.

Run from the repository root, for example::

    python -m benchmarks.benchmark_kernels --samples 4000 --clusters 32

An optional --baseline points to a saved cvi package directory (containing
__init__.py), allowing both implementations to run on identical inputs in the
same process. Timings are descriptive, not assertions or CI performance gates.
"""

import argparse
import importlib.util
from pathlib import Path
from statistics import median
import sys
from time import perf_counter

import numpy as np

import src.cvi as cvi


INDICES = ("CH", "WB", "DB", "XB", "GD43", "GD53", "PS", "cSIL")
MODES = ("batch", "stream", "add-existing", "add-new", "remove", "merge")


def load_baseline(directory):
    spec = importlib.util.spec_from_file_location(
        "_cvi_benchmark_baseline", Path(directory) / "__init__.py",
        submodule_search_locations=[str(Path(directory).resolve())],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def operation(cvi_type, mode, data, labels):
    """Prepare an operation; fixture initialization is outside its timing."""
    if mode == "batch":
        return lambda: cvi_type().get_cvi(data, labels)
    if mode == "stream":
        def stream():
            index = cvi_type()
            for sample, label in zip(data, labels):
                index.get_cvi(sample, int(label))
        return stream

    index = cvi_type()
    index.get_cvi(data, labels)
    if mode == "add-existing":
        return lambda: index.get_cvi(data[0], int(labels[0]))
    if mode == "add-new":
        return lambda: index.get_cvi(data[0], int(labels.max()) + 1)
    if mode == "remove":
        return lambda: index.remove(data[0], int(labels[0]))
    return lambda: index.merge(int(labels[0]), int(labels[1]))


def measure(cvi_type, mode, data, labels, repeats):
    operation(cvi_type, mode, data, labels)()  # Warm up imports/caches.
    times = []
    for _ in range(repeats):
        run = operation(cvi_type, mode, data, labels)
        start = perf_counter()
        run()
        times.append(perf_counter() - start)
    return median(times) * 1000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--clusters", type=int, default=24)
    parser.add_argument("--features", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    args = parser.parse_args()
    if (args.clusters < 2 or args.samples < 2 * args.clusters
            or args.features < 1 or args.repeats < 1):
        parser.error("Require K >= 2, N >= 2K, d >= 1, and repeats >= 1")

    rng = np.random.default_rng(318)
    labels = np.concatenate((np.arange(args.clusters), np.arange(args.clusters),
                             rng.integers(args.clusters,
                                          size=args.samples - 2 * args.clusters)))
    centers = rng.normal(size=(args.clusters, args.features)) * 3
    data = centers[labels] + rng.normal(size=(args.samples, args.features))
    baseline = load_baseline(args.baseline) if args.baseline else None

    print(f"Python {sys.version.split()[0]}, NumPy {np.__version__}; "
          f"N={args.samples}, K={args.clusters}, d={args.features}; "
          f"median of {args.repeats}, milliseconds")
    print("index,operation,current_ms,baseline_ms,speedup")
    with np.errstate(divide="ignore", invalid="ignore"):
        for name in INDICES:
            for mode in args.modes:
                current = measure(getattr(cvi, name), mode, data, labels,
                                  args.repeats)
                if baseline is None:
                    print(f"{name},{mode},{current:.6f},,")
                else:
                    previous = measure(getattr(baseline, name), mode, data,
                                       labels, args.repeats)
                    print(f"{name},{mode},{current:.6f},{previous:.6f},"
                          f"{previous / current:.2f}")


if __name__ == "__main__":
    main()
