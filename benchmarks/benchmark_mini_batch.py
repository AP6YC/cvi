"""Measure aggregate updates, sequential updates, and one-shot initialization.

Records are saved under benchmarks/results/mini_batch and registered in the
shared results/index.json. Timing excludes imports, data generation, and IO.
Peak traced allocations exclude resident input and are not process peak RSS.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from time import perf_counter
import tracemalloc

import numpy as np
import scipy

from src import cvi
from .benchmark_kernels import load_baseline


INDICES = ("CH", "WB", "DB", "XB", "GD43", "GD53", "PS", "cSIL", "rCIP")


def run(index_type, backend, mode, data, labels, chunk_size):
    """Construct a fresh index and return the score after all observations."""
    index = index_type(backend=backend)
    if mode == "batch":
        return index.get_cvi(data, labels)
    if mode == "stream":
        for sample, label in zip(data, labels):
            index.get_cvi(sample, int(label))
    else:
        for start in range(0, len(data), chunk_size):
            stop = start + chunk_size
            if mode == "mini_batch":
                index.update_batch(data[start:stop], labels[start:stop])
            else:
                # Control: sequential state maintenance, evaluated only at
                # chunk boundaries. This is intentionally a private API path.
                for sample, label in zip(data[start:stop], labels[start:stop]):
                    index._param_inc(sample, int(label))
                index._evaluate()
    return index.criterion_value


def main():
    """Validate scores, measure the requested workload, and record each case."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--clusters", type=int, default=24)
    parser.add_argument("--features", type=int, default=8)
    parser.add_argument("--chunk-sizes", nargs="+", type=int, default=[64, 1024, 8192])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=136)
    parser.add_argument("--indices", nargs="+", choices=INDICES)
    parser.add_argument("--backend", choices=["numpy", "numba"], default="numpy")
    parser.add_argument("--baseline", help="Saved cvi package to compare on identical inputs")
    args = parser.parse_args()
    args.indices = args.indices or [
        name for name in INDICES if args.backend in getattr(cvi, name).info.backends
    ]
    if any(args.backend not in getattr(cvi, name).info.backends for name in args.indices):
        parser.error("Selected index does not support the requested backend")
    if (not 2 <= args.clusters <= args.samples or args.features < 1
            or args.repeats < 1 or min(args.chunk_sizes) < 1):
        parser.error("Require N >= K >= 2, d >= 1, repeats >= 1, and chunks >= 1")

    root = Path("benchmarks/results")
    output = root / "mini_batch" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True)
    (output / "benchmark.py").write_bytes(Path(__file__).read_bytes())
    registry = root / "index.json"
    registry_data = json.loads(registry.read_text()) if registry.exists() else {"runs": []}
    registry_data["runs"].append(str((output / "run.json").relative_to(root)))
    registry.write_text(json.dumps(registry_data, indent=2) + "\n")
    record = {
        "hypothesis": (
            "Direct chunk aggregation reduces sequential state maintenance "
            "and repeated scoring while retaining the cumulative CVI."
        ),
        "configuration": vars(args),
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(Path("src/cvi").rglob("*.py"))},
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "scipy": scipy.__version__, "platform": platform.platform()},
        "evaluation": (
            "Each final score checked against float64 one-shot NumPy batch "
            "at rtol=1e-9, atol=1e-10. Seeded data reused across methods."
        ),
        "timing": (
            "Fresh object per trial; median after one warm call; includes "
            "construction, validation, grouping, updates, and evaluation. "
            "Excludes imports, data generation, and IO."
        ),
        "memory": (
            "Separate tracemalloc call for batch and mini_batch; excludes "
            "resident input, untraced native allocations, and IO. Not peak RSS."
        ),
        "limitations": ["Synthetic float64 CPU workload.",
                        "Warmed results exclude optional Numba compilation.",
                        "No JAX comparison or out-of-core IO measurement."],
        "status": "running", "results": [],
    }

    def save():
        (output / "run.json").write_text(json.dumps(record, indent=2) + "\n")

    save()
    print(output / "run.json", flush=True)
    try:
        versions = [("current", cvi)]
        if args.baseline:
            baseline = Path(args.baseline)
            versions.append(("baseline", load_baseline(baseline)))
            record["baseline_source_sha256"] = {
                str(p.relative_to(baseline)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(baseline.rglob("*.py"))
            }
        rng = np.random.default_rng(args.seed)
        labels = np.r_[np.arange(args.clusters),
                       rng.integers(args.clusters, size=args.samples - args.clusters)]
        rng.shuffle(labels)
        centers = rng.normal(size=(args.clusters, args.features)) * 3
        data = centers[labels] + rng.normal(size=(args.samples, args.features))
        cases = [("batch", args.samples), ("stream", 1)]
        cases += [(mode, size) for size in args.chunk_sizes
                  for mode in ("scan_final", "mini_batch")]
        for name, version, package in ((name, version, package) for name in args.indices
                                       for version, package in versions):
            index_type = getattr(package, name)
            expected = index_type().get_cvi(data, labels)
            for mode, size in cases:
                if mode == "mini_batch" and not index_type(backend=args.backend).capabilities.mini_batch:
                    continue
                actual = run(index_type, args.backend, mode, data, labels, size)
                np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-10)
                times = []
                for _ in range(args.repeats):
                    start = perf_counter()
                    run(index_type, args.backend, mode, data, labels, size)
                    times.append(perf_counter() - start)
                peak = None
                if mode in ("batch", "mini_batch"):
                    tracemalloc.start()
                    try:
                        run(index_type, args.backend, mode, data, labels, size)
                        _, peak = tracemalloc.get_traced_memory()
                    finally:
                        tracemalloc.stop()
                median = float(np.median(times))
                record["results"].append(dict(
                    index=name, version=version, mode=mode, chunk_size=size, seconds=times,
                    median_seconds=median, peak_traced_bytes=peak,
                    absolute_score_error=float(abs(actual - expected)),
                ))
                save()
                print(f"{name},{version},{mode},{size},{median * 1000:.3f} ms", flush=True)
        record["status"] = "complete"
    except Exception as error:
        record["status"], record["failure"] = "failed", repr(error)
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
