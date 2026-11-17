r"""Per-sample cost of plain streaming commits (``get_cvi`` without scoring).

Times the incremental parameter update of each index on a stream that has
already reached its final cluster count, optionally against another checkout
of the package (``--baseline``). Each package runs in its own subprocess, and
rounds alternate between them so that machine drift affects both equally.
Nothing is written to disk.

Example::

    git worktree add /tmp/cvi-develop develop
    OPENBLAS_NUM_THREADS=1 python -m benchmarks.benchmark_streaming_add \\
        --baseline /tmp/cvi-develop
"""

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

INDICES = ["CH", "WB", "XB", "DB", "GD43", "GD53", "PS", "cSIL", "rCIP"]


def worker(root, indices, clusters, features, warmup, samples, repeats, seed):
    """Print the best per-sample update time (microseconds) of each index."""
    import copy
    import time

    import numpy as np

    sys.path.insert(0, str(root))
    import src.cvi as cvi

    rng = np.random.default_rng(seed)
    n = warmup + samples
    labels = rng.integers(0, clusters, n)
    labels[:clusters] = np.arange(clusters)
    data = rng.normal(size=(n, features)) + 0.3 * labels[:, None]
    stream = list(zip(data[warmup:], labels[warmup:].tolist()))
    timings = {}
    for name in indices:
        index = getattr(cvi, name)()
        for sample, label in zip(data[:warmup], labels[:warmup]):
            index.get_cvi(sample, int(label))
        best = float("inf")
        for _ in range(repeats):
            trial = copy.deepcopy(index)
            update = trial._param_inc
            start = time.perf_counter()
            for sample, label in stream:
                update(sample, label)
            best = min(best, time.perf_counter() - start)
        timings[name] = 1e6 * best / samples
    print(json.dumps(timings))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baseline", type=Path,
                        help="Checkout (containing src/cvi) to compare against")
    parser.add_argument("--indices", nargs="+", choices=INDICES, default=INDICES)
    parser.add_argument("--clusters", type=int, default=40)
    parser.add_argument("--features", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2000)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    settings = (args.indices, args.clusters, args.features, args.warmup,
                args.samples, args.repeats, args.seed)
    if args.worker is not None:
        worker(args.worker, *settings)
        return

    roots = {"current": Path(__file__).resolve().parents[1]}
    if args.baseline is not None:
        roots["baseline"] = args.baseline.resolve()
    runs = {label: [] for label in roots}
    for _ in range(args.rounds):
        for label, root in roots.items():
            command = [
                sys.executable, "-m", "benchmarks.benchmark_streaming_add",
                "--worker", str(root), "--indices", *args.indices,
                "--clusters", str(args.clusters),
                "--features", str(args.features),
                "--warmup", str(args.warmup), "--samples", str(args.samples),
                "--repeats", str(args.repeats), "--seed", str(args.seed),
            ]
            output = subprocess.run(
                command, check=True, capture_output=True, text=True,
                cwd=roots["current"],
            ).stdout
            runs[label].append(json.loads(output))

    print(f"us per sample (median of {args.rounds} rounds), "
          f"K={args.clusters}, d={args.features}")
    header = f"{'index':6s}" + "".join(f"{label:>10s}" for label in roots)
    print(header + ("   change" if "baseline" in roots else ""))
    for name in args.indices:
        values = {label: statistics.median(run[name] for run in runs[label])
                  for label in roots}
        line = f"{name:6s}" + "".join(f"{values[label]:10.2f}" for label in roots)
        if "baseline" in roots:
            line += f"  {values['current'] / values['baseline'] - 1:+7.1%}"
        print(line)


if __name__ == "__main__":
    main()
