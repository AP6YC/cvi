"""Time all NumPy CVIs on nested, balanced 2D Gaussian-cluster datasets.

Run from the repository root::

    python3 -m benchmarks.benchmark_batch_incremental

Requires matplotlib and the optional ``artlib`` dependency for incremental CONN.
The output directory contains a 12-panel PNG, raw trial CSV, and run metadata.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from importlib.metadata import version
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score

from src import cvi


SAMPLE_SIZES = (100, 250, 500, 1_000, 2_000, 4_000, 6_000, 8_000, 10_000)
REPEATS = 5
SEED = 318
INDEX_NAMES = tuple(module.__name__.split(".")[-1] for module in cvi.MODULES)
SKLEARN_SCORES = {
    "CH": calinski_harabasz_score,
    "DB": davies_bouldin_score,
}
# Two-sided 95% Student t critical value with four degrees of freedom.
T_CRITICAL_95_DF4 = 2.7764451051977987
METHOD_COLORS = {
    "incremental": "#2563eb",
    "batch": "#d97706",
    "scikit-learn": "#16a34a",
}


def make_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Interleave ten Gaussian blobs so every prefix is cluster-balanced."""
    rng = np.random.default_rng(SEED)
    centers = np.array([(x, y) for y in (0.28, 0.72)
                        for x in (0.10, 0.30, 0.50, 0.70, 0.90)])
    labels = np.tile(np.arange(10), SAMPLE_SIZES[-1] // 10)
    points = centers[labels] + rng.normal(0.0, 0.022, (len(labels), 2))
    # Fuzzy ART requires [0, 1]. A single, fixed affine domain is used by all
    # methods and sizes; no per-prefix preprocessing enters a timing.
    if not np.all((points >= 0.0) & (points <= 1.0)):
        raise ValueError("Synthetic points escaped the Fuzzy ART input domain")
    return points, labels


def compute(name: str, method: str, points: np.ndarray,
            labels: np.ndarray) -> float:
    if method == "scikit-learn":
        return float(SKLEARN_SCORES[name](points, labels))

    options = {"model_type": "Fuzzy", "normalize_batch": False} if name == "CONN" else {}
    index = getattr(cvi, name)(**options)
    if method == "batch":
        return float(index.get_cvi(points, labels))
    value = float("nan")
    for point, label in zip(points, labels):
        value = index.get_cvi(point, int(label))
    return float(value)


def methods_for(name: str) -> tuple[str, ...]:
    return ("incremental", "batch", "scikit-learn") if name in SKLEARN_SCORES else (
        "incremental", "batch")


def measure(points: np.ndarray, labels: np.ndarray) -> list[dict]:
    rows = []
    for size in SAMPLE_SIZES:
        prefix = points[:size]
        prefix_labels = labels[:size]
        for name in INDEX_NAMES:
            methods = methods_for(name)
            # Warm imports and lazy setup outside the measured trials.
            if size == SAMPLE_SIZES[0]:
                for method in methods:
                    compute(name, method, prefix, prefix_labels)
            for repeat in range(1, REPEATS + 1):
                # Rotate method order to distribute short-term machine drift.
                ordered = methods[repeat % len(methods):] + methods[:repeat % len(methods)]
                for method in ordered:
                    start = perf_counter()
                    score = compute(name, method, prefix, prefix_labels)
                    elapsed = perf_counter() - start
                    rows.append({"index": name, "method": method,
                                 "samples": size, "repeat": repeat,
                                 "seconds": elapsed, "score": score})
        print(f"Finished {size:,} samples", flush=True)
    return rows


def geometric_mean_ci(seconds: list[float]) -> tuple[float, float, float]:
    """Geometric mean and back-transformed 95% t interval of five log times."""
    logs = np.log(seconds)
    center = float(np.mean(logs))
    half_width = T_CRITICAL_95_DF4 * float(np.std(logs, ddof=1)) / np.sqrt(REPEATS)
    return (float(np.exp(center)), float(np.exp(center - half_width)),
            float(np.exp(center + half_width)))


def plot(points: np.ndarray, labels: np.ndarray, rows: list[dict], path: Path) -> None:
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12,
                         "axes.labelsize": 9, "savefig.facecolor": "white"})
    fig, axes = plt.subplots(3, 4, figsize=(21, 14))
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.10, top=0.91,
                        wspace=0.23, hspace=0.37)
    axes = axes.flat
    fig.suptitle("Batch and incremental CVI timing · 10 balanced 2D Gaussian clusters",
                 fontsize=19, weight="bold", y=0.975)

    for panel, name in enumerate(INDEX_NAMES):
        ax = axes[panel]
        for method in methods_for(name):
            summary = [geometric_mean_ci([
                row["seconds"] for row in rows
                if row["index"] == name and row["method"] == method
                and row["samples"] == size
            ]) for size in SAMPLE_SIZES]
            mean, lower, upper = np.asarray(summary).T * 1_000
            color = METHOD_COLORS[method]
            label = ("scikit-learn (Euclidean)"
                     if name == "DB" and method == "scikit-learn" else method)
            ax.plot(SAMPLE_SIZES, mean, color=color, marker="o", markersize=3,
                    linewidth=1.8, label=label)
            ax.fill_between(SAMPLE_SIZES, lower, upper, color=color, alpha=0.14,
                            linewidth=0)
        title = "DB (CVI: squared distances)" if name == "DB" else name
        ax.set(title=title, xlabel="Samples", ylabel="Total time (ms)", yscale="log")
        ax.grid(alpha=0.25, which="both")
        ax.legend(fontsize=7, loc="best", frameon=False)

    scatter = axes[10]
    scatter.scatter(points[:, 0], points[:, 1], c=labels, cmap="tab10", s=2,
                    alpha=0.45, linewidths=0, rasterized=True)
    scatter.set(title="Dataset · 10,000 samples", xlabel="Feature 1", ylabel="Feature 2",
                xlim=(0, 1), ylim=(0, 1))
    scatter.set_aspect("equal", adjustable="box")
    scatter.grid(alpha=0.2)

    speedup = axes[11]
    ratios = []
    for name in INDEX_NAMES:
        at_max = {method: geometric_mean_ci([
            row["seconds"] for row in rows
            if row["index"] == name and row["method"] == method
            and row["samples"] == SAMPLE_SIZES[-1]
        ])[0] for method in ("batch", "incremental")}
        ratios.append(at_max["batch"] / at_max["incremental"])
    y = np.arange(len(INDEX_NAMES))
    speedup.scatter(ratios, y, s=64, zorder=3,
                    c=["#16a34a" if ratio > 1 else "#d97706" for ratio in ratios])
    speedup.axvline(1, color="#374151", linewidth=1)
    speedup.set(yticks=y, yticklabels=INDEX_NAMES,
                title="Batch / incremental · 10,000 samples", xscale="log",
                xlim=(0.003, 2), xlabel="Time ratio (>1: incremental faster)")
    speedup.invert_yaxis()
    speedup.grid(axis="x", alpha=0.25)

    fig.text(0.5, 0.025,
             "Lines: geometric mean of 5 complete computations; shading: 95% t interval on log time. "
             "Object construction included.\n"
             "CONN uses Fuzzy ART. DB compares different definitions: CVI squared distances and scikit-learn Euclidean distances.",
             ha="center", fontsize=9, color="#4b5563")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("benchmarks/results/batch_incremental"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    points, labels = make_dataset()
    rows = measure(points, labels)
    csv_path = args.output_dir / "timings.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=("index", "method", "samples", "repeat",
                                                   "seconds", "score"))
        writer.writeheader()
        writer.writerows(rows)

    figure_path = args.output_dir / "timing_figure.png"
    plot(points, labels, rows, figure_path)
    metadata = {
        "hypothesis": "Compare total computation time for batch and per-sample CVI APIs.",
        "sample_sizes": SAMPLE_SIZES, "clusters": 10, "features": 2,
        "dataset": "Balanced interleaved Gaussian blobs on a fixed [0, 1] domain; std=0.022",
        "seed": SEED, "repeats": REPEATS, "backend": "numpy",
        "conn_parameters": {"model_type": "Fuzzy", "normalize_batch": False},
        "environment": {"OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS")},
        "timing": "Fresh object and full score computation per trial; no data generation",
        "interval": "Geometric mean and 95% Student t interval of log seconds (df=4)",
        "limitations": "Five trials give descriptive intervals; machine load and timer noise affect them. DB compares different definitions: CVI squared distances and scikit-learn Euclidean distances.",
        "versions": {"python": sys.version.split()[0], "platform": platform.platform(),
                     "numpy": np.__version__, "scikit_learn": sklearn.__version__,
                     "matplotlib": matplotlib.__version__, "artlib": version("artlib"),
                     "cvi": cvi.__version__},
        "files": {"raw_trials": csv_path.name, "figure": figure_path.name},
    }
    metadata["source_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    metadata["git_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip()
    (args.output_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    index_path = args.output_dir.parent / "index.json"
    run_path = str(args.output_dir.relative_to(args.output_dir.parent) / "run.json")
    run_index = json.loads(index_path.read_text()) if index_path.exists() else {"runs": []}
    if run_path not in run_index["runs"]:
        run_index["runs"].append(run_path)
    index_path.write_text(json.dumps(run_index, indent=2) + "\n")
    print(f"Saved {len(rows)} trials to {csv_path} and {figure_path}")


if __name__ == "__main__":
    main()
