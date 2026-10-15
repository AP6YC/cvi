"""Benchmark an existing incremental CVI against repeated batch recomputation.

The stream is deterministic and cluster-balanced. At every sample index, the
incremental object receives only the new sample; the batch object is freshly
constructed and scores the entire prefix. The public incremental API evaluates
a score after every sample.

Run a guarded estimate before a larger experiment::

    python -m benchmarks.benchmark_growing_stream \
        --estimate-only --samples 10000

The normal command runs only when its conservative pilot estimate fits within
``--max-estimated-seconds``. Use ``--force`` only after reviewing the estimate.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
import platform
from pathlib import Path
from statistics import mean, median
import subprocess
import sys
from time import perf_counter
from typing import Callable, Optional, Sequence
import warnings

import matplotlib.pyplot as plt
import numpy as np
import sklearn

from src import cvi
from ._datasets import SEED, make_balanced_gaussian_stream


INDEX_NAMES = tuple(module.__name__.split(".")[-1] for module in cvi.MODULES)
DEFAULT_CONN_BATCH_MODELS = ("MiniBatchKMeans",)
CSV_FIELDS = (
    "index",
    "method",
    "prototype_model",
    "samples",
    "samples_added",
    "repeat",
    "construction_seconds",
    "update_score_seconds",
    "checkpoint_seconds",
    "cumulative_seconds",
    "score",
    "comparison_expected",
    "score_close",
    "score_abs_error",
    "retained_input_bytes",
)
SCORE_RTOL = 1e-9
SCORE_ATOL = 1e-12
METHOD_COLORS = {"incremental": "#2563eb", "batch": "#d97706"}

plt.switch_backend("Agg")


def checkpoints(samples: int) -> tuple[int, ...]:
    """Return every valid batch checkpoint through the final sample."""
    if samples < 2:
        raise ValueError("samples must be at least 2")
    return tuple(range(2, samples + 1))


def _models_for(
    name: str,
    conn_batch_models: Sequence[str],
) -> tuple[str, ...]:
    return tuple(conn_batch_models) if name == "CONN" else ("",)


def _new_index(name: str, prototype_model: str = ""):
    options = {}
    if name == "CONN":
        options = {
            "model_type": prototype_model or "Fuzzy",
            "normalize_batch": False,
        }
        if prototype_model in {"KMeans", "MiniBatchKMeans"}:
            options["kmeans_kwargs"] = {"random_state": SEED}
    return getattr(cvi, name)(**options)


def _timed_incremental(
    name: str,
    points: np.ndarray,
    labels: np.ndarray,
    sample_checkpoints: Sequence[int],
    repeat: int,
    timer: Callable[[], float],
) -> list[dict]:
    start = timer()
    index = _new_index(name, "Fuzzy" if name == "CONN" else "")
    construction = timer() - start
    cumulative = 0.0
    previous = 0
    rows = []
    for sample_count in sample_checkpoints:
        start = timer()
        score = float("nan")
        for point, label in zip(
            points[previous:sample_count], labels[previous:sample_count]
        ):
            score = float(index.get_cvi(point, int(label)))
        update_score = timer() - start
        checkpoint_construction = construction if previous == 0 else 0.0
        elapsed = checkpoint_construction + update_score
        cumulative += elapsed
        rows.append({
            "index": name,
            "method": "incremental",
            "prototype_model": "Fuzzy" if name == "CONN" else "",
            "samples": sample_count,
            "samples_added": sample_count - previous,
            "repeat": repeat,
            "construction_seconds": checkpoint_construction,
            "update_score_seconds": update_score,
            "checkpoint_seconds": elapsed,
            "cumulative_seconds": cumulative,
            "score": score,
            "comparison_expected": "",
            "score_close": "",
            "score_abs_error": "",
            "retained_input_bytes": 0,
        })
        previous = sample_count
    return rows


def _timed_batch(
    name: str,
    prototype_model: str,
    points: np.ndarray,
    labels: np.ndarray,
    sample_checkpoints: Sequence[int],
    repeat: int,
    timer: Callable[[], float],
) -> list[dict]:
    cumulative = 0.0
    rows = []
    for sample_count in sample_checkpoints:
        start = timer()
        index = _new_index(name, prototype_model)
        construction = timer() - start
        start = timer()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            score = float(index.get_cvi(
                points[:sample_count], labels[:sample_count]
            ))
        update_score = timer() - start
        elapsed = construction + update_score
        cumulative += elapsed
        rows.append({
            "index": name,
            "method": "batch",
            "prototype_model": prototype_model,
            "samples": sample_count,
            "samples_added": sample_count,
            "repeat": repeat,
            "construction_seconds": construction,
            "update_score_seconds": update_score,
            "checkpoint_seconds": elapsed,
            "cumulative_seconds": cumulative,
            "score": score,
            "comparison_expected": (
                name != "CONN" or prototype_model == "Fuzzy"
            ),
            "score_close": "",
            "score_abs_error": "",
            "retained_input_bytes": (
                points[:sample_count].nbytes + labels[:sample_count].nbytes
            ),
        })
    return rows


def _annotate_score_comparisons(rows: list[dict]) -> None:
    incremental = {
        (
            row["index"],
            row["repeat"],
            row["samples"],
        ): row["score"]
        for row in rows
        if row["method"] == "incremental"
    }
    for row in rows:
        if row["method"] != "batch" or not row["comparison_expected"]:
            continue
        expected = incremental[
            (
                row["index"],
                row["repeat"],
                row["samples"],
            )
        ]
        observed = row["score"]
        row["score_close"] = bool(np.isclose(
            observed,
            expected,
            rtol=SCORE_RTOL,
            atol=SCORE_ATOL,
            equal_nan=True,
        ))
        if np.isnan(observed) and np.isnan(expected):
            row["score_abs_error"] = 0.0
        else:
            row["score_abs_error"] = abs(observed - expected)


def measure(
    points: np.ndarray,
    labels: np.ndarray,
    *,
    indices: Sequence[str],
    repeats: int,
    conn_batch_models: Sequence[str] = DEFAULT_CONN_BATCH_MODELS,
    timer: Callable[[], float] = perf_counter,
) -> list[dict]:
    """Collect raw checkpoint timings for every requested case."""
    rows = []
    sample_checkpoints = checkpoints(len(points))
    for name in indices:
        for repeat in range(1, repeats + 1):
            batch_models = _models_for(name, conn_batch_models)
            operations = [
                ("incremental", "Fuzzy" if name == "CONN" else "")
            ] + [("batch", model) for model in batch_models]
            if repeat % 2 == 0:
                operations.reverse()
            for method, model in operations:
                if method == "incremental":
                    rows.extend(_timed_incremental(
                        name,
                        points,
                        labels,
                        sample_checkpoints,
                        repeat,
                        timer,
                    ))
                else:
                    rows.extend(_timed_batch(
                        name,
                        model,
                        points,
                        labels,
                        sample_checkpoints,
                        repeat,
                        timer,
                    ))
        print(f"Finished {name} through {len(points):,} samples", flush=True)
    _annotate_score_comparisons(rows)
    return rows


def _mean_at(
    rows: Sequence[dict],
    *,
    name: str,
    method: str,
    model: str,
    sample_count: int,
    field: str,
) -> float:
    values = [
        float(row[field])
        for row in rows
        if row["index"] == name
        and row["method"] == method
        and row["prototype_model"] == model
        and row["samples"] == sample_count
    ]
    return mean(values)


def sustained_break_even(
    sample_counts: Sequence[int],
    incremental_seconds: Sequence[float],
    batch_seconds: Sequence[float],
) -> Optional[int]:
    """Return the first checkpoint after which batch stays no faster."""
    for position, sample_count in enumerate(sample_counts):
        if all(
            batch >= incremental
            for batch, incremental in zip(
                batch_seconds[position:], incremental_seconds[position:]
            )
        ):
            return sample_count
    return None


def summarize(
    rows: Sequence[dict],
    conn_batch_models: Sequence[str] = DEFAULT_CONN_BATCH_MODELS,
) -> dict:
    """Summarize cumulative time, latency, break-even, and correctness."""
    names = tuple(dict.fromkeys(row["index"] for row in rows))
    cases = []
    for name in names:
        incremental_model = "Fuzzy" if name == "CONN" else ""
        sample_counts = sorted({
            row["samples"]
            for row in rows
            if row["index"] == name
        })
        incremental_cumulative = [
            _mean_at(
                rows,
                name=name,
                method="incremental",
                model=incremental_model,
                sample_count=sample_count,
                field="cumulative_seconds",
            )
            for sample_count in sample_counts
        ]
        incremental_update_latency = mean([
            row["update_score_seconds"] / row["samples_added"]
            for row in rows
            if row["index"] == name
            and row["method"] == "incremental"
        ])
        for model in _models_for(name, conn_batch_models):
            batch_cumulative = [
                _mean_at(
                    rows,
                    name=name,
                    method="batch",
                    model=model,
                    sample_count=sample_count,
                    field="cumulative_seconds",
                )
                for sample_count in sample_counts
            ]
            batch_latency = mean([
                row["checkpoint_seconds"]
                for row in rows
                if row["index"] == name
                and row["method"] == "batch"
                and row["prototype_model"] == model
            ])
            cases.append({
                "index": name,
                "batch_prototype_model": model or None,
                "final_samples": sample_counts[-1],
                "incremental_cumulative_seconds_mean": (
                    incremental_cumulative[-1]
                ),
                "batch_cumulative_seconds_mean": batch_cumulative[-1],
                "batch_over_incremental_final": (
                    batch_cumulative[-1] / incremental_cumulative[-1]
                ),
                "incremental_update_seconds_mean": incremental_update_latency,
                "batch_checkpoint_seconds_mean": batch_latency,
                "break_even_samples": sustained_break_even(
                    sample_counts,
                    incremental_cumulative,
                    batch_cumulative,
                ),
                "scores_expected_equal": name != "CONN" or model == "Fuzzy",
            })

    comparable = [
        row
        for row in rows
        if row["method"] == "batch" and row["comparison_expected"]
    ]
    return {
        "cases": cases,
        "score_comparisons": {
            "total": len(comparable),
            "passed": sum(row["score_close"] is True for row in comparable),
            "failed": sum(row["score_close"] is False for row in comparable),
            "rtol": SCORE_RTOL,
            "atol": SCORE_ATOL,
            "equal_nan": True,
        },
    }


def _plot_metric(
    rows: Sequence[dict],
    points: np.ndarray,
    labels: np.ndarray,
    *,
    field: str,
    title: str,
    ylabel: str,
    path: Path,
) -> None:
    """Plot arithmetic mean and trial range at every sample index."""
    names = tuple(dict.fromkeys(row["index"] for row in rows))
    series = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = (row["index"], row["method"], row["prototype_model"])
        series[key][row["samples"]].append(float(row[field]) * 1_000)
    fig, axes = plt.subplots(3, 4, figsize=(21, 14))
    fig.subplots_adjust(
        left=0.06,
        right=0.985,
        bottom=0.08,
        top=0.91,
        wspace=0.25,
        hspace=0.34,
    )
    fig.suptitle(title, fontsize=19, weight="bold", y=0.97)
    flat_axes = axes.flat
    for axis, name in zip(flat_axes, names):
        methods = [
            ("incremental", "Fuzzy" if name == "CONN" else ""),
            *[("batch", model) for model in sorted({
                row["prototype_model"]
                for row in rows
                if row["index"] == name and row["method"] == "batch"
            })],
        ]
        for method, model in methods:
            values_by_sample = series[(name, method, model)]
            sample_counts = sorted(values_by_sample)
            trials = np.asarray([
                values_by_sample[sample_count]
                for sample_count in sample_counts
            ])
            average = np.mean(trials, axis=1)
            color = METHOD_COLORS[method]
            label = method
            if name == "CONN":
                label = f"{method} ({model})"
            axis.plot(
                sample_counts,
                average,
                color=color,
                linewidth=1.6,
                label=label,
            )
            axis.fill_between(
                sample_counts,
                np.min(trials, axis=1),
                np.max(trials, axis=1),
                color=color,
                alpha=0.14,
                linewidth=0,
            )
        axis.set(
            title=name,
            xlabel="Samples seen",
            ylabel=ylabel,
            yscale="log",
        )
        axis.grid(alpha=0.25, which="both")
        axis.legend(fontsize=7, loc="best", frameon=False)

    dataset_axis = flat_axes[len(names)]
    dataset_axis.scatter(
        points[:, 0],
        points[:, 1],
        c=labels,
        cmap="tab10",
        s=5,
        alpha=0.55,
        linewidths=0,
        rasterized=True,
    )
    dataset_axis.set(
        title=f"Dataset · {len(points):,} samples",
        xlabel="Feature 1",
        ylabel="Feature 2",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    dataset_axis.set_aspect("equal", adjustable="box")
    dataset_axis.grid(alpha=0.2)

    speedup_axis = flat_axes[len(names) + 1]
    final_sample = max(row["samples"] for row in rows)
    ratios = []
    for name in names:
        incremental_model = "Fuzzy" if name == "CONN" else ""
        batch_model = next(
            row["prototype_model"]
            for row in rows
            if row["index"] == name and row["method"] == "batch"
        )
        incremental = mean(
            series[(name, "incremental", incremental_model)][final_sample]
        )
        batch = mean(series[(name, "batch", batch_model)][final_sample])
        ratios.append(batch / incremental)
    positions = np.arange(len(names))
    speedup_axis.scatter(
        ratios,
        positions,
        s=64,
        zorder=3,
        c=["#16a34a" if ratio > 1 else "#d97706" for ratio in ratios],
    )
    speedup_axis.axvline(1, color="#374151", linewidth=1)
    speedup_axis.set(
        yticks=positions,
        yticklabels=names,
        title=f"Batch / incremental · {final_sample:,} samples",
        xscale="log",
        xlim=(min(1, min(ratios)) * 0.7, max(1, max(ratios)) * 1.4),
        xlabel="Time ratio (>1: incremental faster)",
    )
    speedup_axis.invert_yaxis()
    speedup_axis.grid(axis="x", alpha=0.25)
    fig.text(
        0.5,
        0.025,
        "Lines: arithmetic mean across repeated trials; shading: trial range. "
        "Object construction is included.",
        ha="center",
        fontsize=9,
        color="#4b5563",
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_results(
    rows: Sequence[dict],
    points: np.ndarray,
    labels: np.ndarray,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Create mean per-sample latency and cumulative-time figures."""
    latency_path = output_dir / "mean_latency.png"
    cumulative_path = output_dir / "mean_cumulative.png"
    _plot_metric(
        rows,
        points,
        labels,
        field="checkpoint_seconds",
        title="Growing stream · mean latency at every sample index",
        ylabel="Checkpoint latency (ms)",
        path=latency_path,
    )
    _plot_metric(
        rows,
        points,
        labels,
        field="cumulative_seconds",
        title="Growing stream · mean cumulative wall time",
        ylabel="Cumulative time (ms)",
        path=cumulative_path,
    )
    return latency_path, cumulative_path


def estimate_runtime(
    points: np.ndarray,
    labels: np.ndarray,
    *,
    indices: Sequence[str],
    repeats: int,
    conn_batch_models: Sequence[str],
    pilot_samples: int,
    safety_factor: float,
    timer: Callable[[], float] = perf_counter,
) -> dict:
    """Time bounded pilot work and conservatively project the requested run."""
    pilot_count = min(len(points), max(20, pilot_samples))
    pilot_points = points[:pilot_count]
    pilot_labels = labels[:pilot_count]
    estimates = []
    total = 0.0
    for name in indices:
        incremental_trials = []
        for _ in range(3):
            start = timer()
            index = _new_index(name, "Fuzzy" if name == "CONN" else "")
            for point, label in zip(pilot_points, pilot_labels):
                index.get_cvi(point, int(label))
            incremental_trials.append(timer() - start)
        incremental_pilot = median(incremental_trials)
        incremental_projected = (
            incremental_pilot
            * len(points)
            / pilot_count
            * repeats
        )
        estimates.append({
            "index": name,
            "method": "incremental",
            "prototype_model": "Fuzzy" if name == "CONN" else None,
            "pilot_seconds": incremental_pilot,
            "projected_seconds": incremental_projected * safety_factor,
        })
        total += incremental_projected

        for model in _models_for(name, conn_batch_models):
            batch_trials = []
            for _ in range(3):
                start = timer()
                index = _new_index(name, model)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    index.get_cvi(pilot_points, pilot_labels)
                batch_trials.append(timer() - start)
            batch_pilot = median(batch_trials)
            scale = sum(
                max(sample_count / pilot_count, 1.0)
                for sample_count in checkpoints(len(points))
            )
            batch_projected = batch_pilot * scale * repeats
            estimates.append({
                "index": name,
                "method": "batch",
                "prototype_model": model or None,
                "pilot_seconds": batch_pilot,
                "projected_seconds": batch_projected * safety_factor,
            })
            total += batch_projected

    return {
        "pilot_samples": pilot_count,
        "pilot_trials": 3,
        "assumed_batch_scaling": "linear in prefix length",
        "safety_factor": safety_factor,
        "projected_seconds": total * safety_factor,
        "cases": estimates,
    }


def _package_version(name: str) -> Optional[str]:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _environment() -> dict:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "matplotlib": _package_version("matplotlib"),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "artlib": _package_version("artlib"),
        "cvi": cvi.__version__,
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
    }


def _git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _update_run_index(output_dir: Path) -> None:
    index_path = output_dir.parent / "index.json"
    run_path = str(output_dir.relative_to(output_dir.parent) / "run.json")
    run_index = (
        json.loads(index_path.read_text())
        if index_path.exists()
        else {"runs": []}
    )
    if run_path not in run_index["runs"]:
        run_index["runs"].append(run_path)
    index_path.write_text(json.dumps(run_index, indent=2) + "\n")


def write_estimate(
    output_dir: Path,
    estimate: dict,
    configuration: dict,
    *,
    status: str,
) -> None:
    """Persist an estimate-only or budget-skipped experiment record."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "estimate.json").write_text(
        json.dumps(estimate, indent=2) + "\n"
    )
    run = {
        "status": status,
        "hypothesis": (
            "Maintaining incremental CVI state becomes faster than repeatedly "
            "recomputing batch scores as a labeled stream grows."
        ),
        "configuration": configuration,
        "environment": _environment(),
        "git_commit": _git_commit(),
        "source_script_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "files": {"estimate": "estimate.json"},
    }
    (output_dir / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    _update_run_index(output_dir)


def write_results(
    output_dir: Path,
    rows: Sequence[dict],
    points: np.ndarray,
    labels: np.ndarray,
    summary: dict,
    estimate: dict,
    configuration: dict,
) -> None:
    """Persist raw timings, summaries, configuration, and environment."""
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "timings.csv.gz"
    with gzip.open(raw_path, "wt", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    latency_path, cumulative_path = plot_results(
        rows, points, labels, output_dir
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    (output_dir / "estimate.json").write_text(
        json.dumps(estimate, indent=2) + "\n"
    )
    run = {
        "status": "complete",
        "hypothesis": (
            "Maintaining incremental CVI state becomes faster than repeatedly "
            "recomputing batch scores as a labeled stream grows."
        ),
        "configuration": configuration,
        "dataset": (
            "Balanced interleaved ten-cluster 2D Gaussian stream on a fixed "
            "[0, 1] domain; standard deviation 0.022."
        ),
        "timing": {
            "incremental": (
                "The first checkpoint includes construction and the first two "
                "updates; each later checkpoint adds and scores one sample."
            ),
            "batch": (
                "Every checkpoint includes fresh construction and scoring of "
                "all "
                "samples seen so far."
            ),
            "excluded": "Dataset generation and result serialization.",
        },
        "retained_data": (
            "retained_input_bytes reports logical raw point and label bytes a "
            "batch consumer must retain. The benchmark process pre-generates "
            "the same full stream for both methods; object state memory is "
            "not measured."
        ),
        "conn": (
            "Incremental CONN uses Fuzzy ART. The default batch CONN uses "
            "MiniBatchKMeans prototypes, so its scores are deliberately "
            "excluded from equality checks. Fuzzy batch replay remains an "
            "explicit opt-in comparison through --conn-batch-models."
        ),
        "limitations": (
            "Pilot projections assume linear batch cost and use a safety "
            "factor; they are guards, not runtime guarantees. Wall times "
            "depend on load and hardware. Arithmetic means and trial ranges "
            "are descriptive, not confidence intervals."
        ),
        "environment": _environment(),
        "git_commit": _git_commit(),
        "source_script_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "files": {
            "raw_trials": raw_path.name,
            "summary": "summary.json",
            "estimate": "estimate.json",
            "mean_latency_figure": latency_path.name,
            "mean_cumulative_figure": cumulative_path.name,
        },
    }
    (output_dir / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    _update_run_index(output_dir)


def _configuration(args: argparse.Namespace) -> dict:
    return {
        "samples": args.samples,
        "indices": list(args.indices),
        "scoring": "every sample index from 2 through samples",
        "repeats": args.repeats,
        "seed": SEED,
        "conn_batch_models": list(args.conn_batch_models),
        "pilot_samples": args.pilot_samples,
        "estimate_safety_factor": args.estimate_safety_factor,
        "max_estimated_seconds": args.max_estimated_seconds,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=1_000)
    parser.add_argument("--indices", nargs="+", choices=INDEX_NAMES,
                        default=INDEX_NAMES)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--conn-batch-models",
        nargs="+",
        choices=("Fuzzy", "KMeans", "MiniBatchKMeans"),
        default=DEFAULT_CONN_BATCH_MODELS,
    )
    parser.add_argument("--pilot-samples", type=int, default=100)
    parser.add_argument("--estimate-safety-factor", type=float, default=2.0)
    parser.add_argument("--max-estimated-seconds", type=float, default=1_800.0)
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run even when the pilot projection exceeds the runtime budget",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/results/growing_stream-n1000"),
    )
    args = parser.parse_args()
    if args.samples < 2 or args.repeats < 1 or args.pilot_samples < 2:
        parser.error(
            "samples >= 2, repeats >= 1, and pilot-samples >= 2 required"
        )
    if args.estimate_safety_factor < 1 or args.max_estimated_seconds <= 0:
        parser.error(
            "estimate safety factor >= 1 and positive budget required"
        )
    return args


def main() -> None:
    """Estimate, guard, execute, validate, and persist the benchmark."""
    args = _parse_args()
    points, labels = make_balanced_gaussian_stream(args.samples, seed=SEED)
    configuration = _configuration(args)
    estimate = estimate_runtime(
        points,
        labels,
        indices=args.indices,
        repeats=args.repeats,
        conn_batch_models=args.conn_batch_models,
        pilot_samples=args.pilot_samples,
        safety_factor=args.estimate_safety_factor,
    )
    print(json.dumps(estimate, indent=2), flush=True)
    if args.estimate_only:
        write_estimate(
            args.output_dir,
            estimate,
            configuration,
            status="estimate-only",
        )
        print(f"Saved guarded estimate to {args.output_dir}")
        return
    if (
        estimate["projected_seconds"] > args.max_estimated_seconds
        and not args.force
    ):
        write_estimate(
            args.output_dir,
            estimate,
            configuration,
            status="skipped-estimate-over-budget",
        )
        raise SystemExit(
            "Projected runtime exceeds --max-estimated-seconds; reduce the "
            "stream/repeats, raise the budget, or pass --force after "
            "reviewing estimate.json."
        )

    rows = measure(
        points,
        labels,
        indices=args.indices,
        repeats=args.repeats,
        conn_batch_models=args.conn_batch_models,
    )
    summary = summarize(rows, args.conn_batch_models)
    write_results(
        args.output_dir,
        rows,
        points,
        labels,
        summary,
        estimate,
        configuration,
    )
    comparisons = summary["score_comparisons"]
    if comparisons["failed"]:
        raise AssertionError(
            f"{comparisons['failed']} of {comparisons['total']} expected "
            "score "
            "comparisons failed"
        )
    print(
        f"Saved {len(rows):,} raw checkpoint timings and "
        f"{len(summary['cases'])} summaries to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
