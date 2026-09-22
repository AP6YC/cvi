"""Generate the incremental-CVI animations embedded in the project README.

Run from any directory with::

    python3 docs/scripts/generate_incremental_cvi_gifs.py

The script intentionally uses the public ``cvi`` API for every score.  It
requires matplotlib and Pillow in addition to the project's core dependencies.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.datasets import load_iris, make_blobs
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import cvi  # noqa: E402


OUTPUT_DIR = PROJECT_ROOT / "docs" / "source" / "_static" / "readme"
FRAME_DURATION_MS = 65
END_PAUSE_MS = 5_000
BACKGROUND = "#f7f8fc"
TEXT = "#172033"
MUTED_TEXT = "#657089"
GRID = "#dfe3eb"
CLUSTER_COLORS = ("#2f80ed", "#f2994a", "#9b51e0")
METRIC_COLORS = ("#007f73", "#d1495b", "#6c5ce7")


@dataclass(frozen=True)
class Dataset:
    slug: str
    title: str
    score_data: np.ndarray
    plot_data: np.ndarray
    labels: np.ndarray
    class_names: tuple[str, ...]
    plot_note: str
    x_label: str
    y_label: str


@dataclass(frozen=True)
class Metric:
    constructor: type[cvi.CVI]
    title: str
    short_name: str
    optimality_symbol: str


METRICS = (
    Metric(cvi.CH, "Calinski–Harabasz", "CH", "↑"),
    Metric(cvi.DB, "Davies–Bouldin", "DB", "↓"),
    Metric(cvi.cSIL, "Centroid Silhouette", "cSIL", "↑"),
)


def build_datasets() -> tuple[Dataset, Dataset]:
    synthetic_data, synthetic_labels = make_blobs(
        n_samples=(50, 50, 50),
        centers=((-3.0, -2.2), (3.0, -1.6), (0.2, 3.0)),
        cluster_std=(0.65, 0.75, 0.70),
        random_state=42,
        shuffle=False,
    )
    synthetic = Dataset(
        slug="synthetic",
        title="Well-separated Gaussian blobs",
        score_data=synthetic_data,
        plot_data=synthetic_data,
        labels=synthetic_labels,
        class_names=("Cluster 1", "Cluster 2", "Cluster 3"),
        plot_note="CVI values and scatter use the same two features",
        x_label="Feature 1",
        y_label="Feature 2",
    )

    iris = load_iris()
    iris_data = StandardScaler().fit_transform(iris.data)
    iris_plot_data = PCA(n_components=2).fit_transform(iris_data)
    iris_dataset = Dataset(
        slug="iris",
        title="Iris dataset",
        score_data=iris_data,
        plot_data=iris_plot_data,
        labels=iris.target,
        class_names=tuple(name.title() for name in iris.target_names),
        plot_note="CVIs use all 4 standardized features; scatter uses PCA only",
        x_label="Principal component 1",
        y_label="Principal component 2",
    )
    return synthetic, iris_dataset


def stream_orders(labels: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "class-order": np.argsort(labels, kind="stable"),
        "shuffled": np.random.default_rng(1729).permutation(len(labels)),
    }


def calculate_scores(
    data: np.ndarray,
    labels: np.ndarray,
    order: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    histories: dict[str, np.ndarray] = {}
    batch_values: dict[str, float] = {}

    for metric in METRICS:
        batch_values[metric.short_name] = float(
            metric.constructor().get_cvi(data, labels)
        )
        incremental = metric.constructor()
        histories[metric.short_name] = np.asarray(
            [
                incremental.get_cvi(data[index], int(labels[index]))
                for index in order
            ],
            dtype=float,
        )
        if not np.isclose(
            histories[metric.short_name][-1],
            batch_values[metric.short_name],
        ):
            raise RuntimeError(
                f"Final incremental {metric.short_name} value does not match batch"
            )

    return histories, batch_values


def calculate_y_limits(
    mode_histories: dict[str, dict[str, np.ndarray]],
    batch_values: dict[str, float],
) -> dict[str, tuple[float, float]]:
    limits: dict[str, tuple[float, float]] = {}
    for metric in METRICS:
        name = metric.short_name
        values = [batch_values[name]]
        for histories in mode_histories.values():
            values.extend(histories[name][np.isfinite(histories[name])])
        minimum, maximum = float(np.min(values)), float(np.max(values))
        span = maximum - minimum
        padding = max(span * 0.12, abs(maximum) * 0.025, 0.025)
        lower, upper = minimum - padding, maximum + padding
        if name in {"CH", "DB"}:
            lower = max(0.0, lower)
        limits[name] = (lower, upper)
    return limits


def padded_limits(values: np.ndarray) -> tuple[float, float]:
    minimum, maximum = float(np.min(values)), float(np.max(values))
    padding = max((maximum - minimum) * 0.08, 0.35)
    return minimum - padding, maximum + padding


def render_animation(
    dataset: Dataset,
    mode: str,
    order: np.ndarray,
    histories: dict[str, np.ndarray],
    batch_values: dict[str, float],
    y_limits: dict[str, tuple[float, float]],
) -> Path:
    n_samples = len(dataset.labels)
    display_mode = "Class-ordered stream" if mode == "class-order" else "Shuffled stream"

    fig = plt.figure(figsize=(10.0, 7.4), dpi=90, facecolor=BACKGROUND)
    grid = fig.add_gridspec(
        2,
        3,
        height_ratios=(3.15, 1.65),
        left=0.075,
        right=0.975,
        top=0.865,
        bottom=0.105,
        hspace=0.34,
        wspace=0.26,
    )
    scatter_axis = fig.add_subplot(grid[0, :])
    metric_axes = [fig.add_subplot(grid[1, index]) for index in range(3)]

    fig.suptitle(
        f"Incremental CVIs · {dataset.title}",
        x=0.075,
        y=0.965,
        ha="left",
        color=TEXT,
        fontsize=18,
        fontweight="bold",
    )
    fig.text(
        0.075,
        0.914,
        f"{display_mode}  ·  {dataset.plot_note}",
        ha="left",
        color=MUTED_TEXT,
        fontsize=10.5,
    )
    fig.text(
        0.5,
        0.035,
        "Solid: incremental value     Dashed: batch value on all samples",
        ha="center",
        color=MUTED_TEXT,
        fontsize=9.5,
    )

    scatter_axis.set_facecolor(BACKGROUND)
    scatter_axis.set_xlim(*padded_limits(dataset.plot_data[:, 0]))
    scatter_axis.set_ylim(*padded_limits(dataset.plot_data[:, 1]))
    scatter_axis.set_xlabel(dataset.x_label, color=MUTED_TEXT, fontsize=9)
    scatter_axis.set_ylabel(dataset.y_label, color=MUTED_TEXT, fontsize=9)
    scatter_axis.tick_params(colors=MUTED_TEXT, labelsize=8)
    scatter_axis.grid(color=GRID, linewidth=0.8, alpha=0.7)
    scatter_axis.set_axisbelow(True)
    for spine in scatter_axis.spines.values():
        spine.set_color(GRID)

    observed_artists = []
    for class_index, (class_name, color) in enumerate(
        zip(dataset.class_names, CLUSTER_COLORS)
    ):
        class_mask = dataset.labels == class_index
        scatter_axis.scatter(
            dataset.plot_data[class_mask, 0],
            dataset.plot_data[class_mask, 1],
            s=42,
            color=color,
            alpha=0.12,
            edgecolors="none",
            zorder=1,
        )
        observed = scatter_axis.scatter(
            [],
            [],
            s=42,
            color=color,
            alpha=0.96,
            edgecolors="white",
            linewidths=0.45,
            label=class_name,
            zorder=3,
        )
        observed_artists.append(observed)

    current_artist = scatter_axis.scatter(
        [],
        [],
        s=115,
        facecolors="none",
        edgecolors=TEXT,
        linewidths=1.8,
        zorder=4,
    )
    scatter_axis.legend(
        loc="upper left",
        ncol=3,
        frameon=False,
        fontsize=9,
        labelcolor=TEXT,
        handletextpad=0.35,
        columnspacing=1.2,
    )
    counter = scatter_axis.text(
        0.985,
        0.965,
        "",
        transform=scatter_axis.transAxes,
        ha="right",
        va="top",
        color=TEXT,
        fontsize=10,
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": GRID,
            "alpha": 0.94,
        },
        zorder=5,
    )

    sample_numbers = np.arange(1, n_samples + 1)
    line_artists = []
    marker_artists = []
    for axis, metric, color in zip(metric_axes, METRICS, METRIC_COLORS):
        name = metric.short_name
        axis.set_facecolor(BACKGROUND)
        axis.set_xlim(1, n_samples)
        axis.set_ylim(*y_limits[name])
        axis.grid(color=GRID, linewidth=0.8, alpha=0.75)
        axis.set_axisbelow(True)
        axis.tick_params(colors=MUTED_TEXT, labelsize=7.5)
        axis.set_xlabel("Samples seen", color=MUTED_TEXT, fontsize=8)
        axis.set_title(
            f"{metric.title} ({name}) {metric.optimality_symbol}",
            loc="left",
            color=TEXT,
            fontsize=9.5,
            fontweight="bold",
            pad=7,
        )
        for spine in axis.spines.values():
            spine.set_color(GRID)
        axis.axhline(
            batch_values[name],
            color=color,
            linestyle=(0, (4, 3)),
            linewidth=1.55,
            alpha=0.72,
            zorder=1,
        )
        axis.text(
            0.985,
            0.925,
            f"batch {batch_values[name]:.3g}",
            transform=axis.transAxes,
            ha="right",
            va="top",
            color=color,
            fontsize=7.5,
            bbox={
                "boxstyle": "round,pad=0.24",
                "facecolor": BACKGROUND,
                "edgecolor": "none",
                "alpha": 0.88,
            },
            zorder=4,
        )
        line, = axis.plot(
            [],
            [],
            color=color,
            linewidth=2.2,
            solid_capstyle="round",
            zorder=2,
        )
        marker, = axis.plot(
            [],
            [],
            marker="o",
            markersize=5.5,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.6,
            linestyle="none",
            zorder=3,
        )
        line_artists.append(line)
        marker_artists.append(marker)

    frames: list[Image.Image] = []
    empty_offsets = np.empty((0, 2))
    for step in range(n_samples + 1):
        observed_indices = order[:step]
        for class_index, artist in enumerate(observed_artists):
            indices = observed_indices[
                dataset.labels[observed_indices] == class_index
            ]
            artist.set_offsets(
                dataset.plot_data[indices] if len(indices) else empty_offsets
            )

        if step:
            current_artist.set_offsets(dataset.plot_data[order[step - 1]][None, :])
        else:
            current_artist.set_offsets(empty_offsets)
        counter.set_text(f"Sample {step} / {n_samples}")

        for metric, line, marker in zip(METRICS, line_artists, marker_artists):
            values = histories[metric.short_name]
            line.set_data(sample_numbers[:step], values[:step])
            if step and np.isfinite(values[step - 1]):
                marker.set_data([step], [values[step - 1]])
            else:
                marker.set_data([], [])

        fig.canvas.draw()
        frame = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
        frames.append(frame)

    plt.close(fig)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"incremental-cvi-{dataset.slug}-{mode}.gif"
    durations = [FRAME_DURATION_MS] * len(frames)
    durations[-1] = END_PAUSE_MS
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        disposal=2,
        optimize=True,
    )
    return output_path


def main() -> None:
    for dataset in build_datasets():
        orders = stream_orders(dataset.labels)
        mode_histories: dict[str, dict[str, np.ndarray]] = {}
        batch_values: dict[str, float] | None = None
        for mode, order in orders.items():
            histories, local_batch_values = calculate_scores(
                dataset.score_data,
                dataset.labels,
                order,
            )
            mode_histories[mode] = histories
            if batch_values is None:
                batch_values = local_batch_values
            elif batch_values != local_batch_values:
                raise RuntimeError("Batch values changed between stream orderings")

        assert batch_values is not None
        y_limits = calculate_y_limits(mode_histories, batch_values)
        for mode, order in orders.items():
            output_path = render_animation(
                dataset,
                mode,
                order,
                mode_histories[mode],
                batch_values,
                y_limits,
            )
            print(output_path.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
