"""Shared deterministic datasets for CVI benchmarks."""

from __future__ import annotations

import numpy as np


SEED = 318


def make_balanced_gaussian_stream(
    samples: int,
    *,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Return an interleaved stream from ten balanced 2D Gaussian blobs."""
    if samples < 2:
        raise ValueError("samples must be at least 2")

    rng = np.random.default_rng(seed)
    centers = np.array([
        (x, y)
        for y in (0.28, 0.72)
        for x in (0.10, 0.30, 0.50, 0.70, 0.90)
    ])
    labels = np.arange(samples, dtype=int) % len(centers)
    points = centers[labels] + rng.normal(0.0, 0.022, (samples, 2))
    if not np.all((points >= 0.0) & (points <= 1.0)):
        raise ValueError("Synthetic points escaped the Fuzzy ART input domain")
    return points, labels
