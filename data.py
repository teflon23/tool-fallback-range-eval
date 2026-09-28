"""data.py: synthetic tool-fallback dataset generator."""
from __future__ import annotations

import numpy as np

# 4 tools, each with a 3-D parameter range (min, max) per dimension.
# Ranges are chosen so that ~5-15% of tasks are truly incompletable
# (no tool's range contains the task's parameter vector).
TOOL_RANGES: list[tuple[np.ndarray, np.ndarray]] = [
    # Tool 0: primary range
    (np.array([0.0, 0.0, 0.0]), np.array([10.0, 10.0, 10.0])),
    # Tool 1: overlaps with tool 0 in p1, p2 but narrower in p3
    (np.array([2.0, 2.0, 0.0]), np.array([12.0, 12.0, 6.0])),
    # Tool 2: overlaps with tool 0 in p1, p3 but narrower in p2
    (np.array([0.0, 4.0, 0.0]), np.array([14.0, 8.0, 12.0])),
    # Tool 3: mostly disjoint; small overlap region
    (np.array([8.0, 8.0, 8.0]), np.array([16.0, 16.0, 16.0])),
]

# Fixed ordered fallback list (indices into TOOL_RANGES), excluding the primary.
# For a given primary tool i, the fallback order is the other tools in a
# deterministic order. We define a global fallback order per primary:
#   primary 0 -> fallbacks [1, 2, 3]
#   primary 1 -> fallbacks [0, 2, 3]
#   primary 2 -> fallbacks [0, 1, 3]
#   primary 3 -> fallbacks [0, 1, 2]
FALLBACK_ORDER: list[list[int]] = [
    [1, 2, 3],
    [0, 2, 3],
    [0, 1, 3],
    [0, 1, 2],
]


def _in_range(params: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> bool:
    """Check whether params lies within [lo, hi] element-wise."""
    return bool(np.all(params >= lo) and np.all(params <= hi))


def make_dataset(seed: int = 42, n_samples: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic tool-fallback dataset.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    n_samples : int
        Number of tasks. Must be >= 32.

    Returns
    -------
    X : np.ndarray, shape (n_samples, 5)
        Columns: [tool_id, p1, p2, p3, unavailable_flag]
        - tool_id: int in {0, 1, 2, 3}, the primary tool for the task.
        - p1, p2, p3: float, sampled uniformly from the primary tool's range.
        - unavailable_flag: float (0.0 or 1.0), Bernoulli(p=0.3).
    y : np.ndarray, shape (n_samples,)
        1 if the task is completable (primary available OR at least one
        other tool's range contains the params), 0 otherwise.
    """
    if n_samples < 32:
        raise ValueError(f"n_samples must be >= 32, got {n_samples}")

    rng = np.random.default_rng(seed)

    # Assign each task a primary tool uniformly at random
    tool_ids = rng.integers(0, 4, size=n_samples)

    # Sample parameters uniformly from the primary tool's range
    p1 = np.empty(n_samples, dtype=np.float64)
    p2 = np.empty(n_samples, dtype=np.float64)
    p3 = np.empty(n_samples, dtype=np.float64)
    for i in range(n_samples):
        t = int(tool_ids[i])
        lo, hi = TOOL_RANGES[t]
        p1[i] = rng.uniform(lo[0], hi[0])
        p2[i] = rng.uniform(lo[1], hi[1])
        p3[i] = rng.uniform(lo[2], hi[2])

    # Unavailability flag: Bernoulli(p=0.3)
    unavailable = (rng.random(n_samples) < 0.3).astype(np.float64)

    # Compute ground-truth completability y:
    # A task is completable if:
    #   - primary is available (unavailable_flag == 0), OR
    #   - at least one other tool's range contains the parameter vector
    y = np.zeros(n_samples, dtype=np.int64)
    for i in range(n_samples):
        if unavailable[i] == 0.0:
            y[i] = 1
            continue
        # Primary unavailable; check if any other tool can handle the params
        t = int(tool_ids[i])
        params = np.array([p1[i], p2[i], p3[i]])
        completable = False
        for other in range(4):
            if other == t:
                continue
            lo, hi = TOOL_RANGES[other]
            if _in_range(params, lo, hi):
                completable = True
                break
        y[i] = 1 if completable else 0

    X = np.column_stack([
        tool_ids.astype(np.float64),
        p1,
        p2,
        p3,
        unavailable,
    ])

    return X, y
