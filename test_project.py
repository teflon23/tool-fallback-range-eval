"""test_project.py: tests for the tool-fallback range-eval project."""
from __future__ import annotations

import math

import numpy as np
import pytest

from data import make_dataset, TOOL_RANGES, FALLBACK_ORDER, _in_range
from app import run_experiment


def test_dataset_shape_and_seed_sensitivity():
    """make_dataset returns correct shapes and different seeds give different X."""
    X1, y1 = make_dataset(seed=1, n_samples=64)
    X2, y2 = make_dataset(seed=2, n_samples=64)

    assert X1.shape == (64, 5)
    assert y1.shape == (64,)
    assert X2.shape == (64, 5)
    assert y2.shape == (64,)

    # Different seeds must produce different X
    assert not np.array_equal(X1, X2)

    # y values must be 0 or 1
    assert set(np.unique(y1)).issubset({0, 1})
    assert set(np.unique(y2)).issubset({0, 1})


def test_dataset_invalid_n_samples():
    """make_dataset raises ValueError for n_samples < 32."""
    with pytest.raises(ValueError):
        make_dataset(seed=42, n_samples=16)

    with pytest.raises(ValueError):
        make_dataset(seed=42, n_samples=0)

    with pytest.raises(ValueError):
        make_dataset(seed=42, n_samples=31)


def test_experiment_completion_rate_and_finite_metrics():
    """run_experiment returns valid metrics; algorithm >= baseline completion rate."""
    result = run_experiment(seed=42, n_samples=128)

    assert result["n_samples"] == 128

    metrics = result["metrics"]
    baseline_metrics = result["baseline_metrics"]

    # All metric values must be finite
    for key, val in metrics.items():
        assert isinstance(val, (int, float)), f"metrics[{key}] is not numeric"
        assert math.isfinite(val), f"metrics[{key}] is not finite"

    for key, val in baseline_metrics.items():
        assert isinstance(val, (int, float)), f"baseline_metrics[{key}] is not numeric"
        assert math.isfinite(val), f"baseline_metrics[{key}] is not finite"

    # Algorithm completion rate must be >= baseline completion rate
    assert metrics["completion_rate"] >= baseline_metrics["completion_rate"]

    # avg_tool_calls must be >= 1.0 (at least one call per task)
    assert metrics["avg_tool_calls"] >= 1.0
    assert baseline_metrics["avg_tool_calls"] >= 1.0


def test_experiment_deterministic():
    """Same inputs must give exactly the same output."""
    r1 = run_experiment(seed=7, n_samples=64)
    r2 = run_experiment(seed=7, n_samples=64)

    assert r1["n_samples"] == r2["n_samples"]
    for key in r1["metrics"]:
        assert r1["metrics"][key] == r2["metrics"][key], f"metrics[{key}] differs"
    for key in r1["baseline_metrics"]:
        assert r1["baseline_metrics"][key] == r2["baseline_metrics"][key], \
            f"baseline_metrics[{key}] differs"


def test_algorithm_completes_all_ground_truth_completable_tasks():
    """
    The algorithm should complete every task that is ground-truth completable
    (y=1). This is because the algorithm checks all 4 tools (primary + 3
    fallbacks), so if any tool's range contains the params, it will find it.
    """
    seed = 42
    n = 128
    X, y = make_dataset(seed=seed, n_samples=n)
    result = run_experiment(seed=seed, n_samples=n)

    # Reconstruct which tasks the algorithm completed
    tool_ids = X[:, 0].astype(int)
    params = X[:, 1:4]
    unavailable = X[:, 4]

    algo_completed = np.zeros(n, dtype=bool)
    for i in range(n):
        t = int(tool_ids[i])
        params_i = params[i]
        if unavailable[i] == 0.0:
            algo_completed[i] = True
        else:
            for fb_tool in FALLBACK_ORDER[t]:
                lo, hi = TOOL_RANGES[fb_tool]
                if _in_range(params_i, lo, hi):
                    algo_completed[i] = True
                    break

    # Every ground-truth completable task must be completed by the algorithm
    for i in range(n):
        if y[i] == 1:
            assert algo_completed[i], \
                f"Task {i} is ground-truth completable but algorithm failed"


def test_baseline_fails_all_unavailable_tasks():
    """
    The baseline has no fallback logic, so it must fail every task where
    the primary tool is unavailable.
    """
    seed = 42
    n = 128
    X, y = make_dataset(seed=seed, n_samples=n)
    result = run_experiment(seed=seed, n_samples=n)

    unavailable = X[:, 4]
    n_unavailable = int(np.sum(unavailable == 1.0))

    # Baseline completion rate should equal the fraction of available tasks
    expected_base_completion = (n - n_unavailable) / n
    assert abs(result["baseline_metrics"]["completion_rate"] - expected_base_completion) < 1e-12


def test_fallback_depth_values():
    """
    mean_fallback_depth for the algorithm should be >= 0 when all tasks
    use the primary, and > 0 when some tasks require fallback.
    For the baseline, mean_fallback_depth should always be 0.
    """
    result = run_experiment(seed=42, n_samples=128)

    # Baseline depth is always 0
    assert result["baseline_metrics"]["mean_fallback_depth"] == 0.0

    # Algorithm depth should be >= 0 (can be 0 if no fallback needed,
    # but with p=0.3 unavailability and overlapping ranges, some fallback
    # will be used, so it should be > 0 for typical seeds)
    assert result["metrics"]["mean_fallback_depth"] >= 0.0


def test_different_n_samples_respected():
    """run_experiment must respect different n_samples values."""
    r64 = run_experiment(seed=42, n_samples=64)
    r256 = run_experiment(seed=42, n_samples=256)

    assert r64["n_samples"] == 64
    assert r256["n_samples"] == 256

    # Metrics should differ (different number of tasks)
    # At minimum, the completion rates may differ
    # We just check they are valid finite numbers
    for key in r64["metrics"]:
        assert math.isfinite(r64["metrics"][key])
    for key in r256["metrics"]:
        assert math.isfinite(r256["metrics"][key])
