"""app.py: tool-fallback selection experiment."""
from __future__ import annotations

import numpy as np

from data import make_dataset, TOOL_RANGES, FALLBACK_ORDER, _in_range


def run_experiment(seed: int = 42, n_samples: int = 256) -> dict:
    """
    Run the tool-fallback selection experiment.

    Algorithm:
      For each task:
        - If the primary tool is available, use it (1 call, depth 0).
        - If the primary tool is unavailable, iterate the fixed ordered
          fallback list for that primary. Select the first fallback whose
          3-D parameter range encloses the task's parameter vector.
          Record the list position (1-indexed) as depth.
        - If no fallback accepts, the task fails (depth = 0, not counted in depth stats).

    Baseline:
      No fallback logic. If the primary is unavailable, the task fails
      immediately (1 call, depth 0).

    Returns a JSON-serializable dict with metrics and baseline_metrics.
    """
    X, y = make_dataset(seed=seed, n_samples=n_samples)

    n = len(X)
    tool_ids = X[:, 0].astype(int)
    params = X[:, 1:4]  # shape (n, 3)
    unavailable = X[:, 4]

    # --- Algorithm metrics ---
    algo_completed = 0
    algo_total_calls = 0
    algo_depths = []

    # --- Baseline metrics ---
    base_completed = 0
    base_total_calls = 0
    base_depths = []

    for i in range(n):
        t = int(tool_ids[i])
        params_i = params[i]
        is_unavailable = unavailable[i] == 1.0

        # --- Algorithm ---
        if not is_unavailable:
            # Primary available: use it
            algo_completed += 1
            algo_total_calls += 1
            algo_depths.append(0)
        else:
            # Primary unavailable: walk fallback list
            fallbacks = FALLBACK_ORDER[t]
            found = False
            for depth_idx, fb_tool in enumerate(fallbacks, start=1):
                lo, hi = TOOL_RANGES[fb_tool]
                if _in_range(params_i, lo, hi):
                    algo_completed += 1
                    # Calls: 1 (primary attempt) + depth_idx (fallback attempts until success)
                    algo_total_calls += 1 + depth_idx
                    algo_depths.append(depth_idx)
                    found = True
                    break
            if not found:
                # No fallback works: calls = 1 (primary) + len(fallbacks) (all fallbacks tried)
                algo_total_calls += 1 + len(fallbacks)
                # For failed tasks, we don't count depth in the mean (or count as 0)
                # To keep mean_fallback_depth >= 0, we only count successful fallbacks
                # and primary uses. Failed tasks contribute 0 to depth sum but not to count.
                # Actually, let's just not append to depths for failures, or append 0.
                # The test expects mean >= 0. If we include -1, it can be negative.
                # Let's define mean_fallback_depth as the average depth of *completed* tasks
                # that used a fallback, or 0 if none.
                # But the metric name is "mean_fallback_depth".
                # Let's stick to: depth is 0 for primary, 1-3 for fallback.
                # For failures, depth is undefined. We'll exclude them from the mean calculation
                # or set them to 0.
                # Given the test failure, including -1 caused negative mean.
                # Let's exclude failed tasks from the depth average.
                pass

        # --- Baseline ---
        if not is_unavailable:
            base_completed += 1
            base_total_calls += 1
            base_depths.append(0)
        else:
            # Primary unavailable: immediate failure, 1 call
            base_total_calls += 1
            base_depths.append(0)

    # Compute metrics
    algo_completion_rate = algo_completed / n
    algo_avg_tool_calls = algo_total_calls / n
    
    # Mean fallback depth: average of depths for tasks that were completed.
    # If no tasks were completed, mean is 0.
    if algo_depths:
        algo_mean_depth = float(np.mean(algo_depths))
    else:
        algo_mean_depth = 0.0

    base_completion_rate = base_completed / n
    base_avg_tool_calls = base_total_calls / n
    base_mean_depth = float(np.mean(base_depths)) if base_depths else 0.0

    metrics = {
        "completion_rate": float(algo_completion_rate),
        "avg_tool_calls": float(algo_avg_tool_calls),
        "mean_fallback_depth": float(algo_mean_depth),
    }

    baseline_metrics = {
        "completion_rate": float(base_completion_rate),
        "avg_tool_calls": float(base_avg_tool_calls),
        "mean_fallback_depth": float(base_mean_depth),
    }

    explanation = (
        "Each task has a primary tool and 3-D parameters sampled from the primary's range. "
        "If the primary is unavailable (Bernoulli p=0.3), the algorithm walks a fixed ordered "
        "fallback list and selects the first tool whose range encloses the parameters. "
        "The baseline fails immediately when the primary is unavailable. "
        "No train/test split is needed: each task is an independent deterministic evaluation. "
        "Ground-truth y indicates whether the task is completable by any tool."
    )

    return {
        "n_samples": int(n),
        "metrics": metrics,
        "baseline_metrics": baseline_metrics,
        "explanation": explanation,
    }
