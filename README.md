# Tool Fallback Selection via Parameter Range Compatibility

## Problem

In tool-augmented systems, a task is often assigned to a primary tool that may become unavailable at runtime (e.g., due to rate limits, maintenance, or resource exhaustion). A naive system fails the task immediately. A more robust system can fall back to alternative tools, but only if the task's parameters lie within the alternative tool's supported domain.

This project evaluates **ordered fallback search with parameter-range checking** against a **no-fallback baseline**. The core question: *when the primary tool is unavailable, how much does a deterministic, range-aware fallback strategy improve task completion?*

## Actual Implementation

- **`data.py`** generates a synthetic dataset of tasks. Each task has a primary tool ID (0–3), three continuous parameters sampled uniformly from the primary tool's 3-D range, and a Bernoulli(p=0.3) unavailability flag. The ground-truth label `y` is 1 if the task is completable (primary available **or** at least one other tool's range contains the parameters), 0 otherwise.
- **`app.py`** implements two strategies and computes metrics:
  - **Algorithm (fallback):** If the primary is available, use it (1 call, depth 0). If unavailable, iterate a fixed ordered fallback list of 3 alternatives; select the first whose 3-D range encloses the task's parameter vector. Record the 1-indexed list position as depth. If no fallback accepts, the task fails.
  - **Baseline (no fallback):** If the primary is unavailable, the task fails immediately after 1 call.
- **`test_project.py`** contains pytest tests validating dataset generation, seed sensitivity, edge cases, and metric properties.
- **`main.py`** (supplied by the host) is the CLI entry point that calls `run_experiment` and writes results to JSON.

## Architecture

```
data.py          →  make_dataset(seed, n_samples) → (X, y)
app.py           →  run_experiment(seed, n_samples) → dict
test_project.py  →  pytest test cases
main.py          →  CLI entry point (host-supplied)
```

There is no model training, no network access, and no external service. The entire pipeline is deterministic given a seed and sample size.

## Synthetic Dataset Assumptions

- **4 tools**, each with a fixed 3-D parameter range `[lo, hi]` per dimension. Ranges are chosen so that ~5–15% of tasks are truly incompletable (no tool's range contains the parameters), making the ground-truth label non-trivial.
- **Parameters** are sampled uniformly from the **primary tool's** range, so the primary can always handle the task if available.
- **Unavailability** is a seeded Bernoulli(p=0.3) flag, independent across tasks.
- **Limitations:** The ranges are hand-tuned and do not reflect real tool domains. The uniform sampling from the primary's range is a simplification; real tasks may have heterogeneous parameter distributions. The fixed fallback order is arbitrary and not learned.

## Algorithm vs. Baseline

| Aspect | Algorithm (fallback) | Baseline (no fallback) |
|---|---|---|
| Primary available | Use primary, 1 call, depth 0 | Use primary, 1 call, depth 0 |
| Primary unavailable | Walk ordered fallback list; use first range-compatible tool | Fail immediately, 1 call, depth 0 |
| No fallback compatible | Task fails after trying all 3 fallbacks | Task fails immediately |
| Expected completion rate | Higher (recovers some unavailable-primary tasks) | Lower (only primary-available tasks succeed) |
| Expected avg tool calls | Higher (extra calls for fallback attempts) | Lower (1 call per task) |

## Metrics (Direction)

| Metric | Direction | Description |
|---|---|---|
| `completion_rate` | Higher is better | Fraction of tasks the strategy completes |
| `avg_tool_calls` | Lower is better | Mean number of tool invocations per task (including the primary attempt) |
| `mean_fallback_depth` | Lower is better | Mean 1-indexed position in the fallback list for completed tasks that used a fallback; 0 for primary-only completions. Failed tasks are excluded from this average. |

## Reproducibility

The project targets **Python 3.11–3.13**. All randomness is controlled by `numpy.random.default_rng(seed)`, so identical inputs produce identical outputs.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py --seed 42 --n-samples 256 --output results.json
python -m pytest -q
```

Refer to `validation_report.json` and `example_results.json` for measured output from a representative run. No numerical results are claimed in this README.

## Limitations

- The tool ranges and fallback order are synthetic and hand-tuned; they do not model real-world tool domains or dynamic availability.
- The fallback order is fixed and not optimized; a different ordering could yield different completion rates.
- The dataset assumes parameters are drawn from the primary tool's range, which is a simplification.
- The evaluation is in-memory and deterministic; it does not simulate concurrent load, latency, or failure cascades.
- Automated tests validate structural and statistical properties of the dataset and metrics but do not guarantee correctness of the fallback logic in all edge cases beyond those explicitly tested.
- This is an educational project and is **not** intended for production use.

## Recorded automated validation

Host contract tests and project tests passed (12 tests, 0 skipped). Demo completed on Python 3.13.15. See `validation_report.json` and `example_results.json`. These checks validate the execution contract, not scientific novelty or every algorithmic claim.
