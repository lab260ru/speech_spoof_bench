# Adding a New Metric

Metrics are a plugin registry inside the package. Adding one (say min-tDCF, accuracy, or
an AUC) is a **package change** that bumps the package version. This is the smallest
"contribute to the package" task and a good first PR.

## How metrics work

- A metric is a function `(scores: dict[str,float], labels: dict[str,int]) -> MetricResult`.
- It registers itself with the `@register_metric(...)` decorator (`metrics/__init__.py`).
- **It only exists if its module is imported.** `metrics/__init__.py` does
  `from . import eer` precisely so the decorator runs on package import. Forget this line
  and `get_metric("your_id")` raises, and any dataset whose `eval.yaml` lists your metric
  fails D7 / `Benchmark.run`.

`MetricResult` carries the `value` plus an `extras` dict (free-form diagnostics, e.g. the
operating threshold and counts). The registry records `id`, `display_name`,
`lower_is_better`, and `requires_audio`.

## Step 1 — Write the metric module

```python
# src/speech_spoof_bench/metrics/accuracy.py
from __future__ import annotations
from . import register_metric, MetricResult   # match the real names in metrics/__init__.py

@register_metric(
    id="accuracy_percent",
    display_name="Accuracy (%)",
    lower_is_better=False,        # higher is better → affects ranking sort direction
    requires_audio=False,         # True only if your metric needs the raw audio, not just scores
)
def compute_accuracy(scores: dict[str, float], labels: dict[str, int]) -> MetricResult:
    # convention: higher score == more bona fide; label 0 == bona fide, 1 == spoof
    correct = sum((s < 0) == (labels[u] == 1) for u, s in scores.items())  # example threshold at 0
    value = 100.0 * correct / len(scores)
    return MetricResult(value=value, extras={"n_trials": len(scores)})
```

> Read `metrics/eer.py` and `metrics/__init__.py` first to copy the exact import names and
> `MetricResult` constructor — they're the source of truth, not this sketch.

## Step 2 — Register it (the line everyone forgets)

```python
# src/speech_spoof_bench/metrics/__init__.py
from . import eer        # existing
from . import accuracy   # ← ADD THIS
```

Without it the decorator never runs.

## Step 3 — `lower_is_better` is load-bearing

The Arena reads `lower_is_better` from the registry to decide ranking sort direction; an
unknown metric defaults to `True` (lower-is-better, like EER). Get this right or the
leaderboard sorts your metric backwards.

## Step 4 — Test it

```python
# tests/metrics/test_accuracy.py
from speech_spoof_bench.metrics import get_metric

def test_accuracy_registered():
    spec = get_metric("accuracy_percent")
    assert spec.lower_is_better is False

def test_accuracy_perfect():
    scores = {"a":  1.0, "b": -1.0}     # a bona fide, b spoof
    labels = {"a": 0,    "b": 1}
    assert get_metric("accuracy_percent").fn(scores, labels).value == 100.0
```

Run `pytest tests/metrics/`.

## Step 5 — Version & roll-out

Adding a metric is **additive** → a **minor** package bump (`0.1.0 → 0.2.0`):

1. Bump `pyproject.toml` `version` and `__init__.py` `__version__`.
2. Tag the pip release.
3. To actually *use* it: a dataset lists it in `eval.yaml` `metrics:` (D7 now passes
   because it's registered), and optionally the manifest's `metrics_in_use` /
   `ranking.metric` reference it.
4. **Bump the Arena pin** in `arena/requirements.txt` to the new package commit so the
   Space can render/rank it. (Otherwise the Space runs the old package and won't know your
   metric — see [../architecture/versioning.md](../architecture/versioning.md).)
5. If you make your metric the *ranking* metric, also set an `absence_penalty` for it in
   the manifest and bump `ranking_version` — otherwise absent datasets are dropped from the
   mean instead of penalised.

Old submissions are unaffected — the registry is append-only and old `bench_version`
artifacts keep working.

## Gotchas

- **`requires_audio=True`** changes the contract: the runner must provide audio to the
  metric, which is heavier. Only set it if you genuinely need the waveform (most detection
  metrics work from scores + labels alone).
- A badge colour scale only exists for `eer_percent` (`badge._color_for_eer`); any other
  metric badges as `blue`. Add a colour function if you want tiered colours for it.
- The error message for an unregistered metric is asserted in tests — if you change it,
  update those tests.
</content>
