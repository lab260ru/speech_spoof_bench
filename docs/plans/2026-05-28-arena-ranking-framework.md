# Arena Ranking Framework + Badges Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A configurable, manifest-driven global ranking formula for the arena (coverage-rewarding, γ-dial unifying pooled/aggregated), plus the tier + global-rank badges and an Overview pooled/aggregated toggle that render it.

**Architecture:** A pure formula in `arena/ranking.py` computes `score(s)` = weighted mean of each system's per-dataset primary-metric value over the **full Core Set**, with absent datasets contributing a configurable penalty and per-dataset weights `base_d(γ)·manual_d`. Config lives in `arena-manifest/manifest.yaml` (validated by the package's `manifest.schema.json`); the arena reads it via `ingest`. Badges (`arena/badges.py`, shields-endpoint JSON) and the Overview tab render the result.

**Tech Stack:** Python 3.10+, FastAPI, Gradio, `huggingface_hub`, `jsonschema`, `pyyaml`, `pytest`, shields.io endpoint badges, HF Docker Space.

**Spec:** `docs/specs/2026-05-28-arena-ranking-framework-design.md` (supersedes `2026-05-28-arena-tier-rank-badges-design.md`).

**Repos touched** (independent `.git` each — always `cd` before git):
- `arena/` — Row/ingest plumbing, ranking formula, badges router, main.py mount, Overview UI (→ HF Space)
- `speech-spoof-bench/` — `manifest.schema.json`, `badge.build_paste_comment` lines (→ GitHub)
- `arena-manifest/` — `manifest.yaml` `ranking` block + tier colors (→ HF dataset)

**Arena Space hostname** (shields endpoint URLs): `speechantispoofingbenchmarks-speechantispoofingarena.hf.space`

**Formula (reference, from spec):**
```
score(s) = ( Σ_d w_d·value(s,d) ) / ( Σ_d w_d )
  w_d        = base_d(γ) · manual_d
  base_d(γ)  = (n_trials_d)**γ / max_core((n_trials)**γ)     ∈ (0,1]
  value(s,d) = metric value if covered else absence_penalty
  direction  = metric.lower_is_better
```

---

# Phase A — Formula core

## Task 1: `Row` carries `n_trials`

**Files:**
- Modify: `arena/schema.py`
- Modify: `arena/ingest.py`
- Modify: `arena/tests/test_ingest.py`

- [ ] **Step 1: Write the failing test**

Append to `arena/tests/test_ingest.py`:

```python
def test_to_row_populates_n_trials():
    import ingest
    sub = {
        "system": {"slug": "s", "name": "S"},
        "dataset": {"revision": "r"},
        "scores": {"eer_percent": 1.5, "n_trials": 71237, "n_skipped": 0},
        "reproduction": {"match": "scoring"},
        "submitted_at": "2026-01-01",
    }
    row = ingest._to_row(sub, "Org/A", "submissions/s.yaml")
    assert row.n_trials == 71237
    # n_trials/n_skipped still excluded from the metric scores dict
    assert "n_trials" not in row.scores and "n_skipped" not in row.scores
    assert row.scores == {"eer_percent": 1.5}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/kirill/speech-spoof-bench/arena && pytest tests/test_ingest.py::test_to_row_populates_n_trials -v`
Expected: FAIL — `Row.__init__() got unexpected... ` / `n_trials` attribute missing.

- [ ] **Step 3: Add the field + populate it**

In `arena/schema.py`, add `n_trials` to `Row` (after `revision`), with a default so legacy cached rows hydrate cleanly:

```python
@dataclass(frozen=True)
class Row:
    system_slug: str
    system_name: str
    dataset_id: str
    revision: str
    scores: dict[str, float]
    reproduction_level: str         # "scoring" | "inference"
    submitted_at: str
    submission_url: str
    n_trials: int = 0               # test-set size; 0 = unknown (legacy cache)
```

In `arena/ingest.py`, set it in `_to_row` (the function currently strips
`n_trials` out of `scores` — keep that, but capture the value first):

```python
def _to_row(sub: dict, dataset_id: str, path: str) -> Row:
    scores = {k: v for k, v in sub["scores"].items() if k not in {"n_trials", "n_skipped"}}
    return Row(
        system_slug=sub["system"]["slug"],
        system_name=sub["system"]["name"],
        dataset_id=dataset_id,
        revision=sub["dataset"]["revision"],
        scores=scores,
        reproduction_level=sub["reproduction"]["match"],
        submitted_at=sub["submitted_at"] if isinstance(sub["submitted_at"], str) else sub["submitted_at"].isoformat(),
        submission_url=f"https://huggingface.co/datasets/{dataset_id}/blob/main/{path}",
        n_trials=int(sub["scores"].get("n_trials", 0)),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/kirill/speech-spoof-bench/arena && pytest tests/test_ingest.py -v`
Expected: PASS (existing + new).

- [ ] **Step 5: Commit**

```bash
cd /home/kirill/speech-spoof-bench/arena
git add schema.py ingest.py tests/test_ingest.py
git commit -m "feat(arena): Row carries n_trials (test-set size) from submissions"
```

---

## Task 2: `cache.json` round-trips `n_trials`

**Files:**
- Modify: `arena/cache_store.py`
- Modify: `arena/tests/test_cache_store.py`

Context: `cache_store` serializes `ArenaState` to `cache.json` and rehydrates it.
The serializer must include `n_trials`; the deserializer must default it to 0
when missing (legacy cache). The content-hash debounce must include `n_trials`
so a changed trial count triggers a rebuild.

- [ ] **Step 1: Read the current serializer/deserializer**

Run: `cd /home/kirill/speech-spoof-bench/arena && sed -n '1,200p' cache_store.py`
Find where a `Row` is converted to/from a dict (look for `Row(` and `asdict`/
field access). Note the exact dict keys used.

- [ ] **Step 2: Write the failing test**

Append to `arena/tests/test_cache_store.py` (adapt imports to match the file's
existing style — it already imports `cache_store` and builds `ArenaState`s):

```python
def test_cache_roundtrip_preserves_n_trials(tmp_path, monkeypatch):
    import cache_store
    from schema import ArenaState, Row
    from datetime import datetime

    row = Row(system_slug="s", system_name="S", dataset_id="Org/A", revision="r",
              scores={"eer_percent": 1.5}, reproduction_level="scoring",
              submitted_at="2026-01-01", submission_url="u", n_trials=71237)
    state = ArenaState(manifest={"x": 1}, rows=[row], loaded_at=datetime(2026, 1, 1))

    blob = cache_store._serialize(state)          # dict ready for json.dump
    restored = cache_store._deserialize(blob)     # ArenaState
    assert restored.rows[0].n_trials == 71237


def test_cache_deserialize_legacy_row_without_n_trials_defaults_zero():
    import cache_store
    legacy = {
        "manifest": {}, "loaded_at": "2026-01-01T00:00:00", "warnings": [],
        "rows": [{
            "system_slug": "s", "system_name": "S", "dataset_id": "Org/A",
            "revision": "r", "scores": {"eer_percent": 1.5},
            "reproduction_level": "scoring", "submitted_at": "2026-01-01",
            "submission_url": "u",
            # no n_trials key
        }],
    }
    restored = cache_store._deserialize(legacy)
    assert restored.rows[0].n_trials == 0
```

If the serializer/deserializer helpers have different names than `_serialize`/
`_deserialize`, rename the test calls to match what Step 1 found, and keep the
two assertions.

- [ ] **Step 3: Run test to verify it fails**

Run: `cd /home/kirill/speech-spoof-bench/arena && pytest tests/test_cache_store.py -k n_trials -v`
Expected: FAIL — `n_trials` not serialized / KeyError or default missing.

- [ ] **Step 4: Implement**

In `cache_store.py`, in the Row→dict serializer add `"n_trials": row.n_trials`.
In the dict→Row deserializer add `n_trials=d.get("n_trials", 0)`. If the
content-hash function builds a tuple/dict per row, include `n_trials` in it so a
trial-count change is not debounced away.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /home/kirill/speech-spoof-bench/arena && pytest tests/test_cache_store.py -v`
Expected: PASS (existing + 2 new).

- [ ] **Step 6: Commit**

```bash
cd /home/kirill/speech-spoof-bench/arena
git add cache_store.py tests/test_cache_store.py
git commit -m "feat(arena): cache.json round-trips Row.n_trials (legacy → 0)"
```

---

## Task 3: `ranking._ranking_config` — parse the manifest block with defaults

**Files:**
- Modify: `arena/ranking.py`
- Modify: `arena/tests/test_ranking.py`

- [ ] **Step 1: Write the failing test**

Append to `arena/tests/test_ranking.py`:

```python
from ranking import _ranking_config


def test_ranking_config_defaults_when_block_absent():
    manifest = {"metrics_in_use": ["eer_percent"], "tiers": [], "core_set": [], "extended": []}
    cfg = _ranking_config(manifest)
    assert cfg["metric"] == "eer_percent"
    assert cfg["absence_penalty"] == 50.0       # eer_percent worst-case default
    assert cfg["gamma_aggregated"] == 0.0
    assert cfg["gamma_pooled"] == 1.0
    assert cfg["default_view"] == "aggregated"
    assert cfg["weights"] == {}


def test_ranking_config_reads_block():
    manifest = {
        "metrics_in_use": ["eer_percent"],
        "ranking": {
            "metric": "eer_percent", "absence_penalty": 42.0,
            "gamma_aggregated": 0.0, "gamma_pooled": 0.5,
            "default_view": "pooled",
            "weights": {"Org/A": 2.0},
        },
    }
    cfg = _ranking_config(manifest)
    assert cfg["absence_penalty"] == 42.0
    assert cfg["gamma_pooled"] == 0.5
    assert cfg["default_view"] == "pooled"
    assert cfg["weights"] == {"Org/A": 2.0}


def test_ranking_config_non_eer_metric_without_penalty_is_none():
    manifest = {"metrics_in_use": ["accuracy"], "ranking": {"metric": "accuracy"}}
    cfg = _ranking_config(manifest)
    # No safe worst-case for an unknown metric → None signals "exclude absent".
    assert cfg["absence_penalty"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/kirill/speech-spoof-bench/arena && pytest tests/test_ranking.py -k ranking_config -v`
Expected: FAIL — `cannot import name '_ranking_config'`.

- [ ] **Step 3: Implement**

Append to `arena/ranking.py`:

```python
# Worst-case metric values used as the absence penalty when not configured.
_DEFAULT_PENALTY = {"eer_percent": 50.0}


def _ranking_config(manifest: dict) -> dict:
    """Resolve the ranking config with defaults. absence_penalty is None when no
    safe worst-case exists (unknown metric + no explicit value) — callers then
    exclude absent datasets instead of penalizing."""
    block = manifest.get("ranking") or {}
    metric = block.get("metric") or (manifest.get("metrics_in_use") or ["eer_percent"])[0]
    if "absence_penalty" in block:
        penalty = block["absence_penalty"]
    else:
        penalty = _DEFAULT_PENALTY.get(metric)  # None for unknown metrics
    return {
        "metric": metric,
        "absence_penalty": penalty,
        "gamma_aggregated": float(block.get("gamma_aggregated", 0.0)),
        "gamma_pooled": float(block.get("gamma_pooled", 1.0)),
        "default_view": block.get("default_view", "aggregated"),
        "weights": dict(block.get("weights", {})),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/kirill/speech-spoof-bench/arena && pytest tests/test_ranking.py -k ranking_config -v`
Expected: PASS (3).

- [ ] **Step 5: Commit**

```bash
cd /home/kirill/speech-spoof-bench/arena
git add ranking.py tests/test_ranking.py
git commit -m "feat(ranking): _ranking_config — manifest block + defaults"
```

---

## Task 4: `ranking.global_scores` — the formula

**Files:**
- Modify: `arena/ranking.py`
- Modify: `arena/tests/test_ranking.py`

- [ ] **Step 1: Write the failing tests**

Append to `arena/tests/test_ranking.py`:

```python
from ranking import global_scores


def _row_n(slug, dataset, eer, n_trials):
    return Row(system_slug=slug, system_name=slug, dataset_id=dataset, revision="r",
               scores={"eer_percent": eer}, reproduction_level="scoring",
               submitted_at="2026-01-01", submission_url="u", n_trials=n_trials)


CORE2 = ["org/a", "org/b"]
MANIFEST2 = {
    "metrics_in_use": ["eer_percent"],
    "core_set": [{"id": "org/a", "revision": "r"}, {"id": "org/b", "revision": "r"}],
    "extended": [],
    "ranking": {"absence_penalty": 50.0},
}


def test_aggregated_is_plain_mean_over_core():
    # one system, both datasets, eer 2 and 4 → mean 3.0 at gamma=0
    rows = [_row_n("s", "org/a", 2.0, 100), _row_n("s", "org/b", 4.0, 100)]
    scores = global_scores(rows, MANIFEST2, view="aggregated")
    assert scores["s"] == 3.0


def test_absence_penalty_applied_for_uncovered_core_dataset():
    # covers only org/a (eer 2); org/b absent → penalty 50; mean = 26.0
    rows = [_row_n("s", "org/a", 2.0, 100)]
    scores = global_scores(rows, MANIFEST2, view="aggregated")
    assert scores["s"] == 26.0


def test_more_coverage_scores_better_than_less():
    rows = [
        _row_n("full", "org/a", 5.0, 100), _row_n("full", "org/b", 5.0, 100),
        _row_n("half", "org/a", 5.0, 100),  # org/b absent → penalty
    ]
    scores = global_scores(rows, MANIFEST2, view="aggregated")
    assert scores["full"] < scores["half"]   # lower eer = better; full wins


def test_pooled_weights_by_n_trials():
    # org/a eer 2 with 900 trials, org/b eer 10 with 100 trials.
    # gamma=1 pooled mean ≈ (2*900 + 10*100)/1000 = 2.8; aggregated = 6.0
    rows = [_row_n("s", "org/a", 2.0, 900), _row_n("s", "org/b", 10.0, 100)]
    agg = global_scores(rows, MANIFEST2, view="aggregated")["s"]
    pooled = global_scores(rows, MANIFEST2, view="pooled")["s"]
    assert agg == 6.0
    assert abs(pooled - 2.8) < 1e-9


def test_manual_weight_shifts_contribution():
    m = {**MANIFEST2, "ranking": {"absence_penalty": 50.0, "weights": {"org/b": 3.0}}}
    rows = [_row_n("s", "org/a", 2.0, 100), _row_n("s", "org/b", 6.0, 100)]
    # aggregated base=1 each; manual b=3 → (2*1 + 6*3)/(1+3) = 20/4 = 5.0
    assert global_scores(rows, m, view="aggregated")["s"] == 5.0


def test_zero_submission_core_dataset_uses_n_trials_one():
    # No rows for org/b at all → n_trials_b defaults to 1; still penalized, no error.
    rows = [_row_n("s", "org/a", 2.0, 100)]
    scores = global_scores(rows, MANIFEST2, view="pooled")
    assert "s" in scores  # no division error


def test_penalty_none_excludes_absent_dataset():
    m = {"metrics_in_use": ["accuracy"],
         "core_set": [{"id": "org/a", "revision": "r"}, {"id": "org/b", "revision": "r"}],
         "extended": [], "ranking": {"metric": "accuracy"}}  # no penalty, unknown metric
    rows = [Row(system_slug="s", system_name="s", dataset_id="org/a", revision="r",
                scores={"accuracy": 0.9}, reproduction_level="scoring",
                submitted_at="2026-01-01", submission_url="u", n_trials=100)]
    # absent org/b excluded → mean over covered only = 0.9
    assert global_scores(rows, m, view="aggregated")["s"] == 0.9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/kirill/speech-spoof-bench/arena && pytest tests/test_ranking.py -k "global_scores or aggregated or pooled or coverage or manual_weight or penalty_none or zero_submission" -v`
Expected: FAIL — `cannot import name 'global_scores'`.

- [ ] **Step 3: Implement**

Append to `arena/ranking.py` (top of file already has `from collections import defaultdict`):

```python
def _core_n_trials(rows: Iterable[Row], core_set_ids: list[str]) -> dict[str, int]:
    """Per-dataset test-set size (n_trials), read from any row on that dataset.
    Datasets with no rows default to 1 (so gamma has no effect there)."""
    out = {ds: 1 for ds in core_set_ids}
    for r in rows:
        if r.dataset_id in out and r.n_trials > 0:
            out[r.dataset_id] = r.n_trials
    return out


def _base_weights(n_trials: dict[str, int], gamma: float) -> dict[str, float]:
    """base_d(gamma) = n_trials_d**gamma / max(n_trials**gamma), in (0, 1]."""
    powered = {ds: (nt ** gamma) for ds, nt in n_trials.items()}
    peak = max(powered.values()) if powered else 1.0
    if peak <= 0:
        peak = 1.0
    return {ds: (p / peak) for ds, p in powered.items()}


def global_scores(rows: Iterable[Row], manifest: dict, view: str) -> dict[str, float]:
    """system_slug -> weighted-mean score over the full Core Set.

    view = 'aggregated' uses gamma_aggregated; 'pooled' uses gamma_pooled.
    Absent core datasets contribute absence_penalty (or are excluded if it is
    None). Returns {} when there are no core datasets.
    """
    rows = list(rows)
    cfg = _ranking_config(manifest)
    core = [e["id"] for e in manifest.get("core_set", [])]
    if not core:
        return {}
    gamma = cfg["gamma_pooled"] if view == "pooled" else cfg["gamma_aggregated"]
    metric = cfg["metric"]
    penalty = cfg["absence_penalty"]
    manual = cfg["weights"]

    n_trials = _core_n_trials(rows, core)
    base = _base_weights(n_trials, gamma)

    by_system: dict[str, dict[str, Row]] = defaultdict(dict)
    for r in rows:
        if r.dataset_id in set(core):
            by_system[r.system_slug][r.dataset_id] = r

    out: dict[str, float] = {}
    for slug, per_ds in by_system.items():
        num = 0.0
        den = 0.0
        for ds in core:
            covered = ds in per_ds and per_ds[ds].scores.get(metric) is not None
            if covered:
                value = per_ds[ds].scores[metric]
            elif penalty is not None:
                value = penalty
            else:
                continue  # exclude absent dataset entirely
            w = base[ds] * float(manual.get(ds, 1.0))
            num += w * value
            den += w
        out[slug] = (num / den) if den else float("inf")
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/kirill/speech-spoof-bench/arena && pytest tests/test_ranking.py -v`
Expected: PASS (all existing + new).

- [ ] **Step 5: Commit**

```bash
cd /home/kirill/speech-spoof-bench/arena
git add ranking.py tests/test_ranking.py
git commit -m "feat(ranking): global_scores — coverage-rewarding weighted-mean formula"
```

---

## Task 5: `ranking.global_rank` — places from scores

**Files:**
- Modify: `arena/ranking.py`
- Modify: `arena/tests/test_ranking.py`

- [ ] **Step 1: Write the failing tests**

Append to `arena/tests/test_ranking.py`:

```python
from ranking import global_rank


def test_global_rank_orders_by_score_lower_is_better():
    rows = [
        _row_n("good", "org/a", 1.0, 100), _row_n("good", "org/b", 1.0, 100),
        _row_n("bad", "org/a", 9.0, 100), _row_n("bad", "org/b", 9.0, 100),
    ]
    rank = global_rank(rows, MANIFEST2, view="aggregated")
    assert rank["good"] == {"place": 1, "out_of": 2}
    assert rank["bad"] == {"place": 2, "out_of": 2}


def test_global_rank_is_board_wide_not_per_tier():
    # 'full' covers both (gold), 'half' covers one (silver) — both appear in one
    # global ordering; coverage makes 'full' win.
    rows = [
        _row_n("full", "org/a", 5.0, 100), _row_n("full", "org/b", 5.0, 100),
        _row_n("half", "org/a", 1.0, 100),  # great on a, but penalized for b
    ]
    rank = global_rank(rows, MANIFEST2, view="aggregated")
    assert rank["full"]["place"] == 1
    assert rank["half"]["place"] == 2
    assert rank["full"]["out_of"] == 2


def test_global_rank_ties_broken_by_slug():
    rows = [
        _row_n("bbb", "org/a", 5.0, 100), _row_n("bbb", "org/b", 5.0, 100),
        _row_n("aaa", "org/a", 5.0, 100), _row_n("aaa", "org/b", 5.0, 100),
    ]
    rank = global_rank(rows, MANIFEST2, view="aggregated")
    assert rank["aaa"]["place"] == 1 and rank["bbb"]["place"] == 2


def test_global_rank_respects_higher_is_better_metric(monkeypatch):
    # A metric where higher wins: stub the registry lookup.
    import ranking
    monkeypatch.setattr(ranking, "_lower_is_better", lambda metric: False)
    m = {"metrics_in_use": ["acc"],
         "core_set": [{"id": "org/a", "revision": "r"}], "extended": [],
         "ranking": {"metric": "acc", "absence_penalty": 0.0}}
    rows = [
        Row(system_slug="hi", system_name="hi", dataset_id="org/a", revision="r",
            scores={"acc": 0.9}, reproduction_level="scoring",
            submitted_at="2026-01-01", submission_url="u", n_trials=100),
        Row(system_slug="lo", system_name="lo", dataset_id="org/a", revision="r",
            scores={"acc": 0.1}, reproduction_level="scoring",
            submitted_at="2026-01-01", submission_url="u", n_trials=100),
    ]
    rank = global_rank(rows, m, view="aggregated")
    assert rank["hi"]["place"] == 1 and rank["lo"]["place"] == 2


def test_global_rank_single_system():
    rows = [_row_n("solo", "org/a", 3.0, 100), _row_n("solo", "org/b", 3.0, 100)]
    assert global_rank(rows, MANIFEST2, view="aggregated")["solo"] == {"place": 1, "out_of": 1}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/kirill/speech-spoof-bench/arena && pytest tests/test_ranking.py -k global_rank -v`
Expected: FAIL — `cannot import name 'global_rank'` / `_lower_is_better`.

- [ ] **Step 3: Implement**

Append to `arena/ranking.py`:

```python
def _lower_is_better(metric: str) -> bool:
    """Metric direction from the package registry; default True (EER-like)."""
    try:
        from speech_spoof_bench.metrics import get_metric
        return get_metric(metric).lower_is_better
    except Exception:  # noqa: BLE001 — unknown/unregistered metric → assume lower better
        return True


def global_rank(rows: Iterable[Row], manifest: dict, view: str) -> dict[str, dict]:
    """system_slug -> {'place': int (1-based), 'out_of': int}. Board-wide
    ordering by global_scores in the metric's direction; ties broken by slug."""
    scores = global_scores(rows, manifest, view)
    if not scores:
        return {}
    cfg = _ranking_config(manifest)
    lower_better = _lower_is_better(cfg["metric"])
    # Sort: best first. For lower-is-better, ascending score; else descending.
    ordered = sorted(
        scores.items(),
        key=lambda kv: (kv[1] if lower_better else -kv[1], kv[0]),
    )
    n = len(ordered)
    return {slug: {"place": i + 1, "out_of": n} for i, (slug, _) in enumerate(ordered)}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/kirill/speech-spoof-bench/arena && pytest tests/test_ranking.py -v`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
cd /home/kirill/speech-spoof-bench/arena
git add ranking.py tests/test_ranking.py
git commit -m "feat(ranking): global_rank — board-wide place from global_scores"
```

---

## Task 6: `manifest.schema.json` — optional `ranking` block + tier `color`

**Files:**
- Modify: `speech-spoof-bench/src/speech_spoof_bench/schema/manifest.schema.json`
- Create: `speech-spoof-bench/tests/test_manifest_schema_ranking.py`

- [ ] **Step 1: Write the failing test**

```python
# speech-spoof-bench/tests/test_manifest_schema_ranking.py
"""manifest.schema.json accepts the optional ranking block + tier color."""
from __future__ import annotations

import json
from importlib import resources

import pytest
from jsonschema import ValidationError, validate


def _schema():
    with resources.files("speech_spoof_bench.schema").joinpath("manifest.schema.json").open("r") as f:
        return json.load(f)


def _m(extra=None, tiers=None):
    base = {
        "ranking_version": "v1", "schema_version": 1,
        "metrics_in_use": ["eer_percent"],
        "tiers": tiers or [{"name": "gold", "min_coverage": 1.0}],
        "core_set": [{"id": "Org/A", "revision": "abc1234"}],
        "extended": [],
    }
    if extra:
        base.update(extra)
    return base


def test_manifest_without_ranking_block_valid():
    validate(_m(), _schema())


def test_manifest_with_full_ranking_block_valid():
    validate(_m({"ranking": {
        "metric": "eer_percent", "absence_penalty": 50.0,
        "gamma_aggregated": 0.0, "gamma_pooled": 1.0,
        "default_view": "aggregated", "weights": {"Org/A": 1.0},
    }}), _schema())


def test_tier_with_color_valid():
    validate(_m(tiers=[{"name": "gold", "min_coverage": 1.0, "color": "#FFD700"}]), _schema())


def test_bad_default_view_rejected():
    with pytest.raises(ValidationError):
        validate(_m({"ranking": {"default_view": "sideways"}}), _schema())


def test_negative_gamma_rejected():
    with pytest.raises(ValidationError):
        validate(_m({"ranking": {"gamma_pooled": -1.0}}), _schema())


def test_non_number_penalty_rejected():
    with pytest.raises(ValidationError):
        validate(_m({"ranking": {"absence_penalty": "high"}}), _schema())


def test_unknown_ranking_field_rejected():
    with pytest.raises(ValidationError):
        validate(_m({"ranking": {"bogus": 1}}), _schema())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/kirill/speech-spoof-bench/speech-spoof-bench && pytest tests/test_manifest_schema_ranking.py -v`
Expected: FAIL — `ranking`/`color` rejected by `additionalProperties: false`.

- [ ] **Step 3: Edit the schema**

In `src/speech_spoof_bench/schema/manifest.schema.json`:

(a) Add `color` to the tier item `properties` (keep it optional):

```json
        "properties": {
          "name": {"type": "string", "minLength": 1},
          "min_coverage": {"type": "number", "minimum": 0, "maximum": 1},
          "color": {"type": "string", "minLength": 1}
        }
```

(b) Add a `ranking` property to the top-level `properties` object (it is NOT
added to the top-level `required` array — it's optional):

```json
    "ranking": {
      "type": "object",
      "additionalProperties": false,
      "properties": {
        "metric": {"type": "string", "minLength": 1},
        "absence_penalty": {"type": "number"},
        "gamma_aggregated": {"type": "number", "minimum": 0},
        "gamma_pooled": {"type": "number", "minimum": 0},
        "default_view": {"enum": ["aggregated", "pooled"]},
        "weights": {
          "type": "object",
          "additionalProperties": {"type": "number"}
        }
      }
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/kirill/speech-spoof-bench/speech-spoof-bench && pytest tests/test_manifest_schema_ranking.py -v`
Expected: PASS (7).

- [ ] **Step 5: Commit**

```bash
cd /home/kirill/speech-spoof-bench/speech-spoof-bench
git add src/speech_spoof_bench/schema/manifest.schema.json tests/test_manifest_schema_ranking.py
git commit -m "feat(schema): optional ranking block + tier color in manifest.schema.json"
```

---

## Task 7: Add the `ranking` block + tier colors to the live manifest

**Files:**
- Modify: `arena-manifest/manifest.yaml`

- [ ] **Step 1: Read the current manifest**

Run: `cd /home/kirill/speech-spoof-bench/arena-manifest && cat manifest.yaml`
Note exact key order and the existing `tiers:` block.

- [ ] **Step 2: Add tier colors + the ranking block**

Edit `arena-manifest/manifest.yaml`. Add `color` to each tier:

```yaml
tiers:
  - {name: gold,   min_coverage: 1.0, color: "#FFD700"}
  - {name: silver, min_coverage: 0.5, color: "#C0C0C0"}
  - {name: bronze, min_coverage: 0.0, color: "#CD7F32"}
```

And add a top-level `ranking` block (place it after `metrics_in_use`):

```yaml
ranking:
  metric: eer_percent
  absence_penalty: 50.0
  gamma_aggregated: 0.0
  gamma_pooled: 1.0
  default_view: aggregated
```

(No `weights` — all manual_d default to 1.) Preserve other fields/order exactly.

- [ ] **Step 3: Validate against the schema**

Run:
```bash
cd /home/kirill/speech-spoof-bench/speech-spoof-bench && python -c "
import json, yaml
from importlib import resources
from jsonschema import validate
with resources.files('speech_spoof_bench.schema').joinpath('manifest.schema.json').open() as f:
    schema = json.load(f)
data = yaml.safe_load(open('/home/kirill/speech-spoof-bench/arena-manifest/manifest.yaml'))
validate(data, schema)
print('manifest valid; ranking:', data['ranking'])
"
```
Expected: prints the `ranking` block, no error.

- [ ] **Step 4: Commit**

```bash
cd /home/kirill/speech-spoof-bench/arena-manifest
git add manifest.yaml
git commit -m "manifest: tier colors + ranking block (gamma dial, absence penalty)"
```

---

# Phase B — Badges

## Task 8: `arena/badges.py` — tier-color helper

**Files:**
- Create: `arena/badges.py`
- Create: `arena/tests/test_badges.py`

- [ ] **Step 1: Write the failing test**

```python
# arena/tests/test_badges.py
"""Tests for the dynamic tier/rank badge endpoints."""
from __future__ import annotations

import badges

TIERS_WITH_COLOR = [
    {"name": "gold",   "min_coverage": 1.0, "color": "#FFD700"},
    {"name": "silver", "min_coverage": 0.5, "color": "#C0C0C0"},
    {"name": "bronze", "min_coverage": 0.0, "color": "#CD7F32"},
]
TIERS_NO_COLOR = [
    {"name": "gold",   "min_coverage": 1.0},
    {"name": "silver", "min_coverage": 0.5},
    {"name": "bronze", "min_coverage": 0.0},
]


def test_tier_color_uses_manifest_color():
    assert badges._tier_color("silver", TIERS_WITH_COLOR) == "#C0C0C0"


def test_tier_color_palette_fallback_by_position():
    assert badges._tier_color("gold", TIERS_NO_COLOR) == badges._PALETTE[0]
    assert badges._tier_color("silver", TIERS_NO_COLOR) == badges._PALETTE[1]


def test_tier_color_unknown_is_lightgrey():
    assert badges._tier_color("platinum", TIERS_WITH_COLOR) == "lightgrey"


def test_tier_color_palette_wraps():
    many = [{"name": f"t{i}", "min_coverage": 0.0} for i in range(len(badges._PALETTE) + 2)]
    assert badges._tier_color("t0", many) == badges._tier_color(f"t{len(badges._PALETTE)}", many)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/kirill/speech-spoof-bench/arena && pytest tests/test_badges.py -v`
Expected: FAIL — `No module named 'badges'`.

- [ ] **Step 3: Create `arena/badges.py`**

```python
"""Dynamic shields-endpoint badges for arena tier + global rank.

Tiny JSON blobs that shields.io renders into images. No new service — routes on
the existing arena FastAPI app, reading live state via ingest.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)
router = APIRouter()

_PALETTE = ["#FFD700", "#C0C0C0", "#CD7F32", "#4C9AFF", "#6554C0"]
_UNRANKED_COLOR = "lightgrey"
_CACHE_CONTROL = "max-age=300"


def _tier_color(tier_name: str, tiers: list[dict]) -> str:
    """manifest `color` → positional palette → lightgrey."""
    for i, t in enumerate(tiers):
        if t["name"] == tier_name:
            return t.get("color") or _PALETTE[i % len(_PALETTE)]
    return _UNRANKED_COLOR
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/kirill/speech-spoof-bench/arena && pytest tests/test_badges.py -v`
Expected: PASS (4).

- [ ] **Step 5: Commit**

```bash
cd /home/kirill/speech-spoof-bench/arena
git add badges.py tests/test_badges.py
git commit -m "feat(badges): tier-color resolution (manifest → palette → grey)"
```

---

## Task 9: `badges._standing` — tier + global rank from live state

**Files:**
- Modify: `arena/badges.py`
- Modify: `arena/tests/test_badges.py`

- [ ] **Step 1: Write the failing test**

Append to `arena/tests/test_badges.py`:

```python
from datetime import datetime
from schema import ArenaState, Row


def _state(rows, tiers):
    manifest = {
        "tiers": tiers,
        "core_set": [{"id": "org/a", "revision": "abc1234"}],
        "extended": [],
        "metrics_in_use": ["eer_percent"],
        "ranking": {"absence_penalty": 50.0, "default_view": "aggregated"},
    }
    return ArenaState(manifest=manifest, rows=rows, loaded_at=datetime(2026, 1, 1))


def _row(slug, dataset="org/a", eer=10.0, n=100):
    return Row(system_slug=slug, system_name=slug, dataset_id=dataset, revision="r",
               scores={"eer_percent": eer}, reproduction_level="scoring",
               submitted_at="2026-01-01", submission_url="u", n_trials=n)


def test_standing_returns_tier_place_color(monkeypatch):
    monkeypatch.setattr(badges, "_load_state", lambda: _state([_row("sys")], TIERS_WITH_COLOR))
    st = badges._standing("sys")
    assert st["tier"] == "gold"
    assert st["place"] == 1 and st["out_of"] == 1
    assert st["color"] == "#FFD700"


def test_standing_place_is_global(monkeypatch):
    rows = [_row("good", eer=1.0), _row("bad", eer=9.0)]
    monkeypatch.setattr(badges, "_load_state", lambda: _state(rows, TIERS_WITH_COLOR))
    assert badges._standing("bad")["place"] == 2
    assert badges._standing("bad")["out_of"] == 2


def test_standing_absent_slug_none(monkeypatch):
    monkeypatch.setattr(badges, "_load_state", lambda: _state([_row("sys")], TIERS_WITH_COLOR))
    assert badges._standing("ghost") is None


def test_standing_swallows_load_failure(monkeypatch):
    def boom():
        raise RuntimeError("hub down")
    monkeypatch.setattr(badges, "_load_state", boom)
    assert badges._standing("sys") is None


def test_endpoint_payload_shape():
    assert badges._endpoint_payload("arena tier", "gold", "#FFD700") == {
        "schemaVersion": 1, "label": "arena tier", "message": "gold", "color": "#FFD700"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/kirill/speech-spoof-bench/arena && pytest tests/test_badges.py -k "standing or payload" -v`
Expected: FAIL — helpers undefined.

- [ ] **Step 3: Implement**

Append to `arena/badges.py`:

```python
def _load_state():
    """Indirection point so tests can monkeypatch the live-state read."""
    import ingest
    return ingest.load_state()


def _standing(slug: str) -> dict | None:
    """{'tier', 'place', 'out_of', 'color'} for slug, or None if unavailable.
    Tier from assign_tiers; place from global_rank in the manifest default view.
    Swallows load errors → None (endpoint then shows 'unranked')."""
    try:
        state = _load_state()
    except Exception as exc:  # noqa: BLE001
        logger.warning("badge: load_state failed: %s", exc)
        return None
    from ranking import assign_tiers, global_rank, _ranking_config

    manifest = state.manifest
    tiers = manifest.get("tiers", [])
    core = [e["id"] for e in manifest.get("core_set", [])]
    tier_map = assign_tiers(state.rows, tiers, core)
    if slug not in tier_map:
        return None
    view = _ranking_config(manifest)["default_view"]
    rank = global_rank(state.rows, manifest, view).get(slug)
    if rank is None:
        return None
    return {
        "tier": tier_map[slug],
        "place": rank["place"],
        "out_of": rank["out_of"],
        "color": _tier_color(tier_map[slug], tiers),
    }


def _endpoint_payload(label: str, message: str, color: str) -> dict:
    return {"schemaVersion": 1, "label": label, "message": message, "color": color}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/kirill/speech-spoof-bench/arena && pytest tests/test_badges.py -v`
Expected: PASS (9).

- [ ] **Step 5: Commit**

```bash
cd /home/kirill/speech-spoof-bench/arena
git add badges.py tests/test_badges.py
git commit -m "feat(badges): _standing — tier + global rank + color from live state"
```

---

## Task 10: The two routes

**Files:**
- Modify: `arena/badges.py`
- Modify: `arena/tests/test_badges.py`

- [ ] **Step 1: Write the failing test**

Append to `arena/tests/test_badges.py`:

```python
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _client(rows, tiers, monkeypatch):
    monkeypatch.setattr(badges, "_load_state", lambda: _state(rows, tiers))
    app = FastAPI()
    app.include_router(badges.router)
    return TestClient(app)


def test_tier_endpoint(monkeypatch):
    c = _client([_row("sys")], TIERS_WITH_COLOR, monkeypatch)
    r = c.get("/badge/sys/tier.json")
    assert r.status_code == 200
    assert r.json() == {"schemaVersion": 1, "label": "arena tier",
                        "message": "gold", "color": "#FFD700"}
    assert r.headers["cache-control"] == "max-age=300"


def test_rank_endpoint_global_message(monkeypatch):
    rows = [_row("good", eer=1.0), _row("bad", eer=9.0)]
    c = _client(rows, TIERS_WITH_COLOR, monkeypatch)
    r = c.get("/badge/bad/rank.json")
    assert r.status_code == 200
    assert r.json()["label"] == "arena rank"
    assert r.json()["message"] == "#2 of 2"
    assert r.json()["color"] == "#FFD700"
    assert r.headers["cache-control"] == "max-age=300"


def test_unknown_slug_unranked(monkeypatch):
    c = _client([_row("sys")], TIERS_WITH_COLOR, monkeypatch)
    for kind in ("tier", "rank"):
        r = c.get(f"/badge/ghost/{kind}.json")
        assert r.status_code == 200
        assert r.json()["message"] == "unranked"
        assert r.json()["color"] == "lightgrey"


def test_palette_color_without_manifest_color(monkeypatch):
    c = _client([_row("sys")], TIERS_NO_COLOR, monkeypatch)
    assert c.get("/badge/sys/tier.json").json()["color"] == badges._PALETTE[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/kirill/speech-spoof-bench/arena && pytest tests/test_badges.py -k "endpoint or unknown or palette_color" -v`
Expected: FAIL — routes return 404.

- [ ] **Step 3: Implement**

Append to `arena/badges.py`:

```python
def _json(payload: dict) -> JSONResponse:
    return JSONResponse(payload, headers={"Cache-Control": _CACHE_CONTROL})


@router.get("/badge/{slug}/tier.json")
def tier_badge(slug: str) -> JSONResponse:
    st = _standing(slug)
    if st is None:
        return _json(_endpoint_payload("arena tier", "unranked", _UNRANKED_COLOR))
    return _json(_endpoint_payload("arena tier", st["tier"], st["color"]))


@router.get("/badge/{slug}/rank.json")
def rank_badge(slug: str) -> JSONResponse:
    st = _standing(slug)
    if st is None:
        return _json(_endpoint_payload("arena rank", "unranked", _UNRANKED_COLOR))
    message = f"#{st['place']} of {st['out_of']}"
    return _json(_endpoint_payload("arena rank", message, st["color"]))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/kirill/speech-spoof-bench/arena && pytest tests/test_badges.py -v`
Expected: PASS (13).

- [ ] **Step 5: Commit**

```bash
cd /home/kirill/speech-spoof-bench/arena
git add badges.py tests/test_badges.py
git commit -m "feat(badges): /badge/<slug>/tier.json + rank.json routes"
```

---

## Task 11: Mount the badge router in `arena/main.py`

**Files:**
- Modify: `arena/main.py`

- [ ] **Step 1: Add the include next to the webhook router**

In `arena/main.py`, after:

```python
from webhook import router as webhook_router
app.include_router(webhook_router)
```

add:

```python
from badges import router as badge_router
app.include_router(badge_router)
```

(Both must be before `gr.mount_gradio_app(app, demo, path="/")` so `/badge/...`
isn't shadowed.)

- [ ] **Step 2: Verify routes register**

Run:
```bash
cd /home/kirill/speech-spoof-bench/arena && python -c "
import main
paths = {r.path for r in main.app.routes}
assert '/badge/{slug}/tier.json' in paths and '/badge/{slug}/rank.json' in paths, paths
print('badge routes registered OK')
"
```
Expected: `badge routes registered OK`

- [ ] **Step 3: Commit**

```bash
cd /home/kirill/speech-spoof-bench/arena
git add main.py
git commit -m "feat(arena): mount badge router before gradio mount"
```

---

## Task 12: `build_paste_comment` emits tier + rank endpoint badges

**Files:**
- Modify: `speech-spoof-bench/src/speech_spoof_bench/badge.py`
- Modify: `speech-spoof-bench/tests/test_badge_build_paste_comment.py`

- [ ] **Step 1: Write the failing test**

Append to `speech-spoof-bench/tests/test_badge_build_paste_comment.py`:

```python
def test_comment_includes_tier_and_rank_endpoint_badges():
    body = _build()
    host = "speechantispoofingbenchmarks-speechantispoofingarena.hf.space"
    assert f"https://img.shields.io/endpoint?url=https://{host}/badge/aasist/tier.json" in body
    assert f"https://img.shields.io/endpoint?url=https://{host}/badge/aasist/rank.json" in body
    assert body.count("?system=aasist)") >= 3  # eer + tier + rank click targets


def test_endpoint_badge_md_builder():
    md = badge._endpoint_badge_md("arena tier", "aasist", "tier")
    assert md.startswith("[![arena tier](https://img.shields.io/endpoint?url=")
    assert "/badge/aasist/tier.json" in md
    assert md.endswith(f"({badge.ARENA_URL}?system=aasist)")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/kirill/speech-spoof-bench/speech-spoof-bench && pytest tests/test_badge_build_paste_comment.py -k endpoint -v`
Expected: FAIL — `_endpoint_badge_md` undefined / endpoint URLs absent.

- [ ] **Step 3: Implement**

In `src/speech_spoof_bench/badge.py`, add the host constant under `ARENA_URL`:

```python
ARENA_URL = "https://huggingface.co/spaces/SpeechAntiSpoofingBenchmarks/SpeechAntiSpoofingArena"
ARENA_HOST = "speechantispoofingbenchmarks-speechantispoofingarena.hf.space"
```

Add the builder near `_shields_url`:

```python
def _endpoint_badge_md(label: str, slug: str, kind: str) -> str:
    """Markdown for a dynamic shields-endpoint badge (kind = 'tier' | 'rank')
    served by the arena, linking to the system's arena page."""
    endpoint = f"https://{ARENA_HOST}/badge/{slug}/{kind}.json"
    img = f"https://img.shields.io/endpoint?url={endpoint}"
    return f"[![{label}]({img})]({ARENA_URL}?system={slug})"
```

In `build_paste_comment`, replace the README step. Change:

```python
        f"### 2. Add the badge line to your README\n\n"
        f"```markdown\n{badge_md}\n```\n\n"
```

to:

```python
        f"### 2. Add the badge lines to your README\n\n"
        f"```markdown\n{badge_md}\n"
        f"{_endpoint_badge_md('arena tier', slug, 'tier')}\n"
        f"{_endpoint_badge_md('arena rank', slug, 'rank')}\n```\n\n"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/kirill/speech-spoof-bench/speech-spoof-bench && pytest tests/test_badge_build_paste_comment.py -v`
Expected: PASS (existing + 2 new).

- [ ] **Step 5: Commit**

```bash
cd /home/kirill/speech-spoof-bench/speech-spoof-bench
git add src/speech_spoof_bench/badge.py tests/test_badge_build_paste_comment.py
git commit -m "feat(badge): post-merge comment emits tier + rank endpoint badges"
```

---

# Phase C — Overview UI toggle

## Task 13: `app.py` — Aggregated/Pooled toggle + global-rank column

**Files:**
- Modify: `arena/app.py`
- Modify: `arena/tests/test_app_overview.py` (create if absent)

Context: read `arena/app.py` first (esp. `_overview_tables` near line 34 and the
Gradio block that renders the Overview tab). The Overview currently renders one
DataFrame per tier via `overview_table`. This task adds a view selector and a
global-rank column, keeping tier grouping.

- [ ] **Step 1: Write the failing test (pure data layer first)**

Create/append `arena/tests/test_app_overview.py`:

```python
"""The Overview data builder honors the view and exposes a global rank column."""
from __future__ import annotations

from datetime import datetime
import app
from schema import ArenaState, Row


def _row(slug, ds, eer, n=100):
    return Row(system_slug=slug, system_name=slug, dataset_id=ds, revision="r",
               scores={"eer_percent": eer}, reproduction_level="scoring",
               submitted_at="2026-01-01", submission_url="u", n_trials=n)


def _state(rows):
    return ArenaState(
        manifest={
            "tiers": [{"name": "gold", "min_coverage": 1.0},
                      {"name": "silver", "min_coverage": 0.5},
                      {"name": "bronze", "min_coverage": 0.0}],
            "core_set": [{"id": "org/a", "revision": "r"}, {"id": "org/b", "revision": "r"}],
            "extended": [], "metrics_in_use": ["eer_percent"],
            "ranking": {"absence_penalty": 50.0, "default_view": "aggregated"},
        },
        rows=rows, loaded_at=datetime(2026, 1, 1))


def test_overview_rows_have_global_rank_column():
    rows = [_row("g", "org/a", 1.0), _row("g", "org/b", 1.0), _row("s", "org/a", 1.0)]
    tables = app._overview_tables_with_rank(_state(rows), view="aggregated")
    # flatten all tier rows
    allrows = [r for tier in tables for r in tier]
    by_system = {r["system"]: r for r in allrows}
    assert by_system["g"]["rank"] == 1   # full coverage wins
    assert by_system["s"]["rank"] == 2


def test_overview_view_changes_rank_when_trials_differ():
    rows = [_row("x", "org/a", 2.0, 900), _row("x", "org/b", 10.0, 100),
            _row("y", "org/a", 6.0, 900), _row("y", "org/b", 6.0, 100)]
    agg = {r["system"]: r["rank"]
           for t in app._overview_tables_with_rank(_state(rows), view="aggregated") for r in t}
    pooled = {r["system"]: r["rank"]
              for t in app._overview_tables_with_rank(_state(rows), view="pooled") for r in t}
    # x: aggregated mean 6.0 vs y 6.0 (tie, slug breaks); pooled x≈2.8 < y 6.0 → x first
    assert pooled["x"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/kirill/speech-spoof-bench/arena && pytest tests/test_app_overview.py -v`
Expected: FAIL — `_overview_tables_with_rank` undefined.

- [ ] **Step 3: Implement the data builder**

In `arena/app.py`, add (near `_overview_tables`):

```python
def _overview_tables_with_rank(state: ArenaState, view: str) -> list[list[dict]]:
    """Per-tier overview rows (as _overview_tables) plus a global 'rank' key on
    each row, computed by ranking.global_rank in the given view."""
    from ranking import global_rank, assign_tiers

    tiers = state.manifest.get("tiers", [])
    core = [e["id"] for e in state.manifest.get("core_set", [])]
    primary = state.manifest.get("metrics_in_use", ["eer_percent"])[0]
    base_tables = overview_table(state.rows, tiers, core, primary_metric=primary)
    ranks = global_rank(state.rows, state.manifest, view)

    # Map display name → slug to look up rank (overview rows carry system *name*).
    name_to_slug = {r.system_name: r.system_slug for r in state.rows}

    out: list[list[dict]] = []
    for t in tiers:
        rows = base_tables.get(t["name"], [])
        for r in rows:
            slug = name_to_slug.get(r["system"])
            r["rank"] = ranks.get(slug, {}).get("place") if slug else None
        out.append(rows)
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/kirill/speech-spoof-bench/arena && pytest tests/test_app_overview.py -v`
Expected: PASS (2).

- [ ] **Step 5: Commit**

```bash
cd /home/kirill/speech-spoof-bench/arena
git add app.py tests/test_app_overview.py
git commit -m "feat(arena): _overview_tables_with_rank — global rank column per view"
```

---

## Task 14: Wire the Gradio radio to the Overview tab

**Files:**
- Modify: `arena/app.py`

Context: this is Gradio wiring (not unit-tested — verified by import + a smoke
check). Read the Overview tab block in `build_demo()` first. The tab currently
renders `gold, silver, bronze = _overview_tables(state)` into three DataFrames.

- [ ] **Step 1: Add a view selector and re-render callback**

In the Overview tab of `build_demo()`:

(a) Add a radio above the tier tables:

```python
        view = gr.Radio(
            choices=["aggregated", "pooled"],
            value=state.manifest.get("ranking", {}).get("default_view", "aggregated"),
            label="Ranking view",
            info="aggregated = all datasets equal · pooled = weighted by trials",
        )
```

(b) Build the three tier DataFrames from `_overview_tables_with_rank(state, view.value)`
instead of `_overview_tables(state)`, so each table includes the `rank` column.

(c) Wire the radio to re-render. Add a callback that recomputes the three tables
for the chosen view and returns them:

```python
        def _rerender(selected_view):
            st = ingest.load_state()
            tables = _overview_tables_with_rank(st, selected_view)
            # tables is [gold_rows, silver_rows, bronze_rows] aligned to tier order
            return tables[0], tables[1], tables[2]

        view.change(_rerender, inputs=[view], outputs=[gold_df, silver_df, bronze_df])
```

(Match the actual DataFrame component variable names in the file; if they are in
a list, return the list.)

- [ ] **Step 2: Smoke-check the app imports and builds**

Run:
```bash
cd /home/kirill/speech-spoof-bench/arena && python -c "
import app
demo = app.build_demo()
print('demo built OK')
"
```
Expected: `demo built OK` (no exception).

- [ ] **Step 3: Run the full arena suite**

Run: `cd /home/kirill/speech-spoof-bench/arena && pytest -q`
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
cd /home/kirill/speech-spoof-bench/arena
git add app.py
git commit -m "feat(arena): Overview Aggregated/Pooled toggle re-sorts by global rank"
```

---

# Phase D — Deploy + push

## Task 15: Push all three repos

**Files:** none (git push only)

- [ ] **Step 1: Run each suite**

```bash
cd /home/kirill/speech-spoof-bench/arena && pytest -q
cd /home/kirill/speech-spoof-bench/speech-spoof-bench && pytest tests/test_manifest_schema_ranking.py tests/test_badge_build_paste_comment.py -q
```
Expected: all pass.

- [ ] **Step 2: Push speech-spoof-bench (GitHub) first**

```bash
cd /home/kirill/speech-spoof-bench/speech-spoof-bench && git push origin main
```

- [ ] **Step 3: Push arena-manifest (HF dataset)**

```bash
cd /home/kirill/speech-spoof-bench/arena-manifest && git push origin main
```

- [ ] **Step 4: Push arena (HF Space) — triggers rebuild**

```bash
cd /home/kirill/speech-spoof-bench/arena && git push origin main
```

- [ ] **Step 5: Wait for the Space to come back up**

```bash
until curl -fsS -o /dev/null https://speechantispoofingbenchmarks-speechantispoofingarena.hf.space/healthz; do sleep 5; done
echo "Space healthy"
```
Expected: `Space healthy`.

---

## Manual end-to-end verification

Run after Task 15. Not automated — confirms the live system.

- [ ] **M1.** Live JSON endpoints:

```bash
curl -s https://speechantispoofingbenchmarks-speechantispoofingarena.hf.space/badge/random-baseline/tier.json
curl -s https://speechantispoofingbenchmarks-speechantispoofingarena.hf.space/badge/random-baseline/rank.json
curl -sI https://speechantispoofingbenchmarks-speechantispoofingarena.hf.space/badge/random-baseline/tier.json | grep -i cache-control
```
Expected: tier.json → `message":"gold"`, `color":"#FFD700"`; rank.json → `message":"#1 of 1"`; `cache-control: max-age=300`.

- [ ] **M2.** Unknown slug → `200`, `unranked`:

```bash
curl -s https://speechantispoofingbenchmarks-speechantispoofingarena.hf.space/badge/nonexistent/tier.json
```

- [ ] **M3.** Rendered shields images resolve:

```bash
curl -s -o /dev/null -w "%{http_code} %{content_type}\n" \
  "https://img.shields.io/endpoint?url=https://speechantispoofingbenchmarks-speechantispoofingarena.hf.space/badge/random-baseline/tier.json"
```
Expected: `200 image/svg+xml`. Repeat for `rank.json`.

- [ ] **M4.** Add the tier + rank badge lines to the `random-baseline-asas` model
  README (next to the EER badge from Phase 9), then confirm all three render and
  click through to the Arena. Run from the speech-spoof-bench env:

```python
from huggingface_hub import HfApi, hf_hub_download
from pathlib import Path
import sys
sys.path.insert(0, "/home/kirill/speech-spoof-bench/speech-spoof-bench/src")
from speech_spoof_bench import badge

api = HfApi()
slug = "random-baseline"
tier_md = badge._endpoint_badge_md("arena tier", slug, "tier")
rank_md = badge._endpoint_badge_md("arena rank", slug, "rank")
p = hf_hub_download("SpeechAntiSpoofingBenchmarks/random-baseline-asas",
                    filename="README.md", repo_type="model")
text = Path(p).read_text()
lines = text.splitlines()
for i, ln in enumerate(lines):
    if ln.startswith("[![EER"):
        lines[i] = ln + "\n" + tier_md + "\n" + rank_md
        break
api.upload_file(path_or_fileobj=("\n".join(lines) + "\n").encode(),
                path_in_repo="README.md",
                repo_id="SpeechAntiSpoofingBenchmarks/random-baseline-asas",
                repo_type="model", commit_message="Add arena tier + rank badges")
print("README updated")
```

- [ ] **M5.** Open the Arena Overview tab; flip the `[Aggregated | Pooled]` radio.
  With one system the order is unchanged but the rank column renders and no error
  appears. (Re-check after a second system exists.)

- [ ] **M6.** Genericity (no code edit): in `arena-manifest/manifest.yaml`, change
  `gamma_aggregated`/`gamma_pooled` or add a new tier with a `color`, commit +
  push, wait for the Space to refresh, and confirm the ranking/badge colors
  change with no code change. (Revert afterward if it was only a probe.)

---

## Self-review notes

- **Spec coverage:** formula (Tasks 4–5), config + defaults (Task 3), manifest schema (Task 6) + live config (Task 7), `Row.n_trials` plumbing (Tasks 1–2), badges (Tasks 8–12), Overview toggle (Tasks 13–14), deploy (Task 15), manual M1–M6. `absence_penalty=None` exclusion path covered (Task 4 `test_penalty_none_excludes_absent_dataset`). Phase D (per-attack) intentionally absent.
- **Placeholder scan:** every code/test step is concrete. The two Gradio-wiring steps (Task 14) name the exact components to match in-file because Gradio layout isn't unit-testable; the data layer it depends on (Task 13) is fully tested.
- **Type consistency:** `Row.n_trials: int` (Task 1) read by `_core_n_trials` (Task 4); `_ranking_config` keys (Task 3) consumed by `global_scores` (Task 4) and `_standing`/`global_rank` (Tasks 5, 9); `global_rank` returns `{place, out_of}` (Task 5) consumed by `_standing` (Task 9) and routes (Task 10) and `_overview_tables_with_rank` (Task 13); `_endpoint_badge_md(label, slug, kind)` (Task 12) matches its M4 use; `_tier_color`/`_PALETTE`/`_endpoint_payload` consistent across Tasks 8–10.
- **Cross-repo discipline:** every commit/push `cd`s into the right sub-repo; push order GitHub → manifest → Space.
