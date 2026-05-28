# Arena ranking framework (design)

**Status**: design, awaiting plan
**Supersedes the ranking model in**: `docs/specs/2026-05-28-arena-tier-rank-badges-design.md`
(the within-tier rank). That badge spec's endpoints/comment/README work is
reused; only the *rank source* changes from within-tier to global.
**Builds on**: Phase 9 badge layer (`docs/specs/2026-05-28-phase-9-badge-design.md`).

## Goal

Give the arena a single, configurable, data-driven ranking formula that produces
a **global** place for each system, where covering more benchmarks raises a
model's place, and where the pooled-vs-aggregated choice is one tunable dial. The
tier/rank badges and the Overview tab render whatever the formula produces.

Concretely the framework must:
- Reward coverage — a model that runs more Core datasets ranks higher.
- Express pooled and aggregated as the same formula with a different weight dial.
- Be tunable from the manifest (no code change to re-rank).
- Keep the arena lightweight — never download raw `scores.txt`; rank only from the
  per-dataset metric values already in submission YAMLs.

## The formula

For a system *s* over the **full Core Set** *D* (every core dataset, covered or
not):

```
score(s) = ( Σ_d  w_d · value(s, d) ) / ( Σ_d  w_d )

  w_d        = base_d(γ) · manual_d
  base_d(γ)  = (n_trials_d) ** γ  /  max_core( (n_trials) ** γ )      ∈ (0, 1]
  value(s,d) = metric value of s on d   if s covered d
             = absence_penalty          otherwise
```

- **γ** — the compression dial:
  - `γ = 0` → every `base_d = 1` → **aggregated** (macro average, all datasets
    equal).
  - `γ = 1` → `base_d ∝ n_trials_d` → **pooled** (trial-weighted, micro).
  - `0 < γ < 1` → dampened (e.g. `0.5` = sqrt: big datasets count more, not
    linearly).
- **manual_d** — optional per-dataset multiplier from the manifest; default `1`.
- **n_trials_d** — a per-dataset constant (the test-set size). Read from any
  submission to *d* (all submissions to the same dataset share it). Used even for
  absent datasets' penalty terms. If a core dataset has **zero** submissions,
  `n_trials_d` defaults to `1` (so γ has no effect for that dataset).
- **value** — the system's primary-metric value, or `absence_penalty` for a core
  dataset the system did not submit to.
- **Sort direction** — from the metric's `lower_is_better` (EER → lower score
  ranks first).

### Why this satisfies the requirements

- **More benchmarks → higher place**: an uncovered core dataset injects
  `absence_penalty` (a bad value) into the mean. Running it for real replaces the
  penalty with a (better) score, improving `score(s)` → higher place.
- **Pooled vs aggregated**: identical formula, different γ. The Overview toggle
  flips γ between the aggregated preset and the pooled preset.
- **Tunable**: γ, `absence_penalty`, and `manual_d` are all manifest config and
  all default to neutral, so an unconfigured manifest yields a plain macro
  average over the Core Set.
- **Lightweight**: only per-dataset metric values + `n_trials` (both already in
  the submission YAML) are needed — no raw-score fetching.
- **Display never lies**: per-dataset tables still show blanks for absent
  datasets. The penalty feeds only the ranking score, not any displayed cell.

## Configuration (lives in `arena-manifest/manifest.yaml`)

Config values live in the manifest repo (single, code-free source of truth, same
place as `tiers`, `core_set`, `metrics_in_use`). The schema that validates the
shape lives in the `speech-spoof-bench` package. The arena reads the manifest at
runtime via `ingest`.

New **optional** `ranking` block:

```yaml
ranking:
  metric: eer_percent           # defaults to metrics_in_use[0] when omitted
  absence_penalty: 50.0         # value used for core datasets a system didn't cover
  gamma_aggregated: 0.0         # γ for the "Aggregated" view
  gamma_pooled: 1.0             # γ for the "Pooled" view
  default_view: aggregated      # "aggregated" | "pooled" — badges + initial Overview
  weights:                      # optional manual per-dataset multipliers
    SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA: 1.0
```

Defaults when the block (or any field) is absent:
- `metric` → `metrics_in_use[0]`
- `absence_penalty` → a worst-case constant for the metric (`50.0` for
  `eer_percent`; for other metrics the operator must set it explicitly, else the
  framework logs a warning and excludes absent datasets from that system's mean —
  documented below)
- `gamma_aggregated` → `0.0`, `gamma_pooled` → `1.0`
- `default_view` → `aggregated`
- `weights` → empty (all manual_d = 1)

Tuning the ranking is a manifest commit + `ranking_version` bump + tagged release
(per the existing manifest DoD §4); no arena redeploy.

`absence_penalty` fallback: if the metric is not `eer_percent` and no
`absence_penalty` is configured, there is no safe worst-case to inject. In that
case the framework **excludes** absent datasets from that system's mean (reverts
to covered-only) and records a warning surfaced on the About tab, so the operator
knows coverage is not being rewarded until they set a penalty.

## Architecture / components

### `arena/ranking.py` — new pure functions

```python
def global_scores(rows, manifest, view: str) -> dict[str, float]:
    """system_slug -> score(s) for the given view ('aggregated' | 'pooled').
    Pure; reads tiers/core_set/ranking config + metric direction from `manifest`
    and the registered metric. view selects gamma_aggregated vs gamma_pooled."""

def global_rank(rows, manifest, view: str) -> dict[str, dict]:
    """system_slug -> {'place': int, 'out_of': int}. Sorts global_scores by the
    metric's direction; ties broken by slug. out_of = number of systems on the
    board (all rows, not per tier)."""
```

- Metric direction (`lower_is_better`) comes from the package's metric registry
  via `speech_spoof_bench.metrics.get_metric(metric_id)`.
- `n_trials_d` is read from `Row` data. **Row gains `n_trials`** (see schema
  change below) so the formula can weight by it; currently `ingest._to_row`
  strips `n_trials`/`n_skipped` out of `scores`.
- `assign_tiers` is unchanged — tiers remain the coverage bands.

### `arena/schema.py` + `arena/ingest.py` — carry `n_trials`

`Row` gains `n_trials: int`. `ingest._to_row` reads `sub["scores"]["n_trials"]`
(it currently discards it). `cache.json` round-trips the new field (the
content-hash debounce in `cache_store` must include it). Backward-compat: a
cached row without `n_trials` hydrates with `n_trials = 0`, treated as "unknown"
→ contributes to the `n_trials_d = 1` fallback.

### `arena/badges.py` — rank badge reads `global_rank`

The prior badge plan was not executed, so Phase B implements the badge code
fresh against this framework. The rank badge's `place`/`out_of` come from
`global_rank(rows, manifest, default_view)` (not within-tier ranking); the tier
badge's tier comes from `assign_tiers`. Endpoint paths, payload shape, caching,
and the unknown-slug `unranked` fallback are exactly as the prior badge spec
described. The `2026-05-28-arena-tier-rank-badges` design/plan are superseded by
this document.

### `arena/app.py` — Overview toggle + global-rank column

- A `[Aggregated | Pooled]` radio on the Overview tab. Default = `default_view`.
- Selecting a view re-sorts the Overview by `global_rank(..., view)` and shows a
  global-rank column. Tier grouping/tables remain.

### `speech-spoof-bench` package — `manifest.schema.json`

Add the optional `ranking` object (and the optional tier `color` carried over
from the badge spec). All additive and optional → backward-compatible. `ranking`
field types: `metric` string; `absence_penalty` number; `gamma_aggregated` /
`gamma_pooled` numbers ≥ 0; `default_view` enum `["aggregated","pooled"]`;
`weights` object of `string → number`.

### `speech-spoof-bench` package — `badge.build_paste_comment`

Unchanged from the badge spec: the comment emits three README badge lines (EER
static + tier + rank endpoint badges). The rank endpoint now reflects global
rank, but the comment template is identical.

## Phasing

One spec; the plan builds in dependency order.

- **Phase A — formula core.** `global_scores`/`global_rank` in `ranking.py`;
  `Row.n_trials` + ingest/cache plumbing; `ranking` block in `manifest.yaml`;
  `manifest.schema.json` optional `ranking` + tier `color`. Unit-tested, no UI.
- **Phase B — badges.** Tier + rank badge endpoints, `build_paste_comment` lines,
  model README. Rank badge reads `global_rank`. Reuses the
  `2026-05-28-arena-tier-rank-badges` plan; only the rank source differs.
- **Phase C — Overview UI.** `[Aggregated | Pooled]` radio + global-rank column.
- **Phase D — deferred, out of scope.** Per-attack data + richer formula inputs;
  requires a dataset-schema change (no attack/condition column today).

## Testing

### Formula — `arena/tests/test_ranking.py` (extend)

- `global_scores`: γ=0 (aggregated) vs γ=1 (pooled) produce different orderings
  when datasets differ in `n_trials`.
- Absence penalty: a system covering more core datasets scores better than one
  covering fewer with otherwise-equal metric values.
- `manual_d`: a manual weight shifts a dataset's contribution.
- `global_rank`: places are global (across all systems), ties broken by slug,
  `out_of` = total systems.
- Direction: a `lower_is_better=False` metric ranks higher values first.
- Edge: single-system board → place 1 of 1; core dataset with zero submissions →
  `n_trials_d` defaults to 1, no error.
- `absence_penalty` unset on a non-EER metric → absent datasets excluded +
  warning recorded.

### Row / ingest — `arena/tests/test_ingest.py` (extend)

- `_to_row` populates `n_trials` from the submission.
- Cache round-trip preserves `n_trials`; legacy cached row (no field) hydrates to
  `n_trials = 0`.

### Badges — `arena/tests/test_badges.py` (extend)

- Rank badge message reflects `global_rank` in `default_view`.
- Unknown slug → `unranked` (unchanged).
- Tier color from manifest/palette (unchanged).

### Schema — `speech-spoof-bench/tests`

- Manifest valid **with** and **without** a `ranking` block.
- Invalid `default_view`, negative γ, non-number `absence_penalty` rejected.

### Comment — `speech-spoof-bench/tests/test_badge_build_paste_comment.py`

- Snapshot still includes the tier + rank endpoint badge lines.

### Manual end-to-end (M1–M6)

- [ ] **M1.** Live JSON endpoints return correct tier + global rank for
  `random-baseline`; `Cache-Control: max-age=300`.
- [ ] **M2.** Unknown slug → `200`, `unranked`.
- [ ] **M3.** Rendered shields images (tier + rank) resolve to `image/svg+xml`.
- [ ] **M4.** Model README shows EER + tier + rank badges, all click through to
  the Arena.
- [ ] **M5.** Overview `[Aggregated | Pooled]` toggle re-sorts the board; with one
  system the order is unchanged but the column renders.
- [ ] **M6.** Genericity: in a local manifest fixture, change γ and add a new
  tier with a `color`; confirm ranking + badge color change with **no code
  edit**.

## Out of scope

- Per-attack / per-condition aggregation and any dataset-schema change (Phase D).
- Exact pooled EER via raw-trial concatenation (the trial-weighted mean at γ=1 is
  the chosen approximation; the arena never downloads `scores.txt`).
- Per-metric ranking direction beyond the registry's `lower_is_better`.
- Multiple simultaneous primary metrics in one ranking (single `metric` per the
  `ranking` block).
