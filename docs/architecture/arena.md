# The Arena (HF Docker Space)

The Arena is the public face: a leaderboard at
`https://huggingface.co/spaces/SpeechAntiSpoofingBenchmarks/SpeechAntiSpoofingArena`. It
is a **single FastAPI process** (`main.py`) that:

- mounts the Gradio leaderboard UI at `/`,
- includes the **webhook** router at `/webhook` (→ [cicd.md](cicd.md)),
- includes the **badge** router at `/badge/...` (→ [badges.md](badges.md)),
- serves `/healthz`.

It is a **Docker** Space (`python:3.11-slim`, `git` installed for HF clones, runs
`uvicorn main:app --host 0.0.0.0 --port 7860`). It depends on the pip package, pinned to
an exact commit in `requirements.txt` (currently `…@bde5949…`). The Arena code itself is
**untagged** — it ships from whatever is on the Space's `main`.

## Module map

| Module | Role |
|--------|------|
| `main.py` | FastAPI host; startup cache hydration + background refresh; mounts everything. |
| `app.py` | Gradio UI: builds the tabs, renders tier tables, wires events. |
| `ingest.py` | Fetch manifest + all submissions from HF, build the in-memory `ArenaState`. |
| `ranking.py` | The ranking maths: tiers, weighted global scores, board rank. |
| `leaderboard.py` | Pure DataFrame/HTML builders (no I/O). |
| `charts.py` | Plain-dict chart data (size scatter, SOTA-over-time). |
| `schema.py` | Frozen dataclasses: `Row`, `Warning`, `ArenaState`. |
| `events.py` | Build the activity feed from rows + the manifest CHANGELOG. |
| `changelog.py` | Fetch the optional `CHANGELOG.yaml` from `arena-manifest`. |
| `cache_store.py` | Serialise `ArenaState` to `cache.json` and commit it back to the Space. |
| `docs_fetch.py` | Fetch the submit guides from the package repo at the pinned SHA. |
| `badges.py`, `webhook.py` | The two FastAPI routers (documented in badges.md / cicd.md). |

## The data model — `schema.py`

- **`Row`** — one (system, dataset) result: `system_slug`, `system_name`, `dataset_id`,
  `revision`, `scores` dict, `n_trials` (test-set size; `0` = unknown legacy),
  `reproduction_level` (`scoring` or `inference`), `submitted_at`, `reproduced_at`,
  URLs, and paper/params metadata.
- **`Warning`** — `(dataset_id, path, reason)` for a submission that couldn't be ingested.
- **`ArenaState`** — `manifest`, `rows`, `loaded_at`, `warnings`. This is the whole world,
  held in memory.

## Ingest — `ingest.py`

`load_state(force_refresh=False)` returns the cached `ArenaState` or rebuilds it:

1. `fetch_manifest()` (from the package) → tiers, core_set, extended, ranking config.
2. For every dataset id (core + extended): list `submissions/*.yaml`, fetch & schema-parse
   each.
3. **Filter:** a submission is only shown if it has a `reproduction.match` of `scoring`
   (✔) or `inference` (★). Missing/unverified → skipped and recorded as a `Warning`
   (surfaced in the UI, not fatal — one bad file never sinks the others).
4. `_to_row()` flattens each into a `Row`.

Thread-safe via a lock. **TTL: 30 min** on success, **60 s** on failure (so a transient HF
outage retries soon but a healthy board isn't rebuilt constantly). The functions
`_fetch_manifest`, `_list_submission_files`, `_fetch_submission_dict` are indirection
points so tests can monkeypatch them — don't inline them.

`hydrate(state)` seeds `_state`/`_loaded_at` from a pre-loaded `cache.json` at startup,
so the UI renders instantly while a background refresh runs.

## The ranking algorithm — `ranking.py`

This is the heart of the leaderboard. Three pieces:

### Tier assignment — `assign_tiers(rows, tiers, core_set_ids)`
For each system, `coverage = (# core datasets it was tested on) / |core|`. It is placed in
the **highest** tier (manifest order, highest first) whose `min_coverage <= coverage`. If
a tier has `requires_paper: true` and the system has no paper, that tier is **skipped** —
the system falls through to the first tier it qualifies for (in practice the unranked
`unpublished` tier).

### Global score — `global_scores(rows, manifest, view)`
A weighted mean over the Core set:

```
score(s) = Σ_d ( base_d(γ) · manual_d · value(s,d) )  /  Σ_d ( base_d(γ) · manual_d )

  base_d(γ) = (n_trials_d)^γ / max_over_core( (n_trials)^γ )
  value(s,d) = the metric on dataset d, or `absence_penalty` if s wasn't tested on d
  manual_d  = optional per-dataset weight from manifest.ranking.weights (default 1.0)
```

The **γ (gamma) dial** controls how much test-set size matters:

- `γ = 0` → every Core dataset weighted equally (`base_d = 1`) — the **aggregated**
  (macro-average) view. This is the default.
- `γ = 1` → datasets weighted by `n_trials` — the **pooled** (micro-average) view.
- `0 < γ < 1` → dampened in between.

The manifest provides both `gamma_aggregated` (default `0.0`) and `gamma_pooled`
(default `1.0`); the UI's view toggle picks which one to use.

The **absence penalty** (default `50.0` for EER) substitutes for datasets a system didn't
run — so partial coverage hurts your mean, nudging contributors to cover the whole Core
set. (If a metric has *no* known penalty, absent datasets are dropped from both numerator
and denominator instead.)

### Board rank — `global_rank(rows, manifest, view)`
Sorts systems by `global_scores` in the metric's direction (`lower_is_better` from the
metric registry; unknown metrics assume `True`), ties broken by slug. Returns
`{slug: {place, out_of}}`.

**Paper-gating:** if *any* tier has `requires_paper: true`, `global_rank` first filters
to systems that have a paper. Paperless systems get **no rank at all** — their tier table
shows no Rank column (`tier_ranks = {}`). So never assume every row has a `place`.

### Worked micro-example
3 Core datasets, γ=0 (aggregated):

- System A tested on 3/3 → coverage 1.0 → **gold** (has paper).
- System B tested on 2/3 → coverage 0.67 → **silver** (has paper). Its mean includes one
  `absence_penalty=50.0` term, so it ranks below A.
- System D tested on 1/3, **no paper** → skips gold/silver/bronze (all `requires_paper`)
  → **unpublished**, unranked.

## The UI — `app.py`

Seven tabs: **Overview** (one ranked table per tier, with the γ view toggle),
**Datasets**, **Per dataset** (ranked within a single dataset by its primary metric),
**By model size** (params-vs-score scatter, log-x), **Over time** (SOTA timeline /
activity feed), **Submit** (the guides fetched live from GitHub), **About**.

- Tier labels are in a `_TIER_LABELS` dict; tables are rendered as **sticky-column HTML**
  by `leaderboard.render_html_table(df, pinned=2)` (Rank + System frozen left;
  `_PINNED_WIDTHS = [44, 150]` — update these if those columns change width).
- A detail strip shows paper link, BibTeX, checkpoint, params, reproduction level.
- Metric columns are **dynamic** — any metric id present in submissions becomes a column.

## Persistence & cold-start — `cache_store.py`

`cache.json` (schema `1`) is the serialised `ArenaState`, committed back into the Space's
own repo so a cold container renders immediately.

- `save_and_commit(state, reason)` serialises, computes a **content hash that excludes
  `loaded_at` and sorts rows/warnings** (so timestamp-only or order-only changes don't
  trigger a commit), writes locally, then commits to HF via `upload_file` **only if** the
  hash changed *and* it's been >30 s since the last commit (debounce).
- Commit uses `SPACE_COMMIT_TOKEN`; message `cache refresh (<reason>)`.
- **Self-loop guard:** the webhook ignores `repo_type == "space"` events, so the Arena's
  own cache commits don't trigger another refresh. Without this it would loop forever.

If `SPACE_COMMIT_TOKEN` is unset, the Arena still works but keeps state in memory only —
nothing survives a restart.

`cache.json` sketch:
```json
{
  "schema_version": 1,
  "loaded_at": "2026-05-28T23:06:17Z",
  "manifest": { "ranking_version": "v2", "tiers": [...], "core_set": [...], "ranking": {...} },
  "rows": [ { "system_slug": "random-baseline", "dataset_id": "...ASVspoof2019_LA",
              "scores": {"eer_percent": 49.87}, "reproduction_level": "scoring",
              "n_trials": 71237, ... } ],
  "warnings": []
}
```

## Activity feed — `events.py` + `changelog.py`

`build_events(rows, manifest, changelog)` derives events from the data — `model_added`
(🆕) per submission, `verification_upgraded` (⬆️) when a row reaches `inference` — and
merges them with the curated `CHANGELOG.yaml` fetched from `arena-manifest` (events like
`dataset_added` ➕, `dataset_repin` ↻, `metric_added` 📏, `note` ✍️). The changelog is
**optional**: if missing/unreachable, the feed is auto-generated from rows only.

## Submit guides — `docs_fetch.py`

The Submit tab is rendered live from the package repo so docs and code stay in lockstep:
`get_doc("submit-model", ref)` fetches
`raw.githubusercontent.com/lab260ru/speech_spoof_bench/<ref>/docs/submitting/submit-model.md`.
`resolve_pin()` reads the `speech-spoof-bench @ git+...@<sha>` line out of
`requirements.txt` so the guides shown match the *exact package version the Space runs*.
Network failure → a fallback message linking to GitHub.
</content>
