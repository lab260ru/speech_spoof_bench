# Phase 10 — Arena polish (frontend overhaul)

**Status:** design approved (brainstorm 2026-05-29)
**Scope:** Arena Space frontend, plus one additive field in the pip package and two new repo docs. **All CI/CD, webhook, badge, manifest, and submission-verification infrastructure is preserved unchanged.**

This implements §3.2 of the v4 spec (the "Arena polish" row of `ROADMAP.md` Phase 10), with adjustments agreed during brainstorming. Phase 9 (badge layer) is already complete; this builds only on the *read/render* side of the Arena.

---

## 1. Goals & non-goals

### Goals
1. Turn the Overview and Per-dataset tables into an **MTEB-style leaderboard**: Rank + System pinned (frozen) on the left, metric columns scrolling horizontally, sortable, searchable, with show/hide-column toggles.
2. Make **every model and every dataset clickable** (model → checkpoint repo; dataset → HF dataset repo).
3. Surface each system's **paper link, BibTeX (copy-able), model link, and description** inline — *without* a per-system page.
4. New **📐 By model size** tab: performance vs. declared parameter count.
5. New **📈 Over time** tab: best-score-over-time chart + an activity event log.
6. New **📤 Submit** tab that **renders canonical markdown docs from the `speech-spoof-bench` repo** (single source of truth), including a "can't run it yourself — write us" callout.
7. A **totals header** (N systems · N datasets · last refreshed) visible across tabs.

### Non-goals (explicitly out)
- **No per-system detail page** (the user dropped §3.2's "System detail" tab). Inline detail strip replaces it.
- No change to ranking math (`global_scores`, tiers), webhook, CI workflows, badge generation, or submission verification flow.
- No new persisted state in the Space beyond the existing `cache.json` (the over-time event log reads existing data + a manifest-side changelog; it does not require the Space to accumulate history).
- No auto-derivation of model param counts from checkpoints — params are submitter-declared.

---

## 2. Tab structure (approved)

`🏆 Overview · 📊 Per dataset · 📐 By model size · 📈 Over time · 📤 Submit · ℹ️ About`

A shared **totals header** renders above the tab bar on load/refresh:
`N systems · N datasets · Last refreshed <ts>Z` (replaces the standalone last-refreshed line; About keeps its detailed version).

---

## 3. Data plumbing (the enabling change)

### 3.1 New/changed fields carried into the Arena

`schema.Row` gains fields that already exist in the submission YAML but are currently dropped at ingest:

```python
@dataclass(frozen=True)
class Row:
    system_slug: str
    system_name: str
    dataset_id: str
    revision: str
    scores: dict[str, float]
    reproduction_level: str
    submitted_at: str
    submission_url: str
    n_trials: int = 0
    # --- new in Phase 10 ---
    description: str = ""
    code_url: str = ""               # system.code        (GitHub)
    checkpoint_url: str = ""         # system.checkpoint  (HF model repo)
    paper_arxiv_id: str = ""         # system.paper.arxiv_id
    paper_url: str = ""              # system.paper.url
    paper_bibtex: str = ""           # system.paper.bibtex
    params_millions: float | None = None   # system.params_millions (new schema field)
    reproduced_at: str = ""          # reproduction.reproduced_at (for verification-upgrade events)
```

`ingest._to_row` is extended to populate these from the fetched submission dict. All fields are defaulted so legacy `cache.json` rows (which lack them) still deserialize — same backward-compat discipline already used for `n_trials`.

### 3.2 Submission schema change (pip package — additive, optional)

In `src/speech_spoof_bench/schema/submission.schema.json`, add to `system.properties`:

```json
"params_millions": {"type": "number", "minimum": 0}
```

- **Optional** (not added to `system.required`), so every existing submission stays valid and all CI/badge/nightly flows are unaffected.
- `additionalProperties: false` on `system` means it MUST be declared in `properties` (done above) to be accepted.

`submit` CLI (`§2.5a`) gains an optional `--params-millions <float>` flag (and an interactive prompt when omitted) that writes `system.params_millions` into the generated YAML. `validate-submission` accepts it via the schema with no extra code. `result.yaml`/badge output is **not** changed (params are a leaderboard-display concern, not a badge concern).

### 3.3 Module layout (refactor for isolation)

`app.py` is currently a single file mixing Gradio wiring and table assembly. Phase 10 splits rendering into focused units so each tab is independently understandable and testable:

```
arena/
├── app.py              # Gradio Blocks wiring only: tabs, state, refresh, event handlers
├── ingest.py           # (extended) carries new Row fields
├── schema.py           # (extended) Row fields above
├── ranking.py          # (unchanged math) + new pure helpers: size_series(), sota_timeline()
├── events.py           # NEW: build the activity-log event list (hybrid source, §7)
├── docs_fetch.py       # NEW: fetch+cache the Submit-tab markdown from the repo (§8)
├── charts.py           # NEW: pure data→(dataframe|figure) builders for scatter & timeline
├── ui/
│   ├── leaderboard.py  # NEW: MTEB-style table builder (gradio_leaderboard config + detail strip)
│   ├── overview.py
│   ├── per_dataset.py
│   ├── by_size.py
│   ├── over_time.py
│   ├── submit.py
│   └── about.py
└── tests/              # + tests per new pure module
```

Pure data transforms (`ranking.py`, `charts.py`, `events.py`) stay free of Gradio imports so they're unit-testable; `ui/*` and `app.py` hold the Gradio-specific wiring.

---

## 4. Overview & Per-dataset — MTEB-style leaderboard

### 4.1 Component
Use the **`gradio_leaderboard`** component (the same one MTEB uses) — added to `arena/requirements.txt`. It provides, out of the box: sortable columns, a search box, show/hide-column selection, and pinned columns.

**Pinned:** `Rank`, `System`. **Scrollable:** `Mean`, then one column per Core dataset (primary metric), then a compact `📄 ⧉` links column.

**Fallback contingency:** if `gradio_leaderboard` row-selection (for the detail strip, §4.3) proves unavailable in the pinned version, the detail strip is driven by a `gr.Dropdown(system)` instead of row-click. If the component itself is unworkable on the Space, fall back to native `gr.Dataframe(pinned_columns=["Rank","System"])` (Gradio 6.14 supports it) — sortable + pinned + cell links, minus search/column-toggles. This fallback is a known, acceptable degradation.

### 4.2 Clickable models & datasets
- **System** cell renders as a markdown link to `checkpoint_url`. Verification badge (`★`/`✔`) appended.
- **Datasets** render as a **clickable chip row above the table** (each → `https://huggingface.co/datasets/<id>`), because Gradio table *headers* cannot be links. Column headers show short dataset names; the chip row is the link affordance.

### 4.3 Detail strip (replaces the per-system page — variant A)
Selecting a row (or choosing from the system dropdown fallback) reveals a strip **below the same table** containing:
- description + `params_millions` (if present) + verification level,
- links: 📄 `paper_url` (arXiv id), 🔗 `checkpoint_url`, 📥 `scores_url`,
- a `gr.Code` block with `paper_bibtex` (Gradio's code box has a native copy button).

The strip is part of the same tab — no routing, no separate page.

### 4.4 Tiers preserved
Gold/Silver/Bronze grouping is kept (one leaderboard section per tier, as today). The `rank` column is the board-wide `global_rank` place (view-aware), as currently computed in `app._overview_tables_with_rank`.

### 4.5 Per-dataset tab
Same leaderboard component, fed by `per_dataset_table` (all metric ids as columns). Dataset chosen via the existing dropdown; chip row + detail strip behave as in Overview.

---

## 5. View toggle (shared semantics)
The existing `aggregated` / `pooled` radio (gamma 0 vs gamma 1) stays on Overview and is **mirrored on the By-size tab** so the "Core mean" axis there equals the Overview score exactly (both call `ranking.global_scores(view)`).

---

## 6. 📐 By model size tab

- **X:** `params_millions` (log scale). Systems lacking a declared param count are listed in a "no size declared" note below the chart, not plotted.
- **Y:** selectable. Default **"Core mean"** = `global_scores(rows, manifest, view)` (view-aware, identical to Overview ranking). Dropdown can switch to any single dataset → that dataset's raw primary-metric value (no aggregation).
- **Marker style by coverage:** filled = full Core coverage; hollow = partial (so penalized/excluded aggregates don't silently mislead). Tooltip shows system, value, params, coverage.
- **Below the chart:** a sortable table (`System · params · value`), models clickable.
- Chart built by `charts.size_series(rows, manifest, view, y_selector)` → returned to a `gr.Plot` (matplotlib/plotly — whichever is already available; plotly preferred for hover).

---

## 7. 📈 Over time tab

### 7.1 Performance chart
- A **best-score-over-time step line** per selected dataset: for each submission date, the best (lowest, for EER) primary-metric value achieved up to that date. Built by `charts.sota_timeline(rows, dataset_id)` purely from `submitted_at` + scores.
- **Dataset re-pin markers:** vertical dashed lines where the dataset's pinned `revision` changed (from the changelog, §7.2), annotated "scores across a re-pin aren't strictly comparable."
- Dataset selector defaults to the first Core dataset.

### 7.2 Activity event log — hybrid source (approved option A)
`events.build_events(rows, manifest, changelog)` merges two sources into one date-sorted feed:

1. **Auto-derived** (free, always current, no new files):
   - 🆕 *model added* — one per `Row`, dated `submitted_at`, text `"<system> added on <dataset> — <metric> <value> <badge>"`.
   - ⬆️ *verification upgraded* — when `reproduction_level == "inference"`, dated `reproduced_at`.
2. **Curated** — a new `CHANGELOG.yaml` in the **`arena-manifest`** repo, read alongside `manifest.yaml`, for infra events the Arena can't infer:
   - ↻ dataset re-pinned, ➕ dataset added to Core/Extended, 📏 metric/tier/ranking-version change, ✍️ free-form notes (e.g. "ran <model> on full set on request").

`CHANGELOG.yaml` shape:
```yaml
events:
  - {date: 2026-05-22, type: dataset_repin, dataset: SpeechAntiSpoofingBenchmarks/ASVspoof2021_DF, text: "Re-pinned to revision abc1234"}
  - {date: 2026-05-20, type: metric_added,  text: "min_tdcf added to metrics_in_use"}
  - {date: 2026-05-18, type: note,          text: "Ran XLSR-big on full Core Set on submitter request"}
```
`type` is free-form (drives the emoji); unknown types get a default bullet. Missing file → feed is auto-events only (graceful). The re-pin markers in §7.1 are derived from `type: dataset_repin` entries.

---

## 8. 📤 Submit tab — renders repo docs

### 8.1 Canonical docs (single source of truth)
Two new markdown files in the **`speech-spoof-bench` repo**:
- `docs/submitting/submit-model.md` — wrap model → run → upload scores → declare params → `submit` → verification → paste badge.
- `docs/submitting/submit-dataset.md` — redistribution/license → `scaffold-dataset` → build parquet → frontmatter/eval.yaml/citation → `validate-dataset` → push + manifest PR → Core vs Extended.

These are the authoritative contribution docs (also linked from the package README, satisfying the "info in repo docs too" requirement).

### 8.2 How the Arena gets them
`docs_fetch.get_doc(name)` fetches the raw markdown from GitHub **at the same commit sha the Space pins** for the `speech-spoof-bench` package in `arena/requirements.txt` (so docs and pinned package logic never disagree), caches it for the session, and on network failure falls back to a short bundled stub + a link to the GitHub docs. The Submit tab is a `gr.Markdown` per guide behind a sub-toggle.

### 8.3 "Write us" callout
A static highlighted callout at the top of the Submit tab (above both guides):
> 💡 **Can't run the full benchmark yourself?** If you can run your model over the complete dataset(s), [write to us](mailto:<maintainer-contact>) and we'll consider running your model for you.

(Requirement #4.) Contact = maintainer contact already used elsewhere.

---

## 9. About tab
Unchanged in content; gains nothing beyond the shared totals header existing above all tabs.

---

## 10. What stays untouched (preservation guarantee)
- `webhook.py`, all GitHub Actions workflows, `badges.py`/badge generation, `cache_store.py` persistence mechanism, `manifest.py`, submission/validation/`reproduce` logic, the submission verification workflow, ranking math (`assign_tiers`, `global_scores`, `global_rank`).
- The only package change is the **optional** `params_millions` schema field + `submit` flag. The only manifest-repo change is the **optional** `CHANGELOG.yaml`.

---

## 11. Delivery slices (staged, with manual verification between each)

1. **Slice 1 — Data + MTEB leaderboard.** Row fields, ingest, `params_millions` schema/`submit` flag, `gradio_leaderboard` Overview + Per-dataset (pinned/scroll/sort/search), clickable models + dataset chips, detail strip with copy-able BibTeX, totals header. *Verify on the Space: existing random-baseline row renders, links work, BibTeX copies.*
2. **Slice 2 — Submit tab + repo docs.** `submit-model.md`, `submit-dataset.md`, `docs_fetch`, callout. *Verify: tab renders the docs at the pinned sha; fallback works offline.*
3. **Slice 3 — By model size.** `charts.size_series`, scatter + table, view mirror, coverage markers. *Verify: random baseline plots (or shows in "no size declared" until a params value exists).*
4. **Slice 4 — Over time.** `charts.sota_timeline`, `events.build_events`, `CHANGELOG.yaml` in arena-manifest, re-pin markers. *Verify: timeline + feed render from current data + a seeded changelog entry.*

Each slice ships independently; the Arena remains usable after each.

---

## 12. Risks / open implementation notes
- **`gradio_leaderboard` capabilities** (row-select for detail strip; pinned + search together) must be confirmed against the version installable on the Space — fallbacks specified in §4.1.
- **Plot library** on the Space (plotly vs matplotlib) — pick whichever the pinned Gradio pulls in; prefer plotly for hover tooltips.
- **Sparse data at launch:** until real models land (Phase 12), By-size and Over-time tabs are thin (one baseline). Acceptable — the structure is what Phase 10 delivers.
- **`docs_fetch` sha resolution:** the Space must know its own pinned `speech-spoof-bench` sha; read it from `requirements.txt` at runtime (it's in the image) rather than hardcoding.
