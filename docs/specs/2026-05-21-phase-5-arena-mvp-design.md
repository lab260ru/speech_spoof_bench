# Phase 5 — Arena MVP Design

**Status**: Approved (brainstorming)
**Date**: 2026-05-21
**Scope**: ROADMAP Phase 5. Build the read-only Gradio leaderboard Space that consumes `arena-manifest` and each dataset repo's `submissions/*.yaml`.
**Out of scope**: system-detail page, paper rendering, Submit tab, Docker, webhook, `cache.json` persistence, CI/CD — all deferred to Phases 7–10.

References:
- `docs/roadmap/PLAN.md` §3 (Arena Space)
- `docs/roadmap/ROADMAP.md` Phase 5
- `docs/specs/2026-05-21-phase-4-manifest-design.md`

---

## 1. Goal

Stand up a Gradio Space at `huggingface.co/spaces/SpeechAntiSpoofingBenchmarks/SpeechAntiSpoofingArena` that:

- Reads `manifest.yaml` from `SpeechAntiSpoofingBenchmarks/arena-manifest`.
- For each dataset in `core_set ∪ extended`, lists `submissions/*.yaml` and fetches each.
- Validates submissions, skips those missing the `reproduction:` block (per PLAN §3.3).
- Renders three tabs (Overview, Per dataset, About) with manual Refresh and a 30-min in-memory TTL cache.
- Cold-starts in <15 s on a free CPU Space with one dataset + one submission.

At launch the leaderboard contains exactly one row: the random baseline on `ASVspoof2019_LA`.

---

## 2. Repo layout

Local working copy lives at `/home/kirill/speech-spoof-bench/arena/`. The HF Space repo already has `README.md` with SDK metadata; we push the rest as a git remote.

```
arena/
├── README.md                  # exists; HF Space SDK frontmatter
├── app.py                     # Gradio Blocks entrypoint
├── ingest.py                  # fetch manifest + submissions → ArenaState
├── ranking.py                 # tier assignment + table builders (pure)
├── schema.py                  # Row, ArenaState dataclasses; schema loader
├── requirements.txt           # gradio, huggingface_hub, pyyaml, jsonschema,
│                              # speech-spoof-bench @ git+https://...
└── tests/
    ├── test_ranking.py
    ├── test_ingest.py
    └── fixtures/
        ├── manifest.yaml
        └── submissions/
            ├── valid.yaml
            ├── missing-reproduction.yaml
            └── schema-invalid.yaml
```

`speech-spoof-bench` is pinned via git+sha in `requirements.txt` until it's published. The arena uses two things from it: the JSON Schema at `speech_spoof_bench.schema.submission_schema_path()` (or equivalent) and `jsonschema.validate` against it. Nothing else.

---

## 3. Data flow

```
cold start  |  Refresh click  |  TTL expired (30 min)
        │
        ▼
ingest.load_state(force_refresh=False)
        │
        ├─► _fetch_manifest()
        │     hf_hub_download("SpeechAntiSpoofingBenchmarks/arena-manifest",
        │                     "manifest.yaml", repo_type="dataset")
        │     yaml.safe_load → dict with: ranking_version, schema_version,
        │                                  metrics_in_use, tiers, core_set, extended
        │
        ▼
   for each dataset_id in core_set ∪ extended:
        │
        ├─► HfApi.list_repo_files(dataset_id, repo_type="dataset")  # at main
        │      filter: starts with "submissions/", ends with ".yaml",
        │              not "results_template.yaml" or "README.md"
        │
        └─► for each submission path:
                hf_hub_download(dataset_id, path, repo_type="dataset")
                yaml.safe_load
                jsonschema.validate(sub, submission_schema)
                    → invalid: warnings.append(...); continue
                if "reproduction" not in sub:
                    → warnings.append(...); continue       # PLAN §3.3
                rows.append(_to_row(sub, dataset_id, path))
        ▼
ArenaState(
    manifest=manifest_dict,
    rows=[Row, ...],
    loaded_at=datetime.utcnow(),
    warnings=[{dataset, path, reason}, ...],
)
        │
        ▼
served to all tabs until TTL or Refresh
```

`submission_url` on `Row` is constructed as
`https://huggingface.co/datasets/<dataset_id>/blob/main/<path>`
for a "view source" link in the Per-dataset tab.

---

## 4. Modules

### 4.1 `schema.py`

```python
@dataclass(frozen=True)
class Row:
    system_slug: str
    system_name: str
    dataset_id: str
    revision: str
    scores: dict[str, float]      # metric_id -> value
    reproduction_level: str        # "scoring" | "inference"
    submitted_at: str
    submission_url: str

@dataclass(frozen=True)
class Warning:
    dataset_id: str
    path: str
    reason: str

@dataclass(frozen=True)
class ArenaState:
    manifest: dict
    rows: list[Row]
    loaded_at: datetime
    warnings: list[Warning]

def load_submission_schema() -> dict:
    """Load JSON Schema bundled with speech-spoof-bench."""
```

### 4.2 `ingest.py`

Public:

```python
def load_state(force_refresh: bool = False) -> ArenaState: ...
```

Private:

- `_fetch_manifest() -> dict`
- `_list_submission_files(api: HfApi, dataset_id: str) -> list[str]`
- `_fetch_submission(dataset_id: str, path: str) -> dict`
- `_to_row(sub: dict, dataset_id: str, path: str) -> Row`

Module-level state:

```python
_state: ArenaState | None = None
_loaded_at: float | None = None
_lock = threading.Lock()
_TTL_SECONDS = 30 * 60
```

`load_state` checks `(time.monotonic() - _loaded_at) < _TTL_SECONDS` outside the lock; if stale or forced, acquires the lock and re-fetches. Double-checked locking — concurrent callers wait once.

### 4.3 `ranking.py`

Pure functions, no I/O, no module state:

```python
def assign_tiers(
    rows: list[Row],
    tiers: list[dict],         # from manifest, ordered highest-first
    core_set_ids: list[str],
) -> dict[str, str]:
    """system_slug -> tier_name (highest matching tier)."""

def overview_table(
    rows: list[Row],
    tiers: list[dict],
    core_set_ids: list[str],
    primary_metric: str,
) -> dict[str, list[dict]]:
    """tier_name -> list of row-dicts for that tier's DataFrame."""

def per_dataset_table(
    rows: list[Row],
    dataset_id: str,
) -> list[dict]:
    """One row per submitting system; columns derived from metric ids present."""
```

Ranking semantics (per PLAN §3.7):

- Coverage for a system = (# Core datasets the system has rows on) / |Core|.
- A system's tier = highest tier whose `min_coverage ≤ coverage`.
- Within a tier: per (Core dataset, primary metric) rank submitting systems, mean rank = mean over cells the system has.
- Tie-break: more covered → mean rank on Extended → earlier `submitted_at`.
- Missing data is `None`, never imputed.

Primary metric (launch) is `metrics_in_use[0]` from the manifest = `eer_percent`. The function signature passes it in so it stays generic.

### 4.4 `app.py`

```python
with gr.Blocks(title="Speech Anti-Spoofing Arena") as demo:
    state = gr.State()
    last_refreshed = gr.Markdown()
    refresh_btn = gr.Button("🔄 Refresh")

    with gr.Tabs():
        with gr.Tab("Overview"):
            overview_widgets = build_overview_tab(state)
        with gr.Tab("Per dataset"):
            per_dataset_widgets = build_per_dataset_tab(state)
        with gr.Tab("About"):
            about_widgets = build_about_tab(state)

    demo.load(fn=_initial_load, outputs=[state, last_refreshed, ...])
    refresh_btn.click(fn=_force_refresh, outputs=[state, last_refreshed, ...])
```

`_initial_load` calls `ingest.load_state()`; `_force_refresh` calls `ingest.load_state(force_refresh=True)`.

---

## 5. UI tabs

### 5.1 Overview

- Header: `Last refreshed: <iso timestamp>` + Refresh button (shared across tabs).
- One `gr.DataFrame` per tier, top-down (gold → silver → bronze). Tier heading + a hint line ("Coverage ≥ 100%" etc.).
- Columns: `tier`, `system`, `coverage` (e.g. `1/1`), `mean rank`, `repro` (`✔` scoring / `★` inference), then one column per Core dataset showing its primary-metric value (or `—`).
- Lower tiers exclude systems already shown above (a gold system doesn't reappear in silver).

### 5.2 Per dataset

- `gr.Dropdown` of `core_set ∪ extended` dataset ids (default = first Core).
- `gr.DataFrame` below: rows = systems that submitted to the selected dataset.
- Columns dynamically built from the union of metric ids present in those submissions, in `metrics_in_use` order (others appended), plus `system`, `reproduction`, `submitted_at`, `submission` (markdown link to `submission_url`).

### 5.3 About

- Schema version, ranking version, manifest revision (commit sha of the manifest fetch).
- `metrics_in_use` from the manifest.
- Last refreshed timestamp.
- Skipped-submission warnings: a table of `dataset / path / reason` from the most recent ingest. Empty state: "No warnings."

No system-detail page, no Submit tab, no paper rendering — Phase 10.

---

## 6. Error handling

| Failure | Behavior |
|---|---|
| Manifest fetch fails | Render error banner on every tab with exception message. Cache an empty `ArenaState` for 60 s to avoid hammering HF. Refresh retries. |
| `list_repo_files` fails for one dataset | That dataset contributes zero rows; logged in `warnings`. Other datasets still load. |
| Submission YAML fails to parse | Skipped; warning recorded. |
| Submission fails JSON Schema validation | Skipped; warning recorded with first validation error message. |
| Submission missing `reproduction:` block | Skipped; warning recorded ("missing reproduction block"). |
| Unknown metric id in `scores:` | Kept as-is. The arena renders whatever ids it sees; metric registration is the package's concern. |
| Concurrent Refresh clicks | `threading.Lock` around the fetch; second caller sees the result the first one populated. |

---

## 7. Testing

`tests/test_ranking.py` — unit tests with fixture `Row` lists:

- Full coverage on Core → gold.
- Coverage = 0.5 with two Core datasets → silver.
- Coverage = 0 → bronze (single submission edge case).
- Tie-break by Extended mean rank, then by `submitted_at`.
- Missing-data cell remains `None` (not 0, not imputed).
- Empty input → empty tables, no exception.

`tests/test_ingest.py` — fixture YAMLs in `tests/fixtures/`:

- Valid submission → `Row` built correctly (slug, scores, level, url).
- Missing `reproduction:` → skipped, warning recorded.
- Schema-invalid (e.g. missing `system.name`) → skipped, warning recorded.
- `HfApi.list_repo_files` and `hf_hub_download` monkeypatched to read from fixtures — no network in CI.
- TTL respected: second call within 30 min returns cached state without re-fetching (count monkeypatched calls).
- `force_refresh=True` bypasses TTL.

End-to-end HF round-trip is **not** in CI — that's the Phase 6 smoke test (manual).

---

## 8. Done-when (Phase 5 exit criteria)

Mirrors ROADMAP Phase 5 + PLAN §3.8:

- [ ] `arena/` repo pushes to `huggingface.co/spaces/SpeechAntiSpoofingBenchmarks/SpeechAntiSpoofingArena`.
- [ ] Space cold-starts in <15 s with seeded data (1 dataset, 1 submission).
- [ ] Overview tab shows the random baseline row in the gold tier on LA.
- [ ] Per dataset tab dropdown contains `ASVspoof2019_LA`; selecting it shows the submission.
- [ ] About tab shows schema/ranking versions, metrics in use, last-refreshed timestamp, empty warnings table.
- [ ] Refresh button forces re-fetch and updates the timestamp.
- [ ] Hand-injecting a malformed `submissions/broken.yaml` (then refreshing) surfaces it under About → Warnings without crashing the Space.
- [ ] `pytest tests/` is green; ranking unit tests cover full/partial/single/ties; ingest tests cover skip cases.

---

## 9. Open items (intentionally deferred)

| Item | Phase |
|---|---|
| System-detail page, paper/BibTeX rendering | 10 |
| Submit tab | 10 |
| Docker Space | 8a |
| `cache.json` committed-back persistence | 8b |
| HF webhook → auto-refresh | 8c |
| GitHub Actions CI / `verify-pr` | 8d–8e |
| `nightly-revalidate` | 8f |
| `inference` ★ badge on existing rows | already supported by display; populated via Phase 8 |
