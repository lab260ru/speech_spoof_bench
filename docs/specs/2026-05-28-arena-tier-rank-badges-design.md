# Arena tier & rank badges (design)

> **SUPERSEDED** by `2026-05-28-arena-ranking-framework-design.md`. The endpoint
> / comment / README mechanics here are reused, but the **rank source** changed
> from within-tier to a global, manifest-configured formula. Read the framework
> spec instead.

**Status**: superseded
**Builds on**: Phase 9 badge layer (`docs/specs/2026-05-28-phase-9-badge-design.md`)

## Goal

Let a submitter display two **dynamic** badges on their HF model page, alongside
the existing static EER badge:

- **tier** — the system's arena tier (gold/silver/bronze/…)
- **rank** — the system's place within that tier (`#1 of 1`)

Both must stay correct as the leaderboard changes. Rank is competitive (changes
when anyone submits), so it cannot be a static badge baked at merge time — it is
served live by the arena.

## Why dynamic, not static

- **Tier** depends only on the system's own coverage (`|core covered| / |core|`),
  so it is intrinsic and would survive as a static badge — until the system adds
  datasets or the manifest's Core Set changes.
- **Rank** is the system's mean-rank position vs every other system; it changes
  whenever any submission lands. A static `#1` badge would silently become wrong.

Therefore both badges read live arena state. This is **not a new service** — the
arena is an already-running FastAPI app (it hosts `/webhook` and the Gradio UI).
We add two JSON routes; shields.io renders the images server-side (no CORS, no
new infrastructure).

## Architecture

New router `arena/badges.py`, mounted in `arena/main.py` alongside the webhook
router and **before** the Gradio mount (so the routes aren't shadowed by `/`):

```
GET /badge/<slug>/tier.json
GET /badge/<slug>/rank.json
```

Each returns shields.io **endpoint** JSON:

```json
{"schemaVersion": 1, "label": "arena tier", "message": "gold", "color": "#FFD700"}
{"schemaVersion": 1, "label": "arena rank", "message": "#1 of 1", "color": "#FFD700"}
```

README usage (shields endpoint syntax):

```markdown
[![arena tier](https://img.shields.io/endpoint?url=https://speechantispoofingbenchmarks-speechantispoofingarena.hf.space/badge/random-baseline/tier.json)](<arena>?system=random-baseline)
[![arena rank](https://img.shields.io/endpoint?url=https://speechantispoofingbenchmarks-speechantispoofingarena.hf.space/badge/random-baseline/rank.json)](<arena>?system=random-baseline)
```

Both endpoints read the live leaderboard via `ingest.load_state()` (same 30-min
TTL the UI uses) and set `Cache-Control: max-age=300` so shields/CDN don't hammer
the Space.

`<arena>` host: `https://speechantispoofingbenchmarks-speechantispoofingarena.hf.space`
(the lowercased Space hostname; the badge module's `ARENA_URL` is the
`huggingface.co/spaces/...` web URL — both forms appear, web URL for the click
target, hostname for the shields endpoint fetch).

## Genericity (first-class requirement)

The whole feature is data-driven so adding a tier later is a **manifest-only**
change with no code edit.

- **Tiers**: `ranking.assign_tiers` and the new `ranking.system_standing`
  iterate `manifest["tiers"]` — any number, any names. Nothing hardcodes
  "gold/silver/bronze" or a count.
- **Tier colors live in the manifest.** Each tier entry gets an optional
  `color`:

  ```yaml
  tiers:
    - {name: gold,     min_coverage: 1.0, color: "#FFD700"}
    - {name: silver,   min_coverage: 0.5, color: "#C0C0C0"}
    - {name: bronze,   min_coverage: 0.0, color: "#CD7F32"}
    # later — no code change:
    - {name: platinum, min_coverage: 1.0, color: "#E5E4E2"}
  ```

  The endpoint reads `tier["color"]`. If absent (older manifests), it falls back
  to a **positional palette** (brightest first, by tier order); final fallback
  `lightgrey`. `manifest.schema.json` gains `color` as an **optional** field —
  fully backward-compatible.
- **Rank badge inherits the tier's color** — one color source, zero
  rank-specific magic numbers. The `#3 of 12` message conveys position; matching
  the tier color makes the two badges read as a pair and stays generic for any
  tier set.
- **Primary metric** comes from `manifest["metrics_in_use"][0]`, exactly as the
  UI already derives it.

**Known limitation (out of scope, flagged):** ranking direction is
lower-is-better, inherited from the existing `overview_table`/`_mean_rank`.
Making rank direction per-metric (`lower_is_better` from the metric registry)
would require changing the existing ranking code as well — a separate change,
not part of this badge work.

## Components

### `arena/ranking.py` — `system_standing` (new, pure)

```python
def system_standing(rows, tiers, core_set_ids, primary_metric, slug) -> dict | None:
    """Return {'tier': str, 'place': int, 'out_of': int} for `slug`, or None if
    the slug has no rows on the board.

    - tier: from assign_tiers(rows, tiers, core_set_ids)
    - place: 1-based position of `slug` among its tier-mates, ordered by the same
      mean-rank rule the Overview tab uses (ascending mean of primary-metric
      values across covered core datasets; ties broken by slug for determinism)
    - out_of: number of systems in that tier
    """
```

Pure (no I/O). Reuses `assign_tiers` and mirrors `_mean_rank` ordering for
consistency with the Overview table.

### `arena/badges.py` — FastAPI router (new)

```python
router = APIRouter()

def _tier_color(tier_name, tiers) -> str: ...   # manifest color → palette → lightgrey
def _standing(slug) -> dict | None: ...          # load_state + system_standing
def _endpoint_payload(label, message, color) -> dict: ...

@router.get("/badge/{slug}/tier.json")
def tier_badge(slug): ...

@router.get("/badge/{slug}/rank.json")
def rank_badge(slug): ...
```

- Reads `ingest.load_state()`; derives `tiers`, `core_set_ids`,
  `primary_metric` from the manifest exactly as `app.py` does.
- Both responses carry `Cache-Control: max-age=300`.

### `arena/main.py` — mount

Add `from badges import router as badge_router; app.include_router(badge_router)`
next to the webhook include, before `gr.mount_gradio_app`.

### `arena-manifest` repo — `manifest.yaml`

Add `color` to the three existing tier entries (gold/silver/bronze hexes above)
so the badges use the intended semantic colors rather than the palette fallback.
The arena re-reads the manifest on its next refresh; no arena redeploy needed for
a manifest-only color tweak. (A color-less manifest still works via the palette
fallback — this step is what turns the fallback into the intended colors.)

### `speech-spoof-bench` package — `badge.build_paste_comment`

Extend the post-merge comment so the pasted snippet includes **all three** badge
lines (EER static + tier dynamic + rank dynamic). The two dynamic lines use the
shields endpoint URLs above, built from the Space hostname and the system slug.
Add a helper `_endpoint_badge_md(label, slug, kind)` and append the two lines to
the README section of the comment template.

## Badge content

| Endpoint | label | message | color |
|---|---|---|---|
| tier.json | `arena tier` | tier name (`gold`) | manifest color → palette → `lightgrey` |
| rank.json | `arena rank` | `#<place> of <out_of>` | same as the system's tier color |

### Edge cases

| Case | Behavior |
|---|---|
| Slug not on the board (typo / not yet merged) | `200` with grey `unranked` badge (both endpoints). Never `404` — a 404 renders as a broken "inaccessible" image in the README. |
| `load_state()` raises | Same grey `unranked` fallback; logged. |
| Manifest tier lacks `color` | Positional palette by tier order; final fallback `lightgrey`. |
| System in a tier of size 1 | `#1 of 1` — correct. |

## Testing

### Unit — `arena/tests/test_ranking.py` (extend)

- `system_standing`: single system → `{tier, place:1, out_of:1}`.
- Multi-system ordering: lower primary metric ranks first.
- Slug absent → `None`.
- Multi-tier: a system's place is within its own tier, not global.
- Tie-break determinism (equal mean rank → by slug).

### Endpoint — `arena/tests/test_badges.py` (new)

- `tier.json` shape + correct tier + manifest color used.
- Palette fallback when a tier has no `color`.
- `rank.json` shape + `#k of n` message + color matches tier.
- Unknown slug → `200`, message `unranked`, grey.
- `Cache-Control: max-age=300` present on both.
- Routes resolve when mounted (not shadowed by Gradio): covered by including the
  router on a bare `FastAPI()` in the test.

### Schema — `speech-spoof-bench/tests`

- `manifest.schema.json` still validates a tier entry **with** and **without**
  `color`.

### Comment — `speech-spoof-bench/tests/test_badge_build_paste_comment.py`

- Snapshot updated: the comment's README section now contains the two
  shields-endpoint badge lines (tier + rank) in addition to the EER line.

## Manual end-to-end verification

Run after deploying the arena change and updating the model README.

- [ ] **M1.** `curl https://…hf.space/badge/random-baseline/tier.json` and
  `/rank.json` → assert shields-endpoint JSON shape, `tier == "gold"`,
  `message == "#1 of 1"`, and `Cache-Control: max-age=300`.
- [ ] **M2.** `curl https://…hf.space/badge/nonexistent/tier.json` → `200`,
  `message == "unranked"`, grey color.
- [ ] **M3.** Fetch the two rendered shields images
  (`https://img.shields.io/endpoint?url=…tier.json` and `…rank.json`) → HTTP 200,
  `content-type: image/svg+xml`, SVG contains the tier name / rank text.
- [ ] **M4.** Add the two badge lines to the `random-baseline-asas` README
  (next to the EER badge). Visit `https://huggingface.co/SpeechAntiSpoofingBenchmarks/random-baseline-asas`
  → both badges render and click through to the Arena.
- [ ] **M5.** (Genericity) Against a local manifest fixture, add a `platinum`
  tier with `color: "#E5E4E2"` and a system covering it; hit `tier.json` → the
  badge color equals the manifest value, proving a new tier needs no code change.

## Out of scope

- Per-metric ranking direction (`lower_is_better`) — see Known limitation.
- A combined single badge (`gold · #1`) — two separate badges chosen for clarity.
- Caching beyond the existing `load_state` TTL + `Cache-Control`.
- Backfilling dynamic badges into already-merged submissions other than the
  random baseline (manual M4 covers the one live system).
