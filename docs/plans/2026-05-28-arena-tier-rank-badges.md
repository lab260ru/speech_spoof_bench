# Arena Tier & Rank Badges Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve two dynamic shields-endpoint badges (arena tier + within-tier rank) from the arena, generic over any manifest-defined tier set, and wire them into the post-merge comment + the random-baseline model page.

**Architecture:** A pure `ranking.system_standing` computes `{tier, place, out_of}` for a slug. A new `arena/badges.py` FastAPI router exposes `/badge/<slug>/tier.json` and `/rank.json` returning shields-endpoint JSON, reading live state via `ingest.load_state()`. Tier colors come from the manifest (optional `color`, positional-palette fallback). The speech-spoof-bench package's `build_paste_comment` gains the two endpoint badge lines.

**Tech Stack:** Python 3.10+, FastAPI, `huggingface_hub`, `jsonschema`, `pyyaml`, `pytest`, shields.io endpoint badges, HF Docker Space, GitHub Actions.

**Spec:** `docs/specs/2026-05-28-arena-tier-rank-badges-design.md`

**Repos touched** (each has its own `.git` — always `cd` before `git`):
- `arena/` — ranking function, badges router, main.py mount (pushes to HF Space)
- `speech-spoof-bench/` — manifest.schema.json (optional `color`), badge.py comment lines (pushes to GitHub)
- `arena-manifest/` — manifest.yaml tier colors (pushes to HF dataset)

**Arena Space hostname** (for shields endpoint URLs): `speechantispoofingbenchmarks-speechantispoofingarena.hf.space`

---

## Slice 1 — `ranking.system_standing` (pure, arena repo)

### Task 1: Add `system_standing` to `arena/ranking.py`

**Files:**
- Modify: `arena/ranking.py`
- Modify: `arena/tests/test_ranking.py`

- [ ] **Step 1: Write the failing tests**

Append to `arena/tests/test_ranking.py`:

```python
from ranking import system_standing


def test_system_standing_single_system_gold():
    rows = [_row("sys", "org/a"), _row("sys", "org/b")]
    st = system_standing(rows, TIERS, CORE, "eer_percent", "sys")
    assert st == {"tier": "gold", "place": 1, "out_of": 1}


def test_system_standing_orders_by_mean_metric_ascending():
    # two gold systems; lower mean eer ranks first
    rows = [
        Row(system_slug="good", system_name="good", dataset_id="org/a", revision="r",
            scores={"eer_percent": 1.0}, reproduction_level="scoring",
            submitted_at="2026-01-01", submission_url="u"),
        Row(system_slug="good", system_name="good", dataset_id="org/b", revision="r",
            scores={"eer_percent": 1.0}, reproduction_level="scoring",
            submitted_at="2026-01-01", submission_url="u"),
        Row(system_slug="bad", system_name="bad", dataset_id="org/a", revision="r",
            scores={"eer_percent": 9.0}, reproduction_level="scoring",
            submitted_at="2026-01-01", submission_url="u"),
        Row(system_slug="bad", system_name="bad", dataset_id="org/b", revision="r",
            scores={"eer_percent": 9.0}, reproduction_level="scoring",
            submitted_at="2026-01-01", submission_url="u"),
    ]
    assert system_standing(rows, TIERS, CORE, "eer_percent", "good") == {
        "tier": "gold", "place": 1, "out_of": 2}
    assert system_standing(rows, TIERS, CORE, "eer_percent", "bad") == {
        "tier": "gold", "place": 2, "out_of": 2}


def test_system_standing_place_is_within_tier_not_global():
    rows = [
        _row("g", "org/a"), _row("g", "org/b"),   # gold
        _row("s", "org/a"),                          # silver
    ]
    assert system_standing(rows, TIERS, CORE, "eer_percent", "s") == {
        "tier": "silver", "place": 1, "out_of": 1}


def test_system_standing_absent_slug_returns_none():
    rows = [_row("sys", "org/a")]
    assert system_standing(rows, TIERS, CORE, "eer_percent", "ghost") is None


def test_system_standing_tie_broken_by_slug_deterministically():
    rows = [
        _row("bbb", "org/a"), _row("bbb", "org/b"),
        _row("aaa", "org/a"), _row("aaa", "org/b"),
    ]
    # equal mean rank (both 10.0) → alphabetical by slug: aaa first
    assert system_standing(rows, TIERS, CORE, "eer_percent", "aaa")["place"] == 1
    assert system_standing(rows, TIERS, CORE, "eer_percent", "bbb")["place"] == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/kirill/speech-spoof-bench/arena && pytest tests/test_ranking.py -k system_standing -v`
Expected: FAIL — `cannot import name 'system_standing'`.

- [ ] **Step 3: Implement `system_standing`**

Append to `arena/ranking.py` (after `_mean_rank`):

```python
def system_standing(
    rows: Iterable[Row],
    tiers: list[dict],
    core_set_ids: list[str],
    primary_metric: str,
    slug: str,
) -> dict | None:
    """Return {'tier', 'place', 'out_of'} for `slug`, or None if it has no rows.

    place is 1-based within the system's tier, ordered by ascending mean of the
    primary metric across covered core datasets (the same order the Overview tab
    uses); ties broken by slug for determinism.
    """
    rows = list(rows)
    tier_map = assign_tiers(rows, tiers, core_set_ids)
    if slug not in tier_map:
        return None
    tier_name = tier_map[slug]

    # Mean primary-metric value per system, over covered core datasets.
    by_system: dict[str, dict[str, Row]] = defaultdict(dict)
    for r in rows:
        by_system[r.system_slug][r.dataset_id] = r

    def mean_metric(s: str) -> float:
        vals = [
            by_system[s][ds].scores.get(primary_metric)
            for ds in core_set_ids
            if ds in by_system[s] and by_system[s][ds].scores.get(primary_metric) is not None
        ]
        return sum(vals) / len(vals) if vals else float("inf")

    tier_mates = [s for s, t in tier_map.items() if t == tier_name]
    tier_mates.sort(key=lambda s: (mean_metric(s), s))
    return {
        "tier": tier_name,
        "place": tier_mates.index(slug) + 1,
        "out_of": len(tier_mates),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/kirill/speech-spoof-bench/arena && pytest tests/test_ranking.py -v`
Expected: PASS (all existing + 5 new).

- [ ] **Step 5: Commit**

```bash
cd /home/kirill/speech-spoof-bench/arena
git add ranking.py tests/test_ranking.py
git commit -m "feat(ranking): system_standing — tier + within-tier place for a slug"
```

---

## Slice 2 — `arena/badges.py` router

### Task 2: Tier-color resolution helper

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


def test_tier_color_uses_manifest_color_when_present():
    assert badges._tier_color("silver", TIERS_WITH_COLOR) == "#C0C0C0"


def test_tier_color_palette_fallback_by_position():
    # No color in manifest → positional palette; first tier gets palette[0].
    c0 = badges._tier_color("gold", TIERS_NO_COLOR)
    c1 = badges._tier_color("silver", TIERS_NO_COLOR)
    assert c0 == badges._PALETTE[0]
    assert c1 == badges._PALETTE[1]
    assert c0 != c1


def test_tier_color_unknown_tier_is_lightgrey():
    assert badges._tier_color("platinum", TIERS_WITH_COLOR) == "lightgrey"


def test_tier_color_palette_wraps_when_more_tiers_than_palette():
    many = [{"name": f"t{i}", "min_coverage": 0.0} for i in range(len(badges._PALETTE) + 2)]
    # index 0 and len(palette) wrap to the same color — no IndexError.
    assert badges._tier_color("t0", many) == badges._tier_color(
        f"t{len(badges._PALETTE)}", many)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/kirill/speech-spoof-bench/arena && pytest tests/test_badges.py -v`
Expected: FAIL — `No module named 'badges'`.

- [ ] **Step 3: Create `arena/badges.py` with the color helper**

```python
"""Dynamic shields-endpoint badges for arena tier + within-tier rank.

Serves tiny JSON blobs that shields.io renders into images. No new service —
routes on the existing arena FastAPI app, reading live state via ingest.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)
router = APIRouter()

# Positional fallback palette (used only when a tier has no manifest `color`).
# Ordered brightest-first so the top tier reads as "best" by default.
_PALETTE = ["#FFD700", "#C0C0C0", "#CD7F32", "#4C9AFF", "#6554C0"]

_UNRANKED_COLOR = "lightgrey"
_CACHE_CONTROL = "max-age=300"


def _tier_color(tier_name: str, tiers: list[dict]) -> str:
    """Color for a tier: manifest `color` → positional palette → lightgrey."""
    for i, t in enumerate(tiers):
        if t["name"] == tier_name:
            if t.get("color"):
                return t["color"]
            return _PALETTE[i % len(_PALETTE)]
    return _UNRANKED_COLOR
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/kirill/speech-spoof-bench/arena && pytest tests/test_badges.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
cd /home/kirill/speech-spoof-bench/arena
git add badges.py tests/test_badges.py
git commit -m "feat(badges): tier-color resolution (manifest → palette → grey)"
```

---

### Task 3: `_standing` + `_endpoint_payload` helpers

**Files:**
- Modify: `arena/badges.py`
- Modify: `arena/tests/test_badges.py`

- [ ] **Step 1: Write the failing test**

Append to `arena/tests/test_badges.py`:

```python
from unittest.mock import MagicMock

from schema import ArenaState, Row
from datetime import datetime


def _state(rows, tiers):
    manifest = {
        "tiers": tiers,
        "core_set": [{"id": "org/a", "revision": "abc1234"}],
        "extended": [],
        "metrics_in_use": ["eer_percent"],
    }
    return ArenaState(manifest=manifest, rows=rows, loaded_at=datetime(2026, 1, 1))


def _row(slug, dataset="org/a", eer=10.0):
    return Row(system_slug=slug, system_name=slug, dataset_id=dataset, revision="r",
               scores={"eer_percent": eer}, reproduction_level="scoring",
               submitted_at="2026-01-01", submission_url="u")


def test_standing_reads_live_state(monkeypatch):
    st = _state([_row("sys")], TIERS_WITH_COLOR)
    monkeypatch.setattr(badges, "_load_state", lambda: st)
    out = badges._standing("sys")
    assert out["tier"] == "gold" and out["place"] == 1 and out["out_of"] == 1
    assert out["color"] == "#FFD700"


def test_standing_absent_slug_returns_none(monkeypatch):
    st = _state([_row("sys")], TIERS_WITH_COLOR)
    monkeypatch.setattr(badges, "_load_state", lambda: st)
    assert badges._standing("ghost") is None


def test_standing_swallows_load_failure(monkeypatch):
    def boom():
        raise RuntimeError("hub down")
    monkeypatch.setattr(badges, "_load_state", boom)
    assert badges._standing("sys") is None


def test_endpoint_payload_shape():
    p = badges._endpoint_payload("arena tier", "gold", "#FFD700")
    assert p == {"schemaVersion": 1, "label": "arena tier",
                 "message": "gold", "color": "#FFD700"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/kirill/speech-spoof-bench/arena && pytest tests/test_badges.py -k "standing or payload" -v`
Expected: FAIL — `_standing` / `_endpoint_payload` / `_load_state` undefined.

- [ ] **Step 3: Implement the helpers**

Append to `arena/badges.py`:

```python
def _load_state():
    """Indirection point so tests can monkeypatch the live-state read."""
    import ingest
    return ingest.load_state()


def _standing(slug: str) -> dict | None:
    """{'tier', 'place', 'out_of', 'color'} for slug, or None if unavailable.

    Swallows any load error (returns None) so the endpoint can degrade to an
    'unranked' badge instead of a broken image.
    """
    try:
        state = _load_state()
    except Exception as exc:  # noqa: BLE001
        logger.warning("badge: load_state failed: %s", exc)
        return None
    from ranking import system_standing

    manifest = state.manifest
    tiers = manifest.get("tiers", [])
    core = [e["id"] for e in manifest.get("core_set", [])]
    primary = manifest.get("metrics_in_use", ["eer_percent"])[0]
    st = system_standing(state.rows, tiers, core, primary, slug)
    if st is None:
        return None
    st["color"] = _tier_color(st["tier"], tiers)
    return st


def _endpoint_payload(label: str, message: str, color: str) -> dict:
    """shields.io endpoint badge JSON."""
    return {"schemaVersion": 1, "label": label, "message": message, "color": color}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/kirill/speech-spoof-bench/arena && pytest tests/test_badges.py -v`
Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
cd /home/kirill/speech-spoof-bench/arena
git add badges.py tests/test_badges.py
git commit -m "feat(badges): _standing (live state → standing+color) + payload helper"
```

---

### Task 4: The two routes

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


def test_tier_endpoint_returns_shields_json(monkeypatch):
    c = _client([_row("sys")], TIERS_WITH_COLOR, monkeypatch)
    r = c.get("/badge/sys/tier.json")
    assert r.status_code == 200
    assert r.json() == {"schemaVersion": 1, "label": "arena tier",
                        "message": "gold", "color": "#FFD700"}
    assert r.headers["cache-control"] == "max-age=300"


def test_rank_endpoint_message_and_color(monkeypatch):
    c = _client([_row("sys")], TIERS_WITH_COLOR, monkeypatch)
    r = c.get("/badge/sys/rank.json")
    assert r.status_code == 200
    body = r.json()
    assert body["label"] == "arena rank"
    assert body["message"] == "#1 of 1"
    assert body["color"] == "#FFD700"  # inherits tier color
    assert r.headers["cache-control"] == "max-age=300"


def test_rank_message_with_multiple_systems(monkeypatch):
    rows = [_row("good", eer=1.0), _row("bad", eer=9.0)]
    # core_set in _state is the single dataset org/a, so both cover 1/1 → gold;
    # within gold, lower eer ranks first, so bad is #2 of 2.
    c = _client(rows, TIERS_WITH_COLOR, monkeypatch)
    r = c.get("/badge/bad/rank.json")
    assert r.json()["message"] == "#2 of 2"


def test_unknown_slug_is_unranked_not_404(monkeypatch):
    c = _client([_row("sys")], TIERS_WITH_COLOR, monkeypatch)
    for kind in ("tier", "rank"):
        r = c.get(f"/badge/ghost/{kind}.json")
        assert r.status_code == 200
        assert r.json()["message"] == "unranked"
        assert r.json()["color"] == "lightgrey"


def test_palette_color_when_manifest_has_no_color(monkeypatch):
    c = _client([_row("sys")], TIERS_NO_COLOR, monkeypatch)
    r = c.get("/badge/sys/tier.json")
    assert r.json()["color"] == badges._PALETTE[0]  # gold position
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/kirill/speech-spoof-bench/arena && pytest tests/test_badges.py -k "endpoint or unknown or palette_color_when or rank_message" -v`
Expected: FAIL — routes return 404 (not defined).

- [ ] **Step 3: Implement the routes**

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
Expected: PASS (13 tests).

- [ ] **Step 5: Commit**

```bash
cd /home/kirill/speech-spoof-bench/arena
git add badges.py tests/test_badges.py
git commit -m "feat(badges): /badge/<slug>/tier.json + rank.json routes"
```

---

### Task 5: Mount the router in `arena/main.py`

**Files:**
- Modify: `arena/main.py`

- [ ] **Step 1: Add the include next to the webhook router**

In `arena/main.py`, find:

```python
from webhook import router as webhook_router
app.include_router(webhook_router)
```

Add immediately below:

```python
from badges import router as badge_router
app.include_router(badge_router)
```

(Both must be before the `gr.mount_gradio_app(app, demo, path="/")` line so the
`/badge/...` routes aren't shadowed by the Gradio mount at `/`.)

- [ ] **Step 2: Verify the app imports and routes are registered**

Run:
```bash
cd /home/kirill/speech-spoof-bench/arena && python -c "
import main
paths = {r.path for r in main.app.routes}
assert '/badge/{slug}/tier.json' in paths, paths
assert '/badge/{slug}/rank.json' in paths, paths
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

## Slice 3 — manifest schema + manifest colors

### Task 6: Allow optional `color` on tiers in `manifest.schema.json`

**Files:**
- Modify: `speech-spoof-bench/src/speech_spoof_bench/schema/manifest.schema.json`
- Create: `speech-spoof-bench/tests/test_manifest_schema_tier_color.py`

- [ ] **Step 1: Write the failing test**

```python
# speech-spoof-bench/tests/test_manifest_schema_tier_color.py
"""manifest.schema.json must accept tiers with AND without an optional color."""
from __future__ import annotations

import json
from importlib import resources

import pytest
from jsonschema import ValidationError, validate


def _schema():
    with resources.files("speech_spoof_bench.schema").joinpath("manifest.schema.json").open("r") as f:
        return json.load(f)


def _manifest(tiers):
    return {
        "ranking_version": "v1",
        "schema_version": 1,
        "metrics_in_use": ["eer_percent"],
        "tiers": tiers,
        "core_set": [{"id": "Org/A", "revision": "abc1234"}],
        "extended": [],
    }


def test_tier_without_color_still_valid():
    validate(_manifest([{"name": "gold", "min_coverage": 1.0}]), _schema())


def test_tier_with_color_valid():
    validate(_manifest([{"name": "gold", "min_coverage": 1.0, "color": "#FFD700"}]), _schema())


def test_unknown_tier_field_still_rejected():
    with pytest.raises(ValidationError):
        validate(_manifest([{"name": "gold", "min_coverage": 1.0, "bogus": 1}]), _schema())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/kirill/speech-spoof-bench/speech-spoof-bench && pytest tests/test_manifest_schema_tier_color.py -v`
Expected: FAIL — `test_tier_with_color_valid` fails (`additionalProperties` rejects `color`).

- [ ] **Step 3: Add `color` to the tier schema**

In `src/speech_spoof_bench/schema/manifest.schema.json`, change the `tiers` item `properties` block from:

```json
        "properties": {
          "name": {"type": "string", "minLength": 1},
          "min_coverage": {"type": "number", "minimum": 0, "maximum": 1}
        }
```

to:

```json
        "properties": {
          "name": {"type": "string", "minLength": 1},
          "min_coverage": {"type": "number", "minimum": 0, "maximum": 1},
          "color": {"type": "string", "minLength": 1}
        }
```

(`color` is optional — not added to the tier's `required` array.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/kirill/speech-spoof-bench/speech-spoof-bench && pytest tests/test_manifest_schema_tier_color.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
cd /home/kirill/speech-spoof-bench/speech-spoof-bench
git add src/speech_spoof_bench/schema/manifest.schema.json tests/test_manifest_schema_tier_color.py
git commit -m "feat(schema): optional tier color in manifest.schema.json"
```

---

### Task 7: Add tier colors to the live manifest

**Files:**
- Modify: `arena-manifest/manifest.yaml`

- [ ] **Step 1: Read the current manifest**

Run: `cd /home/kirill/speech-spoof-bench/arena-manifest && cat manifest.yaml`
Note the existing `tiers:` block.

- [ ] **Step 2: Add `color` to each tier entry**

Edit `arena-manifest/manifest.yaml` so the tiers block reads:

```yaml
tiers:
  - {name: gold,   min_coverage: 1.0, color: "#FFD700"}
  - {name: silver, min_coverage: 0.5, color: "#C0C0C0"}
  - {name: bronze, min_coverage: 0.0, color: "#CD7F32"}
```

(Preserve the file's existing key order and other fields exactly; only add the
`color` key to each tier mapping.)

- [ ] **Step 3: Validate the edited manifest against the schema**

Run:
```bash
cd /home/kirill/speech-spoof-bench/speech-spoof-bench && python -c "
import json, yaml
from importlib import resources
with resources.files('speech_spoof_bench.schema').joinpath('manifest.schema.json').open() as f:
    schema = json.load(f)
from jsonschema import validate
data = yaml.safe_load(open('/home/kirill/speech-spoof-bench/arena-manifest/manifest.yaml'))
validate(data, schema)
print('manifest valid; tiers:', data['tiers'])
"
```
Expected: prints the three tiers each with a `color`, no validation error.

- [ ] **Step 4: Commit**

```bash
cd /home/kirill/speech-spoof-bench/arena-manifest
git add manifest.yaml
git commit -m "manifest: add tier colors (gold/silver/bronze hexes)"
```

---

## Slice 4 — `build_paste_comment` emits all three badge lines

### Task 8: Add the two endpoint badge lines to the comment

**Files:**
- Modify: `speech-spoof-bench/src/speech_spoof_bench/badge.py`
- Modify: `speech-spoof-bench/tests/test_badge_build_paste_comment.py`

- [ ] **Step 1: Write the failing test**

Append to `speech-spoof-bench/tests/test_badge_build_paste_comment.py`:

```python
def test_comment_includes_tier_and_rank_endpoint_badges():
    body = _build()  # existing helper builds the canonical comment
    host = "speechantispoofingbenchmarks-speechantispoofingarena.hf.space"
    # tier badge: shields endpoint pointing at the arena tier.json
    assert (
        f"https://img.shields.io/endpoint?url=https://{host}/badge/aasist/tier.json"
        in body
    )
    # rank badge
    assert (
        f"https://img.shields.io/endpoint?url=https://{host}/badge/aasist/rank.json"
        in body
    )
    # both link back to the arena system page
    assert body.count("?system=aasist)") >= 3  # eer + tier + rank click targets


def test_endpoint_badge_md_builder():
    md = badge._endpoint_badge_md("arena tier", "aasist", "tier")
    assert md.startswith("[![arena tier](https://img.shields.io/endpoint?url=")
    assert "/badge/aasist/tier.json" in md
    assert md.endswith(f"({badge.ARENA_URL}?system=aasist)")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/kirill/speech-spoof-bench/speech-spoof-bench && pytest tests/test_badge_build_paste_comment.py -k "endpoint" -v`
Expected: FAIL — `_endpoint_badge_md` undefined / endpoint URLs absent from body.

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
    served by the arena, linking back to the system's arena page."""
    endpoint = f"https://{ARENA_HOST}/badge/{slug}/{kind}.json"
    img = f"https://img.shields.io/endpoint?url={endpoint}"
    return f"[![{label}]({img})]({ARENA_URL}?system={slug})"
```

In `build_paste_comment`, replace the README step (section 2) so it emits three
badge lines. Change:

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

## Slice 5 — deploy + push

### Task 9: Push all three repos

**Files:** none (git push only)

- [ ] **Step 1: Run each repo's test suite**

```bash
cd /home/kirill/speech-spoof-bench/arena && pytest tests/test_badges.py tests/test_ranking.py -q
cd /home/kirill/speech-spoof-bench/speech-spoof-bench && pytest tests/test_manifest_schema_tier_color.py tests/test_badge_build_paste_comment.py -q
```
Expected: all pass.

- [ ] **Step 2: Push speech-spoof-bench (GitHub) first**

```bash
cd /home/kirill/speech-spoof-bench/speech-spoof-bench
git push origin main
```

- [ ] **Step 3: Push arena-manifest (HF dataset)**

```bash
cd /home/kirill/speech-spoof-bench/arena-manifest
git push origin main
```

- [ ] **Step 4: Push arena (HF Space) — triggers Docker rebuild**

```bash
cd /home/kirill/speech-spoof-bench/arena
git push origin main
```

- [ ] **Step 5: Wait for the Space to come back up**

Run:
```bash
until curl -fsS -o /dev/null https://speechantispoofingbenchmarks-speechantispoofingarena.hf.space/healthz; do sleep 5; done
echo "Space healthy"
```
Expected: `Space healthy`.

---

## Manual end-to-end verification

Run after Task 9. These are not automated — they confirm the live system.

- [ ] **M1.** Fetch the live JSON endpoints:

```bash
curl -s https://speechantispoofingbenchmarks-speechantispoofingarena.hf.space/badge/random-baseline/tier.json
curl -s https://speechantispoofingbenchmarks-speechantispoofingarena.hf.space/badge/random-baseline/rank.json
curl -sI https://speechantispoofingbenchmarks-speechantispoofingarena.hf.space/badge/random-baseline/tier.json | grep -i cache-control
```
Expected: tier.json → `{"schemaVersion":1,"label":"arena tier","message":"gold","color":"#FFD700"}`; rank.json → `message":"#1 of 1"`, `color":"#FFD700"`; `cache-control: max-age=300`.

- [ ] **M2.** Unknown slug degrades gracefully:

```bash
curl -s https://speechantispoofingbenchmarks-speechantispoofingarena.hf.space/badge/nonexistent/tier.json
```
Expected: `200`, `{"...,"message":"unranked","color":"lightgrey"}`.

- [ ] **M3.** Rendered shields images resolve:

```bash
curl -s -o /dev/null -w "%{http_code} %{content_type}\n" \
  "https://img.shields.io/endpoint?url=https://speechantispoofingbenchmarks-speechantispoofingarena.hf.space/badge/random-baseline/tier.json"
```
Expected: `200 image/svg+xml`. Repeat for `rank.json`.

- [ ] **M4.** Add the two badge lines to the `random-baseline-asas` model README
  (next to the EER badge added in Phase 9), then visit
  `https://huggingface.co/SpeechAntiSpoofingBenchmarks/random-baseline-asas` and
  confirm all three badges render and click through to the Arena. Use this
  Python (run from the speech-spoof-bench package env):

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

readme_path = hf_hub_download("SpeechAntiSpoofingBenchmarks/random-baseline-asas",
                              filename="README.md", repo_type="model")
text = Path(readme_path).read_text()
# Insert the two lines right after the existing EER badge line (first "![" line).
lines = text.splitlines()
for i, ln in enumerate(lines):
    if ln.startswith("[![EER"):
        lines[i] = ln + "\n" + tier_md + "\n" + rank_md
        break
new = "\n".join(lines) + "\n"
api.upload_file(path_or_fileobj=new.encode("utf-8"), path_in_repo="README.md",
                repo_id="SpeechAntiSpoofingBenchmarks/random-baseline-asas",
                repo_type="model", commit_message="Add arena tier + rank badges")
print("README updated")
```

- [ ] **M5.** (Genericity) Confirm a new tier needs no code change. Against a
  local manifest dict (no HF write), assert the endpoint colors a `platinum`
  tier from its manifest `color`:

```bash
cd /home/kirill/speech-spoof-bench/arena && python -c "
import badges
from datetime import datetime
from schema import ArenaState, Row
tiers = [{'name':'platinum','min_coverage':1.0,'color':'#E5E4E2'},
         {'name':'gold','min_coverage':0.5,'color':'#FFD700'}]
rows = [Row(system_slug='p', system_name='p', dataset_id='org/a', revision='r',
            scores={'eer_percent':1.0}, reproduction_level='scoring',
            submitted_at='2026-01-01', submission_url='u')]
state = ArenaState(manifest={'tiers':tiers,'core_set':[{'id':'org/a','revision':'abc1234'}],
                             'extended':[],'metrics_in_use':['eer_percent']},
                   rows=rows, loaded_at=datetime(2026,1,1))
badges._load_state = lambda: state
st = badges._standing('p')
assert st['tier']=='platinum' and st['color']=='#E5E4E2', st
print('genericity OK:', st)
"
```
Expected: `genericity OK: {'tier': 'platinum', ..., 'color': '#E5E4E2'}`.

---

## Self-review notes

- **Spec coverage:** endpoints (Tasks 2–5), `system_standing` (Task 1), manifest color schema (Task 6) + live colors (Task 7), `build_paste_comment` three badges (Task 8), deploy (Task 9), manual M1–M5. All spec sections mapped.
- **No placeholders:** every code/test step is concrete. Unknown-slug, palette-fallback, load-failure, tie-break all have explicit tests.
- **Type consistency:** `system_standing` returns `{tier, place, out_of}` (Task 1); `_standing` adds `color` (Task 3); routes read `st["tier"]/["place"]/["out_of"]/["color"]` (Task 4). `_endpoint_badge_md(label, slug, kind)` signature matches its call sites in Task 8 and M4. `_tier_color(name, tiers)`, `_PALETTE`, `_UNRANKED_COLOR`, `_CACHE_CONTROL` consistent across Tasks 2–4.
- **Cross-repo discipline:** every commit/push `cd`s into the right sub-repo. Push order in Task 9 is GitHub → manifest → Space (workflow installs the package fresh; Space rebuild last).
