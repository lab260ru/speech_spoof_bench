# Phase 10 — Arena Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the Arena into an MTEB-style leaderboard with clickable models/datasets, inline paper/BibTeX/model detail, and three new tabs (Submit, By model size, Over time), changing only render/read code plus two additive, backward-compatible data fields.

**Architecture:** Pure data transforms (`ranking.py`, new `charts.py`, `events.py`, `docs_fetch.py`) stay Gradio-free and are unit-tested with real assertions. Gradio wiring lives in `app.py` + new `ui/*` modules and is verified by a smoke test that builds the Blocks plus manual visual checks on the Space. The submission YAML gains one **optional** `system.params_millions`; the `arena-manifest` repo gains one **optional** `CHANGELOG.yaml`. Nothing in CI/webhook/badge/ranking-math/verification changes.

**Tech Stack:** Python 3.10+, Gradio 6.14, `gradio_leaderboard`, pandas, plotly, `huggingface_hub`, `jsonschema`, pytest. Two repos: `speech-spoof-bench` (pip package + docs) and `arena` (HF Docker Space). One data file in `arena-manifest`.

---

## Repos & working directories

- **Package:** `/home/kirill/speech-spoof-bench/speech-spoof-bench/` — run tests with `pytest` from this dir.
- **Arena:** `/home/kirill/speech-spoof-bench/arena/` — run tests with `pytest` from this dir (tests do `import ingest`, so cwd must be `arena/`).
- **Manifest:** `/home/kirill/speech-spoof-bench/arena-manifest/` — holds `manifest.yaml`; gains `CHANGELOG.yaml`.

After every package logic/schema change, the Arena's `arena/requirements.txt` pin (`speech-spoof-bench @ git+...@<sha>`) must be bumped to the new sha or the Space runs stale code (Task 18).

---

## File Structure

### Package (`speech-spoof-bench/`)
- Modify: `src/speech_spoof_bench/data/submission_meta.schema.json` — add optional `system.params_millions`.
- Modify: `src/speech_spoof_bench/schema/submission.schema.json` — add optional `system.params_millions`.
- Modify: `src/speech_spoof_bench/submit.py` (`build_submission_payload`) — pass `params_millions` through when present.
- Create: `docs/submitting/submit-model.md`, `docs/submitting/submit-dataset.md` — canonical contribution docs.
- Test: `tests/test_submit.py` (extend), `tests/test_submission_schema.py` (extend or create).

### Arena (`arena/`)
- Modify: `schema.py` — new `Row` fields.
- Modify: `ingest.py` (`_to_row`) — populate new fields.
- Modify: `ranking.py` — keep math; reuse `global_scores` from new modules (no change to existing functions).
- Modify: `requirements.txt` — add `gradio_leaderboard`, `plotly`; bump package pin.
- Create: `leaderboard.py` — pure builder: rows → ordered `pandas.DataFrame` for the MTEB table + a `links_legend()` helper.
- Create: `charts.py` — pure: `size_series(...)`, `sota_timeline(...)` returning plain dicts/DataFrames.
- Create: `events.py` — pure: `build_events(rows, manifest, changelog)`.
- Create: `docs_fetch.py` — fetch/cache Submit-tab markdown at the pinned sha + bundled fallback.
- Create: `changelog.py` — fetch/parse optional `CHANGELOG.yaml` from `arena-manifest`.
- Modify: `app.py` — wire all six tabs + totals header + detail strip + view radio.
- Test: `tests/test_leaderboard.py`, `tests/test_charts.py`, `tests/test_events.py`, `tests/test_docs_fetch.py`, `tests/test_changelog.py`, extend `tests/test_ingest.py`, `tests/test_app_overview.py`.

### Manifest (`arena-manifest/`)
- Create: `CHANGELOG.yaml` — curated infra-event feed (optional).

---

# SLICE 1 — Data plumbing + MTEB-style leaderboard

## Task 1: Add optional `params_millions` to both schemas (package)

**Files:**
- Modify: `src/speech_spoof_bench/data/submission_meta.schema.json`
- Modify: `src/speech_spoof_bench/schema/submission.schema.json`
- Test: `tests/test_submission_schema.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_submission_schema.py` (or append if it exists):

```python
import json
from importlib import resources

import jsonschema
import pytest

import yaml


def _submission_schema():
    with resources.files("speech_spoof_bench.schema").joinpath("submission.schema.json").open() as f:
        return json.load(f)


def _meta_schema():
    with resources.files("speech_spoof_bench.data").joinpath("submission_meta.schema.json").open() as f:
        return json.load(f)


_BASE_SYSTEM = {
    "name": "AASIST", "slug": "aasist", "description": "x",
    "code": "https://github.com/x/y", "checkpoint": "https://huggingface.co/x/y",
    "paper": {"arxiv_id": "1", "url": "https://arxiv.org/abs/1", "bibtex": "@x{y}"},
}


def test_meta_accepts_params_millions():
    inst = {"system": {**_BASE_SYSTEM, "params_millions": 52.3}}
    jsonschema.validate(inst, _meta_schema())  # must not raise


def test_meta_valid_without_params_millions():
    inst = {"system": dict(_BASE_SYSTEM)}
    jsonschema.validate(inst, _meta_schema())  # optional → still valid


def test_meta_rejects_negative_params():
    inst = {"system": {**_BASE_SYSTEM, "params_millions": -1}}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(inst, _meta_schema())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_submission_schema.py -v`
Expected: FAIL — `test_meta_accepts_params_millions` raises `ValidationError` (Additional properties not allowed: 'params_millions').

- [ ] **Step 3: Add the field to both schemas**

In `src/speech_spoof_bench/data/submission_meta.schema.json`, inside `properties.system.properties`, after the `paper` block, add:

```json
        "params_millions": {"type": "number", "minimum": 0}
```

In `src/speech_spoof_bench/schema/submission.schema.json`, inside `properties.system.properties`, after the `paper` block, add the same line:

```json
        "params_millions": {"type": "number", "minimum": 0}
```

(Do NOT add `params_millions` to either `system.required` — it stays optional.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_submission_schema.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Run the full package suite to confirm no regression**

Run: `pytest -q`
Expected: PASS (existing submissions without the field still validate).

- [ ] **Step 6: Commit**

```bash
git add src/speech_spoof_bench/data/submission_meta.schema.json src/speech_spoof_bench/schema/submission.schema.json tests/test_submission_schema.py
git commit -m "feat(schema): add optional system.params_millions"
```

---

## Task 2: Pass `params_millions` through `build_submission_payload` (package)

**Files:**
- Modify: `src/speech_spoof_bench/submit.py:81-88`
- Test: `tests/test_submit.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_submit.py`:

```python
from speech_spoof_bench.submit import build_submission_payload


def _result_yaml():
    return {
        "dataset": {"id": "Org/DS", "revision": "abc1234", "split": "test"},
        "scores": {"eer_percent": 1.0, "n_trials": 10, "n_skipped": 0},
        "artifact": {"bench_version": "speech-spoof-bench==0.1.0"},
    }


def _meta(extra_system=None):
    system = {
        "name": "AASIST", "slug": "aasist", "description": "x",
        "code": "https://github.com/x/y", "checkpoint": "https://huggingface.co/x/y",
        "paper": {"arxiv_id": "1", "url": "https://arxiv.org/abs/1", "bibtex": "@x{y}"},
    }
    if extra_system:
        system.update(extra_system)
    return {"system": system}


def test_payload_includes_params_when_present():
    payload = build_submission_payload(
        result_yaml=_result_yaml(), meta=_meta({"params_millions": 52.3}),
        scores_url="https://huggingface.co/x/y/resolve/abc1234/scores.txt",
        scores_sha256="0" * 64, hf_username="u", contact="c@e.com",
        submitted_at="2026-05-29",
    )
    assert payload["system"]["params_millions"] == 52.3


def test_payload_omits_params_when_absent():
    payload = build_submission_payload(
        result_yaml=_result_yaml(), meta=_meta(),
        scores_url="https://huggingface.co/x/y/resolve/abc1234/scores.txt",
        scores_sha256="0" * 64, hf_username="u", contact="c@e.com",
        submitted_at="2026-05-29",
    )
    assert "params_millions" not in payload["system"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_submit.py -k params -v`
Expected: FAIL — `test_payload_includes_params_when_present` (KeyError / assert: key missing).

- [ ] **Step 3: Implement the pass-through**

In `src/speech_spoof_bench/submit.py`, after the `"system": { ... "paper": dict(sys_meta["paper"]) }` block is assembled (currently lines 81-88), add immediately after the `payload` dict is created (after line 99, before the `notes` handling):

```python
    if "params_millions" in sys_meta:
        payload["system"]["params_millions"] = sys_meta["params_millions"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_submit.py -k params -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/speech_spoof_bench/submit.py tests/test_submit.py
git commit -m "feat(submit): carry optional params_millions into submission YAML"
```

---

## Task 3: New `Row` fields + ingest mapping (arena)

**Files:**
- Modify: `arena/schema.py:10-21`
- Modify: `arena/ingest.py:114-126` (`_to_row`)
- Test: `arena/tests/test_ingest.py`

- [ ] **Step 1: Write the failing test**

Append to `arena/tests/test_ingest.py`:

```python
def test_row_carries_paper_model_and_params(monkeypatch, fixtures_dir, tmp_path):
    # Build a submission fixture with params_millions on the fly.
    import yaml
    base = yaml.safe_load((fixtures_dir / "submissions" / "valid.yaml").read_text())
    base["system"]["params_millions"] = 52.3
    sub_dir = tmp_path / "submissions"
    sub_dir.mkdir()
    (sub_dir / "withparams.yaml").write_text(yaml.safe_dump(base))

    def fake_fetch_manifest():
        return yaml.safe_load((fixtures_dir / "manifest.yaml").read_text())

    def fake_list(dataset_id, *, api=None):
        if dataset_id == "SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA":
            return ["submissions/withparams.yaml"]
        return []

    def fake_fetch_submission(dataset_id, path):
        from speech_spoof_bench.submission import parse_submission
        return parse_submission((tmp_path / path).read_text())

    monkeypatch.setattr(ingest, "_fetch_manifest", fake_fetch_manifest)
    monkeypatch.setattr(ingest, "_list_submission_files", fake_list)
    monkeypatch.setattr(ingest, "_fetch_submission_dict", fake_fetch_submission)

    state = ingest.load_state(force_refresh=True)
    row = state.rows[0]
    assert row.params_millions == 52.3
    assert row.checkpoint_url == "https://huggingface.co/example/x"
    assert row.paper_url == "https://arxiv.org/abs/1911.01601"
    assert row.paper_bibtex == "@misc{x,title={x}}"
    assert row.description == "stub"
    assert row.reproduced_at == "2026-05-21"


def test_row_defaults_when_params_absent(monkeypatch, fixtures_dir):
    _patch_hf(monkeypatch, fixtures_dir, {
        "SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA": ["valid.yaml"],
        "SpeechAntiSpoofingBenchmarks/ASVspoof2021_LA": [],
        "SpeechAntiSpoofingBenchmarks/InTheWild": [],
    })
    row = ingest.load_state(force_refresh=True).rows[0]
    assert row.params_millions is None
    assert row.paper_arxiv_id == "1911.01601"
```

- [ ] **Step 2: Run test to verify it fails**

Run (from `arena/`): `pytest tests/test_ingest.py -k "params or carries" -v`
Expected: FAIL — `TypeError`/`AttributeError`: `Row` has no field `params_millions`.

- [ ] **Step 3: Add the fields to `Row`**

In `arena/schema.py`, extend the `Row` dataclass (keep existing fields and order; append the new ones with defaults so legacy `cache.json` deserializes):

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
    description: str = ""
    code_url: str = ""
    checkpoint_url: str = ""
    paper_arxiv_id: str = ""
    paper_url: str = ""
    paper_bibtex: str = ""
    params_millions: float | None = None
    reproduced_at: str = ""
```

- [ ] **Step 4: Populate them in `_to_row`**

In `arena/ingest.py`, replace the `_to_row` body's `Row(...)` construction with one that reads the new fields:

```python
def _to_row(sub: dict, dataset_id: str, path: str) -> Row:
    scores = {k: v for k, v in sub["scores"].items() if k not in {"n_trials", "n_skipped"}}
    system = sub["system"]
    paper = system.get("paper", {})
    repro = sub.get("reproduction") or {}
    return Row(
        system_slug=system["slug"],
        system_name=system["name"],
        dataset_id=dataset_id,
        revision=sub["dataset"]["revision"],
        scores=scores,
        reproduction_level=repro.get("match", "scoring"),
        submitted_at=sub["submitted_at"] if isinstance(sub["submitted_at"], str) else sub["submitted_at"].isoformat(),
        submission_url=f"https://huggingface.co/datasets/{dataset_id}/blob/main/{path}",
        n_trials=int(sub["scores"].get("n_trials", 0)),
        description=system.get("description", ""),
        code_url=system.get("code", ""),
        checkpoint_url=system.get("checkpoint", ""),
        paper_arxiv_id=paper.get("arxiv_id", ""),
        paper_url=paper.get("url", ""),
        paper_bibtex=paper.get("bibtex", ""),
        params_millions=system.get("params_millions"),
        reproduced_at=str(repro.get("reproduced_at", "")),
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run (from `arena/`): `pytest tests/test_ingest.py -v`
Expected: PASS (existing + 2 new).

- [ ] **Step 6: Commit**

```bash
git add arena/schema.py arena/ingest.py arena/tests/test_ingest.py
git commit -m "feat(arena): carry paper/model/params/date fields into Row"
```

---

## Task 4: Leaderboard DataFrame builder (arena, pure)

**Files:**
- Create: `arena/leaderboard.py`
- Test: `arena/tests/test_leaderboard.py`

This builds the exact table shape the MTEB component renders. The System cell is a markdown link to the checkpoint; a `Links` column holds 📄/⧉ markdown. Pure function → testable; the Gradio component consumes its output in Task 5.

- [ ] **Step 1: Write the failing test**

Create `arena/tests/test_leaderboard.py`:

```python
import pandas as pd

from schema import Row
from leaderboard import overview_dataframe, dataset_chip_links


def _row(slug, ds, eer, **kw):
    base = dict(
        system_slug=slug, system_name=slug.upper(), dataset_id=ds, revision="r",
        scores={"eer_percent": eer}, reproduction_level="scoring",
        submitted_at="2026-05-20", submission_url="u", n_trials=100,
        checkpoint_url=f"https://huggingface.co/{slug}", paper_url=f"https://arxiv.org/abs/{slug}",
    )
    base.update(kw)
    return Row(**base)


def test_overview_dataframe_columns_and_links():
    rows = [
        _row("aasist", "Org/A", 0.9, reproduction_level="inference"),
        _row("aasist", "Org/B", 2.0),
        _row("rnd", "Org/A", 49.0),
    ]
    ranks = {"aasist": {"place": 1}, "rnd": {"place": 2}}
    df = overview_dataframe(rows, ["aasist", "rnd"], ["Org/A", "Org/B"],
                            primary_metric="eer_percent", ranks=ranks)
    assert list(df.columns) == ["Rank", "System", "Mean", "Org/A", "Org/B", "Links"]
    top = df.iloc[0]
    assert top["Rank"] == 1
    # System is a markdown link to the checkpoint, with the verification badge.
    assert "huggingface.co/aasist" in top["System"]
    assert "★" in top["System"]              # inference badge present on any inference row
    assert top["Org/A"] == 0.9
    assert "arxiv.org/abs/aasist" in top["Links"]  # 📄 paper link


def test_overview_dataframe_blank_for_missing_cell():
    rows = [_row("rnd", "Org/A", 49.0)]
    df = overview_dataframe(rows, ["rnd"], ["Org/A", "Org/B"],
                            primary_metric="eer_percent", ranks={"rnd": {"place": 1}})
    assert pd.isna(df.iloc[0]["Org/B"]) or df.iloc[0]["Org/B"] == ""


def test_dataset_chip_links_markdown():
    md = dataset_chip_links(["SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA"])
    assert "huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA" in md
    assert "ASVspoof2019_LA" in md
```

- [ ] **Step 2: Run test to verify it fails**

Run (from `arena/`): `pytest tests/test_leaderboard.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'leaderboard'`.

- [ ] **Step 3: Implement `leaderboard.py`**

Create `arena/leaderboard.py`:

```python
"""Pure builders: Row list -> DataFrame for the MTEB-style table. No Gradio."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import pandas as pd

from schema import Row


def _short(dataset_id: str) -> str:
    return dataset_id.split("/")[-1]


def _system_cell(name: str, checkpoint_url: str, level: str) -> str:
    badge = " ★" if level == "inference" else " ✔"
    if checkpoint_url:
        return f"[{name}]({checkpoint_url}){badge}"
    return f"{name}{badge}"


def _links_cell(paper_url: str, bibtex: str) -> str:
    parts = []
    if paper_url:
        parts.append(f"[📄]({paper_url})")
    if bibtex:
        parts.append("⧉")  # BibTeX surfaced in the detail strip (Task 5)
    return " ".join(parts)


def overview_dataframe(
    rows: Iterable[Row],
    slug_order: list[str],
    dataset_ids: list[str],
    primary_metric: str,
    ranks: dict[str, dict],
) -> pd.DataFrame:
    """One row per system_slug (in slug_order). Columns:
    Rank, System (md link + badge), Mean (unweighted over present cells),
    one per dataset_id (primary metric or NaN), Links (md)."""
    by_system: dict[str, dict[str, Row]] = defaultdict(dict)
    for r in rows:
        by_system[r.system_slug][r.dataset_id] = r

    records = []
    for slug in slug_order:
        per_ds = by_system.get(slug, {})
        if not per_ds:
            continue
        any_row = next(iter(per_ds.values()))
        levels = {r.reproduction_level for r in per_ds.values()}
        level = "inference" if "inference" in levels else "scoring"
        cells = {ds: per_ds[ds].scores.get(primary_metric) for ds in dataset_ids if ds in per_ds}
        present = [v for v in cells.values() if v is not None]
        rec = {
            "Rank": ranks.get(slug, {}).get("place"),
            "System": _system_cell(any_row.system_name, any_row.checkpoint_url, level),
            "Mean": round(sum(present) / len(present), 3) if present else None,
        }
        for ds in dataset_ids:
            rec[_short(ds)] = cells.get(ds)
        rec["Links"] = _links_cell(any_row.paper_url, any_row.paper_bibtex)
        records.append(rec)

    columns = ["Rank", "System", "Mean"] + [_short(d) for d in dataset_ids] + ["Links"]
    df = pd.DataFrame(records, columns=columns)
    return df.sort_values("Rank", na_position="last").reset_index(drop=True)


def dataset_chip_links(dataset_ids: list[str]) -> str:
    """Markdown row of clickable dataset chips (headers can't be links in Gradio)."""
    chips = [f"[{_short(d)} ↗](https://huggingface.co/datasets/{d})" for d in dataset_ids]
    return "**Datasets:** " + " · ".join(chips)
```

Note: the test asserts column header `Org/A` but `overview_dataframe` shortens to the last path segment. Update the test's expected columns to the shortened names: change `["Rank", "System", "Mean", "Org/A", "Org/B", "Links"]` to `["Rank", "System", "Mean", "A", "B", "Links"]` and `top["Org/A"]`→`top["A"]`, `iloc[0]["Org/B"]`→`iloc[0]["B"]` before running. (Datasets in real use are `SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA` → `ASVspoof2019_LA`.)

- [ ] **Step 4: Run test to verify it passes**

Run (from `arena/`): `pytest tests/test_leaderboard.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add arena/leaderboard.py arena/tests/test_leaderboard.py
git commit -m "feat(arena): pure MTEB-style overview dataframe builder"
```

---

## Task 5: Wire Overview tab — leaderboard component + chips + detail strip + view radio + totals header (arena)

**Files:**
- Modify: `arena/requirements.txt`
- Modify: `arena/app.py`
- Test: `arena/tests/test_app_overview.py`

The `gradio_leaderboard` component renders the DataFrame with pinned columns, sort, search, and column-select. The detail strip is driven by a system `gr.Dropdown` (robust regardless of the component's row-select support — the documented fallback in the spec §4.1). BibTeX renders in a `gr.Code` with its native copy button.

- [ ] **Step 1: Add dependencies**

In `arena/requirements.txt` add two lines (keep the existing package pin line for now; Task 18 bumps the sha):

```
gradio_leaderboard>=0.0.13
plotly>=5.20
```

Install locally to develop: `pip install gradio_leaderboard plotly`.

- [ ] **Step 2: Write a smoke test for the detail strip helper**

Add to `arena/tests/test_app_overview.py`:

```python
def test_detail_markdown_renders_links_and_params():
    from app import _detail_markdown
    from schema import Row
    r = Row(
        system_slug="aasist", system_name="AASIST", dataset_id="Org/A", revision="r",
        scores={"eer_percent": 0.9}, reproduction_level="inference",
        submitted_at="2026-05-20", submission_url="u", n_trials=100,
        description="ref impl", checkpoint_url="https://huggingface.co/x",
        paper_url="https://arxiv.org/abs/1", paper_arxiv_id="1", params_millions=52.3,
    )
    md, bibtex = _detail_markdown([r], "aasist")
    assert "52.3" in md
    assert "arxiv.org/abs/1" in md
    assert "huggingface.co/x" in md


def test_detail_markdown_unknown_slug():
    from app import _detail_markdown
    md, bibtex = _detail_markdown([], "nope")
    assert md == "" and bibtex == ""
```

- [ ] **Step 3: Run smoke test to verify it fails**

Run (from `arena/`): `pytest tests/test_app_overview.py -k detail -v`
Expected: FAIL — `ImportError: cannot import name '_detail_markdown'`.

- [ ] **Step 4: Implement `_detail_markdown` and rewire the Overview tab**

In `arena/app.py` add the helper near the other `_…` helpers:

```python
def _detail_markdown(rows: list, slug: str) -> tuple[str, str]:
    """(markdown, bibtex) for the selected system's detail strip. ('','') if absent."""
    matches = [r for r in rows if r.system_slug == slug]
    if not matches:
        return "", ""
    r = matches[0]
    level = "inference ★" if any(m.reproduction_level == "inference" for m in matches) else "scoring ✔"
    params = f" · **{r.params_millions:g}M params**" if r.params_millions is not None else ""
    links = []
    if r.paper_url:
        links.append(f"[📄 arXiv:{r.paper_arxiv_id}]({r.paper_url})")
    if r.checkpoint_url:
        links.append(f"[🔗 checkpoint]({r.checkpoint_url})")
    md = f"**{r.system_name}** — {r.description}{params} · _{level}_\n\n" + " · ".join(links)
    return md, (r.paper_bibtex or "")
```

Then in `build_demo`, replace the three `gr.DataFrame` tier widgets in the Overview tab with `gradio_leaderboard.Leaderboard` widgets and add the chip row + detail strip. Concretely, in the `with gr.Tab("Overview"):` block:

```python
            from gradio_leaderboard import Leaderboard, SelectColumns

            ds_chips = gr.Markdown()
            gr.Markdown("## 🥇 Gold (coverage = 100%)")
            gold_lb = Leaderboard(
                value=pd.DataFrame(),
                pinned_columns=2,                 # Rank + System frozen
                search_columns=["System"],
                select_columns=SelectColumns(default_selection=None, allow=True),
            )
            gr.Markdown("## 🥈 Silver (coverage ≥ 50%)")
            silver_lb = Leaderboard(value=pd.DataFrame(), pinned_columns=2, search_columns=["System"])
            gr.Markdown("## 🥉 Bronze (coverage ≥ 0%)")
            bronze_lb = Leaderboard(value=pd.DataFrame(), pinned_columns=2, search_columns=["System"])

            gr.Markdown("### System detail")
            detail_pick = gr.Dropdown(label="Show details for system", choices=[], value=None)
            detail_md = gr.Markdown()
            detail_bibtex = gr.Code(label="BibTeX", language=None)
```

Replace the per-tier DataFrame builder `_overview_dfs` so it returns `leaderboard.overview_dataframe(...)` per tier (using `assign_tiers` to split slugs into tiers and `global_rank` for the `ranks` arg):

```python
        def _overview_dfs(state: ArenaState, selected_view: str):
            from ranking import assign_tiers, global_rank
            import leaderboard as lb
            tiers = state.manifest.get("tiers", [])
            core = [e["id"] for e in state.manifest.get("core_set", [])]
            primary = state.manifest.get("metrics_in_use", ["eer_percent"])[0]
            ranks = global_rank(state.rows, state.manifest, selected_view)
            tier_of = assign_tiers(state.rows, tiers, core)
            dfs = []
            for t in tiers:
                slugs = [s for s, name in tier_of.items() if name == t["name"]]
                slugs.sort(key=lambda s: ranks.get(s, {}).get("place", 10**9))
                dfs.append(lb.overview_dataframe(state.rows, slugs, core, primary, ranks))
            return dfs  # ordered like tiers
```

Update `_populate`/`_initial`/`_do_refresh` to also set `ds_chips` (via `lb.dataset_chip_links(core)`) and the `detail_pick` choices (all `system_slug`s). Wire `detail_pick.change` to call `_detail_markdown(state.rows, slug)` → `(detail_md, detail_bibtex)`. Add the totals header below.

- [ ] **Step 5: Add the totals header**

Replace the standalone `ts_md` line with a totals string. Add helper:

```python
def _totals(state: ArenaState) -> str:
    n_systems = len({r.system_slug for r in state.rows})
    n_datasets = len({r.dataset_id for r in state.rows})
    ts = state.loaded_at.isoformat(timespec="seconds")
    return f"**{n_systems} systems · {n_datasets} datasets · Last refreshed {ts}Z**"
```

and use `_totals(state)` for the top `ts_md` Markdown in `_render`/`_refresh`.

- [ ] **Step 6: Run the smoke test + a build test**

Add to `arena/tests/test_app_overview.py`:

```python
def test_build_demo_constructs():
    import app
    demo = app.build_demo()   # must not raise
    assert demo is not None
```

Run (from `arena/`): `pytest tests/test_app_overview.py -v`
Expected: PASS (detail tests + build test). If `gradio_leaderboard` import fails in the test env, install it first (Step 1).

- [ ] **Step 7: Manual visual check**

Run (from `arena/`): `python -c "import app; app.build_demo().launch(server_port=7860)"` and open `http://localhost:7860`.
Expected: Overview shows the random-baseline row under Bronze; Rank+System pinned; System name links to the checkpoint repo; dataset chips clickable above; selecting the system in the dropdown shows description + links + a copy-able BibTeX block; totals header reads "1 systems · 1 datasets · …". Ctrl-C to stop.

- [ ] **Step 8: Commit**

```bash
git add arena/requirements.txt arena/app.py arena/tests/test_app_overview.py
git commit -m "feat(arena): MTEB-style Overview with pinned cols, chips, detail strip, totals"
```

---

## Task 6: Wire Per-dataset tab to the leaderboard component (arena)

**Files:**
- Modify: `arena/leaderboard.py` (add `per_dataset_dataframe`)
- Modify: `arena/app.py` (Per dataset tab)
- Test: `arena/tests/test_leaderboard.py`

- [ ] **Step 1: Write the failing test**

Append to `arena/tests/test_leaderboard.py`:

```python
def test_per_dataset_dataframe():
    from leaderboard import per_dataset_dataframe
    rows = [_row("aasist", "Org/A", 0.9), _row("rnd", "Org/A", 49.0), _row("aasist", "Org/B", 2.0)]
    df = per_dataset_dataframe(rows, "Org/A", ["eer_percent"])
    assert "System" in df.columns and "eer_percent" in df.columns
    assert set(df["eer_percent"]) == {0.9, 49.0}
    assert "huggingface.co/aasist" in df.iloc[0]["System"]
```

- [ ] **Step 2: Run test to verify it fails**

Run (from `arena/`): `pytest tests/test_leaderboard.py -k per_dataset -v`
Expected: FAIL — `ImportError: cannot import name 'per_dataset_dataframe'`.

- [ ] **Step 3: Implement `per_dataset_dataframe`**

Append to `arena/leaderboard.py`:

```python
def per_dataset_dataframe(
    rows: Iterable[Row],
    dataset_id: str,
    metrics_in_use: list[str],
) -> pd.DataFrame:
    """System (md link) + each present metric + Links, for one dataset."""
    matching = [r for r in rows if r.dataset_id == dataset_id]
    all_metrics: set[str] = set()
    for r in matching:
        all_metrics.update(r.scores.keys())
    ordered = [m for m in metrics_in_use if m in all_metrics]
    ordered += sorted(all_metrics - set(ordered))
    records = []
    for r in matching:
        rec = {"System": _system_cell(r.system_name, r.checkpoint_url, r.reproduction_level)}
        for m in ordered:
            rec[m] = r.scores.get(m)
        rec["Links"] = _links_cell(r.paper_url, r.paper_bibtex)
        records.append(rec)
    columns = ["System"] + ordered + ["Links"]
    return pd.DataFrame(records, columns=columns)
```

- [ ] **Step 4: Run test to verify it passes**

Run (from `arena/`): `pytest tests/test_leaderboard.py -k per_dataset -v`
Expected: PASS.

- [ ] **Step 5: Rewire the Per dataset tab**

In `arena/app.py`, in the `with gr.Tab("Per dataset"):` block, replace `per_ds_df = gr.DataFrame(...)` with a `Leaderboard(value=pd.DataFrame(), pinned_columns=1, search_columns=["System"])` and add a `ds_chips_pd = gr.Markdown()` chip line. Update `_per_dataset_data` to call `leaderboard.per_dataset_dataframe(state.rows, dataset_id, metrics)` and `_on_dataset_change` to also refresh the chip line for the single selected dataset.

- [ ] **Step 6: Run build test + manual check**

Run (from `arena/`): `pytest tests/test_app_overview.py::test_build_demo_constructs -v`
Expected: PASS. Launch as in Task 5 Step 7 and confirm Per dataset renders the row with a clickable System link.

- [ ] **Step 7: Commit**

```bash
git add arena/leaderboard.py arena/app.py arena/tests/test_leaderboard.py
git commit -m "feat(arena): MTEB-style Per-dataset tab"
```

---

## Slice 1 checkpoint (manual)

Deploy to the Space (push `arena/`), hard-refresh, confirm: Overview + Per-dataset render the random baseline with pinned Rank/System, clickable model + dataset chips, working detail strip with BibTeX copy, totals header. **Stop and verify before Slice 2.**

---

# SLICE 2 — Submit tab + repo docs

## Task 7: Write canonical contribution docs (package)

**Files:**
- Create: `docs/submitting/submit-model.md`
- Create: `docs/submitting/submit-dataset.md`

- [ ] **Step 1: Write `submit-model.md`**

Create `docs/submitting/submit-model.md`:

```markdown
# Submit a model

1. **Wrap your model** as an `AntiSpoofingModel` subclass (`load`, `score`/`score_batch`, `unload`). Higher score = more bonafide.
2. **Install:** `pip install speech-spoof-bench`.
3. **Run the benchmark:**
   ```bash
   speech-spoof-bench run --model-module mypkg.mymod:MyModel --datasets all
   ```
   This writes `results/<dataset>/scores.txt` + `result.yaml`.
4. **Upload `scores.txt`** to your HF model repo under
   `.eval_results/<dataset-org>/<dataset-name>/scores.txt`.
5. **Author `meta.yaml`** describing your system, including the optional
   `system.params_millions` (your model's parameter count, in millions — used by
   the "By model size" tab).
6. **Submit** (runs + uploads + opens the PR):
   ```bash
   speech-spoof-bench submit \
     --model-module mypkg.mymod:MyModel --datasets all \
     --model-repo <you>/<repo> --submission-meta meta.yaml \
     --hf-username <you> --contact <you@example.com> \
     --params-millions 52.3
   ```
7. **Verification:** a maintainer runs `reproduce --scoring` (fast; mandatory)
   and optionally `--inference` (full re-run; upgrades the ★ badge), then merges.
8. **After merge:** paste the badge snippet from the post-merge comment into your
   model README to link back to the Arena.

> Can't run the full benchmark yourself? See the note at the top of the Submit tab.
```

- [ ] **Step 2: Write `submit-dataset.md`**

Create `docs/submitting/submit-dataset.md`:

```markdown
# Submit a dataset

1. **Redistribution check:** the dataset must be redistributable under its
   upstream license (§1.8). Loader-only repos are out of scope.
2. **Scaffold:**
   ```bash
   speech-spoof-bench scaffold-dataset --name <Source><Year>_<Partition> --output-dir ./<name>
   ```
3. **Build parquet** to the canonical schema `{path, audio(16kHz), label[bonafide,spoof], notes(JSON w/ utterance_id)}` (§1.2).
4. **README frontmatter** (incl. `arena-ready` tag + `arxiv:` list), **`eval.yaml`** (metrics list), and a **Citation** block with arXiv link + BibTeX.
5. **Validate** until green:
   ```bash
   speech-spoof-bench validate-dataset ./<name>
   ```
6. **Push** to `huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/<name>`, then
   open a PR on `arena-manifest` adding it under `core_set` or `extended` with a pinned `revision`.
7. **Core vs Extended:** Core datasets count toward tier coverage and global rank;
   Extended are shown but don't gate tiers.
```

- [ ] **Step 3: Commit**

```bash
git add docs/submitting/submit-model.md docs/submitting/submit-dataset.md
git commit -m "docs: add submit-model and submit-dataset contribution guides"
```

---

## Task 8: `docs_fetch` module (arena, pure-ish with injectable fetcher)

**Files:**
- Create: `arena/docs_fetch.py`
- Test: `arena/tests/test_docs_fetch.py`

- [ ] **Step 1: Write the failing test**

Create `arena/tests/test_docs_fetch.py`:

```python
import docs_fetch


def test_resolve_pin_reads_requirements(tmp_path):
    req = tmp_path / "requirements.txt"
    req.write_text(
        "gradio==6.14.0\n"
        "speech-spoof-bench @ git+https://github.com/lab260ru/speech_spoof_bench.git@deadbeef1234\n"
    )
    assert docs_fetch.resolve_pin(req) == "deadbeef1234"


def test_resolve_pin_defaults_to_main(tmp_path):
    req = tmp_path / "requirements.txt"
    req.write_text("gradio==6.14.0\n")
    assert docs_fetch.resolve_pin(req) == "main"


def test_get_doc_uses_fetcher_and_caches():
    calls = []

    def fake_fetcher(url):
        calls.append(url)
        return "# Hello"

    docs_fetch._CACHE.clear()
    out = docs_fetch.get_doc("submit-model", ref="abc123", fetcher=fake_fetcher)
    assert out == "# Hello"
    assert "abc123" in calls[0] and "submit-model.md" in calls[0]
    # second call cached → fetcher not called again
    docs_fetch.get_doc("submit-model", ref="abc123", fetcher=fake_fetcher)
    assert len(calls) == 1


def test_get_doc_fallback_on_error():
    def boom(url):
        raise RuntimeError("offline")

    docs_fetch._CACHE.clear()
    out = docs_fetch.get_doc("submit-model", ref="abc123", fetcher=boom)
    assert "github.com" in out.lower()  # fallback links to the docs on GitHub
```

- [ ] **Step 2: Run test to verify it fails**

Run (from `arena/`): `pytest tests/test_docs_fetch.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'docs_fetch'`.

- [ ] **Step 3: Implement `docs_fetch.py`**

Create `arena/docs_fetch.py`:

```python
"""Fetch the Submit-tab markdown from the speech-spoof-bench repo at the pinned sha."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

import urllib.request

_REPO = "lab260ru/speech_spoof_bench"
_RAW = "https://raw.githubusercontent.com/{repo}/{ref}/docs/submitting/{name}.md"
_PIN_RE = re.compile(r"speech[-_]spoof[-_]bench\s*@\s*git\+\S+@([0-9a-fA-F]+)")
_CACHE: dict[tuple[str, str], str] = {}


def resolve_pin(requirements_path: str | Path = "requirements.txt") -> str:
    """Return the sha pinned for speech-spoof-bench in requirements.txt, or 'main'."""
    try:
        text = Path(requirements_path).read_text()
    except OSError:
        return "main"
    m = _PIN_RE.search(text)
    return m.group(1) if m else "main"


def _http_get(url: str) -> str:
    with urllib.request.urlopen(url, timeout=10) as resp:  # noqa: S310 (trusted host)
        return resp.read().decode("utf-8")


def get_doc(name: str, ref: str, fetcher: Callable[[str], str] = _http_get) -> str:
    """Fetch docs/submitting/<name>.md at <ref>. Cached per (name, ref).
    On failure returns a short fallback linking to the docs on GitHub."""
    key = (name, ref)
    if key in _CACHE:
        return _CACHE[key]
    url = _RAW.format(repo=_REPO, ref=ref, name=name)
    try:
        out = fetcher(url)
    except Exception:  # noqa: BLE001 — any network/parse error → graceful fallback
        out = (
            f"_Could not load this guide right now._ "
            f"Read it on GitHub: https://github.com/{_REPO}/blob/{ref}/docs/submitting/{name}.md"
        )
    _CACHE[key] = out
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run (from `arena/`): `pytest tests/test_docs_fetch.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add arena/docs_fetch.py arena/tests/test_docs_fetch.py
git commit -m "feat(arena): docs_fetch for Submit-tab guides at pinned sha"
```

---

## Task 9: Wire the Submit tab (arena)

**Files:**
- Modify: `arena/app.py`
- Test: `arena/tests/test_app_overview.py`

- [ ] **Step 1: Implement the Submit tab**

In `arena/app.py` `build_demo`, add after the Per dataset tab:

```python
            with gr.Tab("Submit"):
                import docs_fetch
                gr.Markdown(
                    "> 💡 **Can't run the full benchmark yourself?** If you can run your "
                    "model over the complete dataset(s), email the maintainer "
                    "(see About) and we'll consider running your model for you."
                )
                _ref = docs_fetch.resolve_pin("requirements.txt")
                with gr.Tabs():
                    with gr.Tab("Submit a model"):
                        gr.Markdown(docs_fetch.get_doc("submit-model", _ref))
                    with gr.Tab("Submit a dataset"):
                        gr.Markdown(docs_fetch.get_doc("submit-dataset", _ref))
```

- [ ] **Step 2: Verify build + manual check**

Run (from `arena/`): `pytest tests/test_app_overview.py::test_build_demo_constructs -v`
Expected: PASS. Launch and confirm the Submit tab shows the callout + both guides (or the fallback link if offline).

- [ ] **Step 3: Commit**

```bash
git add arena/app.py
git commit -m "feat(arena): Submit tab renders repo docs + write-us callout"
```

---

## Slice 2 checkpoint (manual)

Push `arena/`; confirm Submit tab renders the two guides at the pinned sha and the callout shows. **Stop and verify before Slice 3.**

---

# SLICE 3 — By model size tab

## Task 10: `charts.size_series` (arena, pure)

**Files:**
- Create: `arena/charts.py`
- Test: `arena/tests/test_charts.py`

- [ ] **Step 1: Write the failing test**

Create `arena/tests/test_charts.py`:

```python
from schema import Row
from charts import size_series


def _row(slug, ds, eer, params=None, **kw):
    return Row(
        system_slug=slug, system_name=slug.upper(), dataset_id=ds, revision="r",
        scores={"eer_percent": eer}, reproduction_level="scoring",
        submitted_at="2026-05-20", submission_url="u", n_trials=100,
        params_millions=params, **kw,
    )


_MANIFEST = {
    "core_set": [{"id": "Org/A", "revision": "r"}, {"id": "Org/B", "revision": "r"}],
    "metrics_in_use": ["eer_percent"], "ranking": {"default_view": "aggregated"},
}


def test_size_series_core_mean_full_vs_partial_coverage():
    rows = [
        _row("aasist", "Org/A", 0.9, params=52.3), _row("aasist", "Org/B", 2.0, params=52.3),
        _row("rnd", "Org/A", 49.0, params=0.1),  # partial coverage (no Org/B)
    ]
    pts = size_series(rows, _MANIFEST, view="aggregated", y_selector="__core_mean__")
    by_slug = {p["slug"]: p for p in pts}
    assert by_slug["aasist"]["params"] == 52.3
    assert by_slug["aasist"]["full_coverage"] is True
    assert by_slug["rnd"]["full_coverage"] is False
    # core mean for aasist is (0.9+2.0)/2 = 1.45 under aggregated (equal weight)
    assert abs(by_slug["aasist"]["y"] - 1.45) < 1e-6


def test_size_series_single_dataset():
    rows = [_row("aasist", "Org/A", 0.9, params=52.3), _row("aasist", "Org/B", 2.0, params=52.3)]
    pts = size_series(rows, _MANIFEST, view="aggregated", y_selector="Org/A")
    assert pts[0]["y"] == 0.9


def test_size_series_excludes_missing_params():
    rows = [_row("noparam", "Org/A", 1.0, params=None)]
    pts = size_series(rows, _MANIFEST, view="aggregated", y_selector="Org/A")
    assert pts == []
```

- [ ] **Step 2: Run test to verify it fails**

Run (from `arena/`): `pytest tests/test_charts.py -k size -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'charts'`.

- [ ] **Step 3: Implement `charts.size_series`**

Create `arena/charts.py`:

```python
"""Pure chart-data builders. No Gradio, no plotly objects returned here —
just plain lists/dicts the UI turns into a figure."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from ranking import global_scores
from schema import Row

CORE_MEAN = "__core_mean__"


def size_series(rows: Iterable[Row], manifest: dict, view: str, y_selector: str) -> list[dict]:
    """One point per system with a declared params_millions.
    y = core-mean (via global_scores, view-aware) when y_selector == CORE_MEAN,
    else the system's primary-metric value on that single dataset_id."""
    rows = list(rows)
    core = [e["id"] for e in manifest.get("core_set", [])]
    primary = (manifest.get("metrics_in_use") or ["eer_percent"])[0]

    params: dict[str, float] = {}
    covered: dict[str, set[str]] = defaultdict(set)
    single: dict[str, float] = {}
    names: dict[str, str] = {}
    for r in rows:
        if r.params_millions is not None:
            params[r.system_slug] = r.params_millions
            names[r.system_slug] = r.system_name
        if r.dataset_id in core:
            covered[r.system_slug].add(r.dataset_id)
        if y_selector != CORE_MEAN and r.dataset_id == y_selector:
            v = r.scores.get(primary)
            if v is not None:
                single[r.system_slug] = v

    mean = global_scores(rows, manifest, view) if y_selector == CORE_MEAN else {}
    pts = []
    for slug, p in params.items():
        if y_selector == CORE_MEAN:
            y = mean.get(slug)
        else:
            y = single.get(slug)
        if y is None:
            continue
        pts.append({
            "slug": slug, "name": names.get(slug, slug), "params": p, "y": y,
            "full_coverage": len(covered.get(slug, set())) == len(core) and len(core) > 0,
        })
    return pts
```

- [ ] **Step 4: Run test to verify it passes**

Run (from `arena/`): `pytest tests/test_charts.py -k size -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add arena/charts.py arena/tests/test_charts.py
git commit -m "feat(arena): size_series chart-data builder (view-aware core mean)"
```

---

## Task 11: Wire the By model size tab (arena)

**Files:**
- Modify: `arena/app.py`
- Test: `arena/tests/test_app_overview.py`

- [ ] **Step 1: Implement the tab**

In `arena/app.py` `build_demo`, add a tab after Per dataset:

```python
            with gr.Tab("By model size"):
                size_view = gr.Radio(choices=["aggregated", "pooled"],
                                     value=_default_view, label="Ranking view")
                size_y = gr.Dropdown(label="Y axis", choices=[], value=None)
                size_plot = gr.Plot()
                size_note = gr.Markdown()
```

Add a figure builder helper:

```python
def _size_figure(state, view, y_selector):
    import plotly.graph_objects as go
    import charts
    sel = charts.CORE_MEAN if (not y_selector or y_selector == "Core mean") else y_selector
    pts = charts.size_series(state.rows, state.manifest, view, sel)
    fig = go.Figure()
    for full in (True, False):
        sub = [p for p in pts if p["full_coverage"] is full]
        if sub:
            fig.add_trace(go.Scatter(
                x=[p["params"] for p in sub], y=[p["y"] for p in sub],
                text=[p["name"] for p in sub], mode="markers+text", textposition="top center",
                marker=dict(size=12, symbol="circle" if full else "circle-open"),
                name="full coverage" if full else "partial",
            ))
    fig.update_xaxes(type="log", title="parameters (M, log)")
    fig.update_yaxes(title="score (lower better)")
    declared = {p["slug"] for p in pts}
    missing = sorted({r.system_name for r in state.rows if r.params_millions is None and r.system_slug not in declared})
    note = ("_No size declared:_ " + ", ".join(missing)) if missing else ""
    return fig, note
```

Wire `size_y` choices to `["Core mean"] + [short(core ids)]`, default `"Core mean"`; on `demo.load`/refresh and on `size_view.change`/`size_y.change`, call `_size_figure`. Add `size_plot`, `size_note` to the relevant output lists.

- [ ] **Step 2: Verify build + manual check**

Run (from `arena/`): `pytest tests/test_app_overview.py::test_build_demo_constructs -v`
Expected: PASS. Launch; with no declared params the chart is empty and the note lists "random-baseline". (Add a `params_millions` to the live submission later to see a point.)

- [ ] **Step 3: Commit**

```bash
git add arena/app.py
git commit -m "feat(arena): By model size tab (scatter + coverage markers + no-size note)"
```

---

## Slice 3 checkpoint (manual)

Push `arena/`; confirm the By model size tab loads, the view radio matches Overview, and the "no size declared" note appears for the param-less baseline. **Stop and verify before Slice 4.**

---

# SLICE 4 — Over time tab

## Task 12: `charts.sota_timeline` (arena, pure)

**Files:**
- Modify: `arena/charts.py`
- Test: `arena/tests/test_charts.py`

- [ ] **Step 1: Write the failing test**

Append to `arena/tests/test_charts.py`:

```python
from charts import sota_timeline


def test_sota_timeline_best_to_date():
    rows = [
        _row("rnd", "Org/A", 49.0), _row("rawnet", "Org/A", 1.4), _row("aasist", "Org/A", 0.9),
    ]
    rows[0] = rows[0]._replace if hasattr(rows[0], "_replace") else rows[0]
    # set dates explicitly
    from dataclasses import replace
    rows = [replace(rows[0], submitted_at="2026-05-01"),
            replace(rows[1], submitted_at="2026-05-10"),
            replace(rows[2], submitted_at="2026-05-20")]
    pts = sota_timeline(rows, "Org/A", metric="eer_percent", lower_is_better=True)
    # best-so-far is monotonically non-increasing for EER
    ys = [p["best"] for p in pts]
    assert ys == [49.0, 1.4, 0.9]
    assert pts[0]["date"] == "2026-05-01"


def test_sota_timeline_empty():
    assert sota_timeline([], "Org/A", metric="eer_percent", lower_is_better=True) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run (from `arena/`): `pytest tests/test_charts.py -k sota -v`
Expected: FAIL — `ImportError: cannot import name 'sota_timeline'`.

- [ ] **Step 3: Implement `sota_timeline`**

Append to `arena/charts.py`:

```python
def sota_timeline(rows: Iterable[Row], dataset_id: str, metric: str, lower_is_better: bool) -> list[dict]:
    """Best metric value achieved on dataset_id by each submission date (sorted)."""
    pts = sorted(
        ({"date": r.submitted_at, "value": r.scores[metric], "name": r.system_name}
         for r in rows if r.dataset_id == dataset_id and metric in r.scores),
        key=lambda d: d["date"],
    )
    out, best = [], None
    for p in pts:
        if best is None:
            best = p["value"]
        else:
            best = min(best, p["value"]) if lower_is_better else max(best, p["value"])
        out.append({"date": p["date"], "value": p["value"], "best": best, "name": p["name"]})
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run (from `arena/`): `pytest tests/test_charts.py -k sota -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add arena/charts.py arena/tests/test_charts.py
git commit -m "feat(arena): sota_timeline best-to-date series"
```

---

## Task 13: `events.build_events` (arena, pure)

**Files:**
- Create: `arena/events.py`
- Test: `arena/tests/test_events.py`

- [ ] **Step 1: Write the failing test**

Create `arena/tests/test_events.py`:

```python
from dataclasses import replace

from schema import Row
from events import build_events


def _row(slug, ds, eer, date, level="scoring", repro_at=""):
    return Row(
        system_slug=slug, system_name=slug.upper(), dataset_id=ds, revision="r",
        scores={"eer_percent": eer}, reproduction_level=level,
        submitted_at=date, submission_url="u", n_trials=100, reproduced_at=repro_at,
    )


def test_build_events_merges_and_sorts_desc():
    rows = [_row("aasist", "Org/A", 0.9, "2026-05-20", level="inference", repro_at="2026-05-24")]
    changelog = {"events": [
        {"date": "2026-05-18", "type": "dataset_added", "text": "InTheWild joined Core"},
        {"date": "2026-05-22", "type": "dataset_repin", "dataset": "Org/B", "text": "re-pinned"},
    ]}
    evs = build_events(rows, manifest={}, changelog=changelog)
    dates = [e["date"] for e in evs]
    assert dates == sorted(dates, reverse=True)  # newest first
    kinds = {e["kind"] for e in evs}
    assert {"model_added", "verification_upgraded", "dataset_added", "dataset_repin"} <= kinds


def test_build_events_no_changelog():
    rows = [_row("rnd", "Org/A", 49.0, "2026-05-01")]
    evs = build_events(rows, manifest={}, changelog=None)
    assert len(evs) == 1 and evs[0]["kind"] == "model_added"
```

- [ ] **Step 2: Run test to verify it fails**

Run (from `arena/`): `pytest tests/test_events.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'events'`.

- [ ] **Step 3: Implement `events.py`**

Create `arena/events.py`:

```python
"""Build the Over-time activity feed: auto-derived model events + curated changelog."""

from __future__ import annotations

from typing import Iterable

from schema import Row

_EMOJI = {
    "model_added": "🆕", "verification_upgraded": "⬆️",
    "dataset_repin": "↻", "dataset_added": "➕",
    "metric_added": "📏", "note": "✍️",
}


def _short(dataset_id: str) -> str:
    return dataset_id.split("/")[-1]


def build_events(rows: Iterable[Row], manifest: dict, changelog: dict | None) -> list[dict]:
    """Return date-desc list of {date, kind, emoji, text}."""
    events: list[dict] = []
    for r in rows:
        metric = next(iter(r.scores), "")
        val = r.scores.get(metric)
        badge = "★" if r.reproduction_level == "inference" else "✔"
        events.append({
            "date": r.submitted_at, "kind": "model_added", "emoji": _EMOJI["model_added"],
            "text": f"{r.system_name} added on {_short(r.dataset_id)} — {metric} {val} {badge}",
        })
        if r.reproduction_level == "inference" and r.reproduced_at:
            events.append({
                "date": r.reproduced_at, "kind": "verification_upgraded",
                "emoji": _EMOJI["verification_upgraded"],
                "text": f"{r.system_name} on {_short(r.dataset_id)} upgraded to inference ★",
            })

    for e in (changelog or {}).get("events", []):
        kind = e.get("type", "note")
        events.append({
            "date": str(e.get("date", "")), "kind": kind,
            "emoji": _EMOJI.get(kind, "•"), "text": e.get("text", ""),
        })

    return sorted(events, key=lambda e: e["date"], reverse=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run (from `arena/`): `pytest tests/test_events.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add arena/events.py arena/tests/test_events.py
git commit -m "feat(arena): build_events hybrid activity feed"
```

---

## Task 14: `changelog.py` reader + seed `CHANGELOG.yaml` (arena + manifest)

**Files:**
- Create: `arena/changelog.py`
- Create: `arena-manifest/CHANGELOG.yaml`
- Test: `arena/tests/test_changelog.py`

- [ ] **Step 1: Write the failing test**

Create `arena/tests/test_changelog.py`:

```python
import changelog


def test_parse_changelog_text():
    text = "events:\n  - {date: 2026-05-18, type: note, text: hi}\n"
    out = changelog.parse_changelog(text)
    assert out["events"][0]["text"] == "hi"


def test_fetch_changelog_missing_returns_none(monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError("404")
    monkeypatch.setattr(changelog, "_download", boom)
    assert changelog.fetch_changelog() is None
```

- [ ] **Step 2: Run test to verify it fails**

Run (from `arena/`): `pytest tests/test_changelog.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'changelog'`.

- [ ] **Step 3: Implement `changelog.py`**

Create `arena/changelog.py`:

```python
"""Optional CHANGELOG.yaml reader from the arena-manifest repo."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from huggingface_hub import hf_hub_download

_REPO = "SpeechAntiSpoofingBenchmarks/arena-manifest"
_FILENAME = "CHANGELOG.yaml"


def parse_changelog(text: str) -> dict[str, Any]:
    data = yaml.safe_load(text) or {}
    data.setdefault("events", [])
    return data


def _download() -> str:
    local = hf_hub_download(repo_id=_REPO, repo_type="dataset", filename=_FILENAME)
    return Path(local).read_text()


def fetch_changelog() -> dict | None:
    """Return parsed changelog, or None if the file doesn't exist / can't be read."""
    try:
        return parse_changelog(_download())
    except Exception:  # noqa: BLE001 — missing file or network → feed is auto-events only
        return None
```

- [ ] **Step 4: Run test to verify it passes**

Run (from `arena/`): `pytest tests/test_changelog.py -v`
Expected: PASS.

- [ ] **Step 5: Seed the changelog in the manifest repo**

Create `arena-manifest/CHANGELOG.yaml`:

```yaml
events:
  - {date: 2026-05-29, type: note, text: "Arena Phase 10 — leaderboard, by-size, over-time, submit tabs live"}
```

- [ ] **Step 6: Commit (two repos)**

```bash
git add arena/changelog.py arena/tests/test_changelog.py
git commit -m "feat(arena): optional CHANGELOG.yaml reader"
cd ../arena-manifest && git add CHANGELOG.yaml && git commit -m "chore: seed CHANGELOG.yaml" && cd -
```

---

## Task 15: Wire the Over time tab (arena)

**Files:**
- Modify: `arena/app.py`
- Modify: `arena/ingest.py` (carry changelog into `ArenaState`) — see note
- Test: `arena/tests/test_app_overview.py`

Note: rather than enlarge `ArenaState`, fetch the changelog lazily in the tab's render fn (cached by `changelog.py`'s own download cache via `hf_hub_download`). Keep `ArenaState` unchanged.

- [ ] **Step 1: Implement the tab**

In `arena/app.py` `build_demo`, add after By model size:

```python
            with gr.Tab("Over time"):
                ot_dataset = gr.Dropdown(label="Dataset", choices=[], value=None)
                ot_plot = gr.Plot()
                ot_log = gr.Markdown()
```

Add helpers:

```python
def _over_time_figure(state, dataset_id):
    import plotly.graph_objects as go
    import charts
    from ranking import _lower_is_better
    if not dataset_id:
        return go.Figure(), ""
    primary = (state.manifest.get("metrics_in_use") or ["eer_percent"])[0]
    pts = charts.sota_timeline(state.rows, dataset_id, primary, _lower_is_better(primary))
    fig = go.Figure()
    if pts:
        fig.add_trace(go.Scatter(x=[p["date"] for p in pts], y=[p["best"] for p in pts],
                                 mode="lines+markers", line_shape="hv", name="best-to-date"))
    fig.update_yaxes(title=f"best {primary} (lower better)")
    fig.update_xaxes(title="submission date")
    return fig, _activity_markdown(state)


def _activity_markdown(state):
    import changelog as cl
    import events as ev
    feed = ev.build_events(state.rows, state.manifest, cl.fetch_changelog())
    if not feed:
        return "_No activity yet._"
    lines = [f"- `{e['date']}` {e['emoji']} {e['text']}" for e in feed]
    return "### Activity log\n" + "\n".join(lines)
```

Wire `ot_dataset` choices to the full dataset id list (defaulting to the first core id), call `_over_time_figure` on load/refresh and on `ot_dataset.change`, outputting `(ot_plot, ot_log)`.

- [ ] **Step 2: Verify build + manual check**

Run (from `arena/`): `pytest tests/test_app_overview.py::test_build_demo_constructs -v`
Expected: PASS. Launch; Over time shows a single-point line for the baseline on the first dataset and an activity log with the model-added entry + the seeded changelog note.

- [ ] **Step 3: Commit**

```bash
git add arena/app.py
git commit -m "feat(arena): Over time tab (sota timeline + activity log)"
```

---

# Finalization

## Task 16: Full arena test sweep

- [ ] **Step 1: Run everything**

Run (from `arena/`): `pytest -q`
Expected: PASS (all suites: ingest, leaderboard, charts, events, docs_fetch, changelog, app, ranking, schema, badges, cache_store, webhook).

- [ ] **Step 2: Run package tests**

Run (from `speech-spoof-bench/`): `pytest -q`
Expected: PASS.

## Task 17: Update ROADMAP

**Files:**
- Modify: `speech-spoof-bench/docs/roadmap/ROADMAP.md:223-232`

- [ ] **Step 1: Check off Phase 10 items**

Mark the Phase 10 bullets done, adjusting wording where the design diverged (no per-system page; inline detail strip; By model size + Over time + Submit tabs; totals header). Commit:

```bash
git add docs/roadmap/ROADMAP.md
git commit -m "docs(roadmap): mark Phase 10 arena polish complete"
```

## Task 18: Bump the package pin in the Space + deploy

**Files:**
- Modify: `arena/requirements.txt`

- [ ] **Step 1: Find the new package sha**

After pushing the `speech-spoof-bench` package commits (Tasks 1, 2, 7) to GitHub, get the latest sha:

```bash
cd ../speech-spoof-bench && git rev-parse HEAD
```

- [ ] **Step 2: Update the pin**

In `arena/requirements.txt`, set the `speech-spoof-bench @ git+...@<sha>` to the new sha (so the Space's `docs_fetch` resolves the same ref and the schema with `params_millions` is installed).

- [ ] **Step 3: Commit + push the Space**

```bash
git add arena/requirements.txt
git commit -m "chore(arena): bump speech-spoof-bench pin for Phase 10"
# push arena/ to the HF Space remote per existing deploy flow
```

- [ ] **Step 4: Final manual verification on the live Space**

Hard-refresh the Space. Confirm all six tabs render, links work, detail strip + BibTeX copy work, By model size + Over time load, Submit shows the guides at the new pin. **Phase 10 done.**

---

## Self-review notes (author)

- **Spec coverage:** MTEB table (Tasks 4–6) · clickable models+datasets (Task 4 chips + system cell) · inline paper/bibtex/model, no per-system page (Task 5 detail strip) · params source = submitter (Tasks 1–2) · By model size view-aware core mean + coverage markers (Tasks 10–11) · Over time chart + hybrid event log + re-pin source (Tasks 12–15) · Submit renders repo docs + write-us note (Tasks 7–9) · totals header (Task 5) · preservation (only additive schema field + optional CHANGELOG; Tasks 16–18 verify) — all covered.
- **Known divergence to confirm at build time:** `gradio_leaderboard` API (`pinned_columns`, `search_columns`, `SelectColumns`) and whether `pinned_columns` takes an int count or column names — verify against the installed version in Task 5 Step 1; the spec's native-`gr.Dataframe(pinned_columns=[...])` fallback applies if the component misbehaves.
- **Re-pin markers (spec §7.1):** sourced from `type: dataset_repin` changelog entries; surfacing them as vertical lines on the timeline is a polish add-on inside Task 15's figure (add `fig.add_vline` per repin date for the selected dataset) — left as an enhancement so the core timeline ships first.
```
