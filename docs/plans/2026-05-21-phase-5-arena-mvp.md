# Phase 5 — Arena MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the read-only Gradio leaderboard Space at `huggingface.co/spaces/SpeechAntiSpoofingBenchmarks/SpeechAntiSpoofingArena` that consumes `arena-manifest` and each dataset's `submissions/*.yaml`, showing tiered rankings, per-dataset tables, and an About tab with warnings.

**Architecture:** Plain Gradio Blocks app. `ingest.py` fetches manifest + submissions from HF Hub (anonymous), validates each YAML against the JSON Schema bundled with `speech-spoof-bench`, and caches results in memory with a 30-min TTL and a Refresh button. `ranking.py` is pure (no I/O) and produces tier/table data structures consumed by the three tabs. The pip package gets a new `submission.schema.json` + `submission.fetch_submissions(...)` API so the arena does not reimplement validation.

**Tech Stack:** Python 3.11, Gradio, `huggingface_hub`, `pyyaml`, `jsonschema`, `speech-spoof-bench` (pinned via git+sha in `requirements.txt`).

**Reference spec:** `docs/specs/2026-05-21-phase-5-arena-mvp-design.md`.

---

## File map

**Pip package (`speech-spoof-bench/`)** — additive only, no behavior change for existing callers:

| Path | Change | Responsibility |
|---|---|---|
| `src/speech_spoof_bench/schema/submission.schema.json` | create | JSON Schema for v4 `submissions/*.yaml`. Single source of truth. |
| `src/speech_spoof_bench/submission.py` | create | `load_submission_schema()`, `parse_submission(text)`, `list_submission_files(dataset_id)`, `fetch_submission(dataset_id, path)`. Mirrors style of `manifest.py`. |
| `tests/test_submission.py` | create | Unit tests for parse/validate. |

**Arena (`/home/kirill/speech-spoof-bench/arena/`)**:

| Path | Responsibility |
|---|---|
| `pyproject.toml` | dev deps for tests; not used at HF Space runtime |
| `requirements.txt` | runtime deps the HF Space installs at build |
| `schema.py` | `Row`, `Warning`, `ArenaState` dataclasses |
| `ranking.py` | pure functions: `assign_tiers`, `overview_table`, `per_dataset_table` |
| `ingest.py` | `load_state(force_refresh=False)` with TTL + lock + warnings list |
| `app.py` | Gradio Blocks: Overview / Per dataset / About tabs + Refresh button |
| `tests/conftest.py` | fixtures path, monkeypatch helpers |
| `tests/test_ranking.py` | unit tests for ranking |
| `tests/test_ingest.py` | unit tests; HF calls monkeypatched |
| `tests/fixtures/manifest.yaml` | minimal manifest for tests |
| `tests/fixtures/submissions/valid.yaml` | passes schema + has reproduction |
| `tests/fixtures/submissions/missing-reproduction.yaml` | passes schema but `reproduction` absent |
| `tests/fixtures/submissions/schema-invalid.yaml` | missing `system.name` |

---

## Task 1: Add `submission.schema.json` to the pip package

**Files:**
- Create: `speech-spoof-bench/src/speech_spoof_bench/schema/submission.schema.json`

- [ ] **Step 1: Write the schema file**

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "speech-spoof-bench submission (v4)",
  "type": "object",
  "additionalProperties": false,
  "required": ["schema_version", "system", "dataset", "scores", "artifact", "reproduction", "submitter", "submitted_at"],
  "properties": {
    "schema_version": {"type": "integer", "const": 4},
    "system": {
      "type": "object",
      "additionalProperties": false,
      "required": ["name", "slug", "description", "code", "checkpoint", "paper"],
      "properties": {
        "name": {"type": "string", "minLength": 1},
        "slug": {"type": "string", "pattern": "^[a-z0-9][a-z0-9-]*$"},
        "description": {"type": "string"},
        "code": {"type": "string", "format": "uri"},
        "checkpoint": {"type": "string", "format": "uri"},
        "paper": {
          "type": "object",
          "additionalProperties": false,
          "required": ["arxiv_id", "url", "bibtex"],
          "properties": {
            "arxiv_id": {"type": "string", "minLength": 1},
            "url": {"type": "string", "format": "uri"},
            "bibtex": {"type": "string", "minLength": 1}
          }
        }
      }
    },
    "dataset": {
      "type": "object",
      "additionalProperties": false,
      "required": ["id", "revision", "split"],
      "properties": {
        "id": {"type": "string", "pattern": "^[^/]+/[^/]+$"},
        "revision": {"type": "string", "minLength": 1},
        "split": {"type": "string", "minLength": 1}
      }
    },
    "scores": {
      "type": "object",
      "required": ["n_trials", "n_skipped"],
      "properties": {
        "n_trials": {"type": "integer", "minimum": 0},
        "n_skipped": {"type": "integer", "minimum": 0}
      },
      "patternProperties": {
        "^(?!n_trials$|n_skipped$).+$": {"type": "number"}
      },
      "additionalProperties": false
    },
    "artifact": {
      "type": "object",
      "additionalProperties": false,
      "required": ["scores_url", "scores_sha256", "bench_version"],
      "properties": {
        "scores_url": {
          "type": "string",
          "pattern": "^https://huggingface\\.co/[^/]+/[^/]+/resolve/[0-9a-f]{7,40}/"
        },
        "scores_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "bench_version": {"type": "string", "minLength": 1}
      }
    },
    "reproduction": {
      "type": "object",
      "additionalProperties": false,
      "required": ["reproduced_by", "reproduced_at", "reproduced_bench_version", "match"],
      "properties": {
        "reproduced_by": {"type": "string", "minLength": 1},
        "reproduced_at": {"type": "string", "format": "date"},
        "reproduced_bench_version": {"type": "string", "minLength": 1},
        "match": {"enum": ["scoring", "inference"]}
      }
    },
    "submitter": {
      "type": "object",
      "additionalProperties": false,
      "required": ["hf_username", "contact"],
      "properties": {
        "hf_username": {"type": "string", "minLength": 1},
        "contact": {"type": "string", "minLength": 1}
      }
    },
    "submitted_at": {"type": "string", "format": "date"},
    "notes": {"type": "string"}
  }
}
```

- [ ] **Step 2: Sanity-check the schema against the real submission**

Run: `cd /home/kirill/speech-spoof-bench/speech-spoof-bench && python -c "
import json, yaml
from importlib import resources
from jsonschema import validate
schema = json.loads(resources.files('speech_spoof_bench.schema').joinpath('submission.schema.json').read_text())
sub = yaml.safe_load(open('/home/kirill/mnt/drive3_8tb/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA/submissions/random-baseline.yaml'))
validate(sub, schema)
print('ok')
"`

Expected: `ok`.

If it fails, fix the schema to match the actual YAML shape (it's the ground truth at this point). Do not change the YAML.

- [ ] **Step 3: Commit**

```bash
cd /home/kirill/speech-spoof-bench/speech-spoof-bench
git add src/speech_spoof_bench/schema/submission.schema.json
git commit -m "schema: add submission.schema.json (v4)"
```

---

## Task 2: Add `submission.py` to the pip package

**Files:**
- Create: `speech-spoof-bench/src/speech_spoof_bench/submission.py`
- Create: `speech-spoof-bench/tests/test_submission.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_submission.py`:

```python
import json
import textwrap

import pytest
import yaml

from speech_spoof_bench.submission import (
    SubmissionValidationError,
    load_submission_schema,
    parse_submission,
)


VALID_YAML = textwrap.dedent("""
schema_version: 4
system:
  name: random-baseline
  slug: random-baseline
  description: stub
  code: https://github.com/example/x
  checkpoint: https://huggingface.co/example/x
  paper:
    arxiv_id: "1911.01601"
    url: https://arxiv.org/abs/1911.01601
    bibtex: "@misc{x,title={x}}"
dataset:
  id: SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA
  revision: 151aa4c6
  split: test
scores:
  eer_percent: 49.87
  n_trials: 71237
  n_skipped: 0
artifact:
  scores_url: https://huggingface.co/example/x/resolve/abcdef1/path/scores.txt
  scores_sha256: 71ac000c0712a4551873dba87183e746cb9730cd5ab17aaa87892009bde55587
  bench_version: speech-spoof-bench==0.1.0
reproduction:
  reproduced_by: SpeechAntiSpoofingBenchmarks
  reproduced_at: 2026-05-21
  reproduced_bench_version: speech-spoof-bench==0.1.0
  match: scoring
submitter:
  hf_username: example
  contact: e@example.com
submitted_at: 2026-05-21
""")


def test_load_schema_has_v4_const():
    schema = load_submission_schema()
    assert schema["properties"]["schema_version"]["const"] == 4


def test_parse_submission_returns_dict_for_valid_yaml():
    sub = parse_submission(VALID_YAML)
    assert sub["system"]["slug"] == "random-baseline"
    assert sub["scores"]["eer_percent"] == 49.87


def test_parse_submission_rejects_missing_reproduction():
    bad = yaml.safe_load(VALID_YAML)
    del bad["reproduction"]
    with pytest.raises(SubmissionValidationError):
        parse_submission(yaml.safe_dump(bad))


def test_parse_submission_rejects_unpinned_scores_url():
    bad = yaml.safe_load(VALID_YAML)
    bad["artifact"]["scores_url"] = "https://huggingface.co/example/x/resolve/main/path/scores.txt"
    with pytest.raises(SubmissionValidationError):
        parse_submission(yaml.safe_dump(bad))
```

- [ ] **Step 2: Run test, confirm it fails**

Run: `cd /home/kirill/speech-spoof-bench/speech-spoof-bench && pytest tests/test_submission.py -x`
Expected: `ModuleNotFoundError: No module named 'speech_spoof_bench.submission'`.

- [ ] **Step 3: Implement `submission.py`**

Create `src/speech_spoof_bench/submission.py`:

```python
"""Submission YAML loader + schema validator.

Mirrors the shape of `manifest.py`. Public functions:
  - load_submission_schema()
  - parse_submission(text) -> dict
  - list_submission_files(dataset_id) -> list[str]
  - fetch_submission(dataset_id, path) -> dict
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any

import yaml
from huggingface_hub import HfApi, hf_hub_download
from jsonschema import ValidationError, validate

SCHEMA_PACKAGE = "speech_spoof_bench.schema"
SCHEMA_FILENAME = "submission.schema.json"
SUBMISSIONS_DIR = "submissions"
_EXCLUDED_FILENAMES = {"README.md", "results_template.yaml"}


class SubmissionValidationError(ValueError):
    """Raised when a submission YAML fails schema validation."""


def load_submission_schema() -> dict[str, Any]:
    with resources.files(SCHEMA_PACKAGE).joinpath(SCHEMA_FILENAME).open("r") as f:
        return json.load(f)


def parse_submission(text: str) -> dict[str, Any]:
    """Parse YAML text and validate against the submission schema."""
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise SubmissionValidationError("submission YAML is not a mapping")
    try:
        validate(instance=data, schema=load_submission_schema())
    except ValidationError as exc:
        raise SubmissionValidationError(exc.message) from exc
    return data


def list_submission_files(dataset_id: str, *, api: HfApi | None = None) -> list[str]:
    """List `submissions/*.yaml` files in a dataset repo at main.

    Excludes README.md and results_template.yaml.
    """
    api = api or HfApi()
    files = api.list_repo_files(repo_id=dataset_id, repo_type="dataset")
    out: list[str] = []
    for f in files:
        if not f.startswith(SUBMISSIONS_DIR + "/"):
            continue
        if not f.endswith(".yaml"):
            continue
        name = f.rsplit("/", 1)[-1]
        if name in _EXCLUDED_FILENAMES:
            continue
        out.append(f)
    return out


def fetch_submission(dataset_id: str, path: str) -> dict[str, Any]:
    """Download a submission YAML and parse+validate it."""
    local = hf_hub_download(repo_id=dataset_id, filename=path, repo_type="dataset")
    return parse_submission(Path(local).read_text())
```

- [ ] **Step 4: Run tests, confirm pass**

Run: `cd /home/kirill/speech-spoof-bench/speech-spoof-bench && pytest tests/test_submission.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/kirill/speech-spoof-bench/speech-spoof-bench
git add src/speech_spoof_bench/submission.py tests/test_submission.py
git commit -m "submission: schema loader, parser, HF fetch helpers"
```

---

## Task 3: Bootstrap arena workspace (pyproject + requirements + pytest)

**Files:**
- Create: `arena/pyproject.toml`
- Create: `arena/requirements.txt`
- Create: `arena/tests/__init__.py`
- Create: `arena/tests/conftest.py`
- Create: `arena/tests/fixtures/manifest.yaml`
- Create: `arena/tests/fixtures/submissions/valid.yaml`
- Create: `arena/tests/fixtures/submissions/missing-reproduction.yaml`
- Create: `arena/tests/fixtures/submissions/schema-invalid.yaml`

- [ ] **Step 1: Write `requirements.txt`** (runtime — what the HF Space installs)

```
gradio==6.14.0
huggingface_hub>=0.20
pyyaml>=6.0
jsonschema>=4.0
speech-spoof-bench @ git+https://github.com/lab260ru/speech_spoof_bench.git@main
```

Note: replace `lab260ru/speech_spoof_bench` with the actual GitHub repo URL if different. We use `@main` for now and will pin to a sha at Phase 5 close (Task 9).

- [ ] **Step 2: Write `pyproject.toml`** (dev only — for tests; not used at HF runtime)

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "arena"
version = "0.0.0"
description = "SpeechAntiSpoofingBenchmarks Arena (HF Space sources)"
requires-python = ">=3.11"
dependencies = [
    "gradio==6.14.0",
    "huggingface_hub>=0.20",
    "pyyaml>=6.0",
    "jsonschema>=4.0",
    "speech-spoof-bench",
]

[project.optional-dependencies]
dev = ["pytest>=7.0"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra"
```

- [ ] **Step 3: Write fixture `manifest.yaml`**

Create `tests/fixtures/manifest.yaml`:

```yaml
ranking_version: v1
schema_version: 1
metrics_in_use:
  - eer_percent
tiers:
  - {name: gold,   min_coverage: 1.0}
  - {name: silver, min_coverage: 0.5}
  - {name: bronze, min_coverage: 0.0}
core_set:
  - id: SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA
    revision: 9b2040e8
  - id: SpeechAntiSpoofingBenchmarks/ASVspoof2021_LA
    revision: deadbeef
extended:
  - id: SpeechAntiSpoofingBenchmarks/InTheWild
    revision: cafef00d
```

- [ ] **Step 4: Write fixture `valid.yaml`**

Create `tests/fixtures/submissions/valid.yaml`:

```yaml
schema_version: 4
system:
  name: random-baseline
  slug: random-baseline
  description: stub
  code: https://github.com/example/x
  checkpoint: https://huggingface.co/example/x
  paper:
    arxiv_id: "1911.01601"
    url: https://arxiv.org/abs/1911.01601
    bibtex: "@misc{x,title={x}}"
dataset:
  id: SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA
  revision: 9b2040e8
  split: test
scores:
  eer_percent: 49.87
  n_trials: 71237
  n_skipped: 0
artifact:
  scores_url: https://huggingface.co/example/x/resolve/abcdef1/path/scores.txt
  scores_sha256: 71ac000c0712a4551873dba87183e746cb9730cd5ab17aaa87892009bde55587
  bench_version: speech-spoof-bench==0.1.0
reproduction:
  reproduced_by: SpeechAntiSpoofingBenchmarks
  reproduced_at: 2026-05-21
  reproduced_bench_version: speech-spoof-bench==0.1.0
  match: scoring
submitter:
  hf_username: example
  contact: e@example.com
submitted_at: 2026-05-21
```

- [ ] **Step 5: Write fixture `missing-reproduction.yaml`**

Copy `valid.yaml` and delete the `reproduction:` block. (The schema will still pass — wait, the schema requires `reproduction`. So this fixture must trip the schema validator, not a separate post-check.)

Actually re-read spec §3.3: "skip any submission missing a `reproduction:` block". And per the schema we wrote in Task 1, `reproduction` is `required`. So a missing-reproduction submission is *schema-invalid*. The arena does not need a separate "missing reproduction" check at the ingest layer beyond what the schema enforces.

Therefore: this fixture exercises the same code path as schema-invalid. Simplify by **removing this fixture** and keeping only `valid.yaml` and `schema-invalid.yaml` (which we'll make missing-reproduction-shaped for concreteness).

- [ ] **Step 6: Write fixture `schema-invalid.yaml`** (missing `reproduction:`)

Create `tests/fixtures/submissions/schema-invalid.yaml`:

```yaml
schema_version: 4
system:
  name: random-baseline
  slug: random-baseline
  description: stub
  code: https://github.com/example/x
  checkpoint: https://huggingface.co/example/x
  paper:
    arxiv_id: "1911.01601"
    url: https://arxiv.org/abs/1911.01601
    bibtex: "@misc{x,title={x}}"
dataset:
  id: SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA
  revision: 9b2040e8
  split: test
scores:
  eer_percent: 49.87
  n_trials: 71237
  n_skipped: 0
artifact:
  scores_url: https://huggingface.co/example/x/resolve/abcdef1/path/scores.txt
  scores_sha256: 71ac000c0712a4551873dba87183e746cb9730cd5ab17aaa87892009bde55587
  bench_version: speech-spoof-bench==0.1.0
submitter:
  hf_username: example
  contact: e@example.com
submitted_at: 2026-05-21
```

- [ ] **Step 7: Write `tests/conftest.py`**

```python
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES
```

- [ ] **Step 8: Touch `tests/__init__.py`** (empty file)

- [ ] **Step 9: Verify pytest discovers tests dir**

Run: `cd /home/kirill/speech-spoof-bench/arena && pip install -e .[dev] && pytest --collect-only`
Expected: "no tests ran" (nothing yet) without errors.

- [ ] **Step 10: Commit**

```bash
cd /home/kirill/speech-spoof-bench/arena
git add pyproject.toml requirements.txt tests/__init__.py tests/conftest.py tests/fixtures/
git commit -m "arena: bootstrap pyproject, requirements, test fixtures"
```

---

## Task 4: `schema.py` — dataclasses + ergonomics

**Files:**
- Create: `arena/schema.py`
- Create: `arena/tests/test_schema.py`

- [ ] **Step 1: Write failing test**

```python
from datetime import datetime

from schema import ArenaState, Row, Warning


def test_row_is_frozen():
    r = Row(
        system_slug="x",
        system_name="X",
        dataset_id="org/ds",
        revision="abc",
        scores={"eer_percent": 1.0},
        reproduction_level="scoring",
        submitted_at="2026-05-21",
        submission_url="https://example",
    )
    import pytest
    with pytest.raises(Exception):
        r.system_slug = "y"  # type: ignore[misc]


def test_arena_state_holds_rows_and_warnings():
    state = ArenaState(
        manifest={"ranking_version": "v1"},
        rows=[],
        loaded_at=datetime(2026, 5, 21),
        warnings=[Warning(dataset_id="d", path="p", reason="r")],
    )
    assert state.warnings[0].reason == "r"
```

- [ ] **Step 2: Run, confirm fail**

Run: `cd /home/kirill/speech-spoof-bench/arena && pytest tests/test_schema.py -x`
Expected: `ModuleNotFoundError: No module named 'schema'`.

- [ ] **Step 3: Implement `schema.py`**

```python
"""Arena dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


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


@dataclass(frozen=True)
class Warning:
    dataset_id: str
    path: str
    reason: str


@dataclass(frozen=True)
class ArenaState:
    manifest: dict[str, Any]
    rows: list[Row]
    loaded_at: datetime
    warnings: list[Warning] = field(default_factory=list)
```

- [ ] **Step 4: Run, confirm pass**

Run: `pytest tests/test_schema.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add schema.py tests/test_schema.py
git commit -m "arena: Row / Warning / ArenaState dataclasses"
```

---

## Task 5: `ranking.py` — `assign_tiers`

**Files:**
- Create: `arena/ranking.py`
- Create: `arena/tests/test_ranking.py`

- [ ] **Step 1: Write failing tests**

```python
from schema import Row
from ranking import assign_tiers


TIERS = [
    {"name": "gold",   "min_coverage": 1.0},
    {"name": "silver", "min_coverage": 0.5},
    {"name": "bronze", "min_coverage": 0.0},
]
CORE = ["org/a", "org/b"]


def _row(slug: str, dataset: str) -> Row:
    return Row(
        system_slug=slug,
        system_name=slug,
        dataset_id=dataset,
        revision="r",
        scores={"eer_percent": 10.0},
        reproduction_level="scoring",
        submitted_at="2026-01-01",
        submission_url="u",
    )


def test_full_core_coverage_is_gold():
    rows = [_row("sys", "org/a"), _row("sys", "org/b")]
    assert assign_tiers(rows, TIERS, CORE) == {"sys": "gold"}


def test_half_core_coverage_is_silver():
    rows = [_row("sys", "org/a")]
    assert assign_tiers(rows, TIERS, CORE) == {"sys": "silver"}


def test_no_core_coverage_is_bronze():
    rows = [_row("sys", "org/extra-not-in-core")]
    assert assign_tiers(rows, TIERS, CORE) == {"sys": "bronze"}


def test_empty_rows_yields_empty_mapping():
    assert assign_tiers([], TIERS, CORE) == {}


def test_extra_dataset_does_not_inflate_coverage():
    rows = [_row("sys", "org/a"), _row("sys", "org/extra")]
    # 1/2 core covered → silver
    assert assign_tiers(rows, TIERS, CORE) == {"sys": "silver"}
```

- [ ] **Step 2: Run, confirm fail**

Run: `pytest tests/test_ranking.py -x`
Expected: ModuleNotFoundError on `ranking`.

- [ ] **Step 3: Implement `assign_tiers`**

Create `ranking.py`:

```python
"""Pure ranking + table-building functions. No I/O."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from schema import Row


def assign_tiers(
    rows: Iterable[Row],
    tiers: list[dict],
    core_set_ids: list[str],
) -> dict[str, str]:
    """Map system_slug -> highest-tier name whose min_coverage <= coverage.

    `tiers` is assumed ordered highest-first.
    Coverage = (# distinct core dataset ids the system has rows on) / |core|.
    If core_set_ids is empty, coverage is 0 for everyone.
    """
    core = set(core_set_ids)
    covered: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        if r.dataset_id in core:
            covered[r.system_slug].add(r.dataset_id)
        else:
            covered[r.system_slug]  # ensure key exists

    n_core = len(core)
    out: dict[str, str] = {}
    for slug, hits in covered.items():
        coverage = (len(hits) / n_core) if n_core else 0.0
        for tier in tiers:
            if coverage >= tier["min_coverage"]:
                out[slug] = tier["name"]
                break
    return out
```

- [ ] **Step 4: Run, confirm pass**

Run: `pytest tests/test_ranking.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add ranking.py tests/test_ranking.py
git commit -m "arena: assign_tiers"
```

---

## Task 6: `ranking.py` — table builders

**Files:**
- Modify: `arena/ranking.py`
- Modify: `arena/tests/test_ranking.py`

- [ ] **Step 1: Add failing tests for `overview_table` and `per_dataset_table`**

Append to `tests/test_ranking.py`:

```python
from ranking import overview_table, per_dataset_table


def test_overview_table_one_row_per_system_per_tier():
    rows = [
        _row("aasist", "org/a"),
        _row("aasist", "org/b"),
        _row("rawnet", "org/a"),
    ]
    out = overview_table(rows, TIERS, CORE, primary_metric="eer_percent")
    # aasist: gold; rawnet: silver
    assert {r["system"] for r in out["gold"]} == {"aasist"}
    assert {r["system"] for r in out["silver"]} == {"rawnet"}
    assert out["bronze"] == []


def test_overview_table_includes_coverage_and_per_dataset_score():
    rows = [_row("aasist", "org/a"), _row("aasist", "org/b")]
    out = overview_table(rows, TIERS, CORE, primary_metric="eer_percent")
    gold = out["gold"][0]
    assert gold["coverage"] == "2/2"
    assert gold["org/a"] == 10.0
    assert gold["org/b"] == 10.0
    assert gold["repro"] == "scoring"


def test_overview_table_missing_dataset_renders_dash():
    rows = [_row("aasist", "org/a")]
    out = overview_table(rows, TIERS, CORE, primary_metric="eer_percent")
    silver = out["silver"][0]
    assert silver["org/a"] == 10.0
    assert silver["org/b"] is None


def test_per_dataset_table_filters_to_dataset():
    rows = [_row("aasist", "org/a"), _row("rawnet", "org/b")]
    out = per_dataset_table(rows, dataset_id="org/a", metrics_in_use=["eer_percent"])
    assert len(out) == 1
    assert out[0]["system"] == "aasist"
    assert out[0]["eer_percent"] == 10.0


def test_per_dataset_table_columns_union_of_metrics():
    r1 = Row(system_slug="a", system_name="a", dataset_id="org/a", revision="r",
             scores={"eer_percent": 1.0, "min_tdcf": 0.2},
             reproduction_level="scoring", submitted_at="2026-01-01", submission_url="u")
    r2 = Row(system_slug="b", system_name="b", dataset_id="org/a", revision="r",
             scores={"eer_percent": 2.0},
             reproduction_level="inference", submitted_at="2026-01-02", submission_url="u")
    out = per_dataset_table([r1, r2], dataset_id="org/a", metrics_in_use=["eer_percent"])
    assert out[0]["min_tdcf"] == 0.2
    assert out[1]["min_tdcf"] is None
    # eer_percent listed before min_tdcf because it's in metrics_in_use first
    keys = list(out[0].keys())
    assert keys.index("eer_percent") < keys.index("min_tdcf")
```

- [ ] **Step 2: Run, confirm fail**

Expected: `ImportError: cannot import name 'overview_table'`.

- [ ] **Step 3: Implement table builders**

Append to `ranking.py`:

```python
def overview_table(
    rows: Iterable[Row],
    tiers: list[dict],
    core_set_ids: list[str],
    primary_metric: str,
) -> dict[str, list[dict]]:
    """tier_name -> list of row-dicts ready for Gradio DataFrame.

    Each dict has: system, coverage ('k/n'), repro, then one column per core
    dataset id with the system's primary-metric value or None.
    """
    rows = list(rows)
    tier_map = assign_tiers(rows, tiers, core_set_ids)

    # Group rows by system → dataset_id → primary metric value
    by_system: dict[str, dict[str, Row]] = defaultdict(dict)
    for r in rows:
        by_system[r.system_slug][r.dataset_id] = r

    out: dict[str, list[dict]] = {t["name"]: [] for t in tiers}
    n_core = len(core_set_ids)
    for slug, tier_name in tier_map.items():
        per_ds = by_system[slug]
        # Best (highest) repro level wins for the badge.
        levels = {r.reproduction_level for r in per_ds.values()}
        repro = "inference" if "inference" in levels else "scoring"
        # Pick any row for the display name.
        any_row = next(iter(per_ds.values()))
        covered = sum(1 for ds in core_set_ids if ds in per_ds)
        row = {
            "system": any_row.system_name,
            "coverage": f"{covered}/{n_core}",
            "repro": repro,
        }
        for ds in core_set_ids:
            row[ds] = per_ds[ds].scores.get(primary_metric) if ds in per_ds else None
        out[tier_name].append(row)

    # Sort each tier by primary-metric mean rank over Core (lower is better).
    for tier_name, table in out.items():
        table.sort(key=lambda r: _mean_rank(r, core_set_ids))
    return out


def _mean_rank(row: dict, core_set_ids: list[str]) -> float:
    vals = [row[ds] for ds in core_set_ids if row[ds] is not None]
    if not vals:
        return float("inf")
    return sum(vals) / len(vals)


def per_dataset_table(
    rows: Iterable[Row],
    dataset_id: str,
    metrics_in_use: list[str],
) -> list[dict]:
    """Rows for the Per-dataset tab.

    Columns: system, <each metric id present>, reproduction, submitted_at,
    submission. Metric column order is `metrics_in_use` first, then any
    extras in alphabetical order.
    """
    matching = [r for r in rows if r.dataset_id == dataset_id]
    all_metrics = set()
    for r in matching:
        all_metrics.update(r.scores.keys())
    ordered = [m for m in metrics_in_use if m in all_metrics]
    ordered += sorted(all_metrics - set(ordered))

    out = []
    for r in matching:
        row = {"system": r.system_name}
        for m in ordered:
            row[m] = r.scores.get(m)
        row["reproduction"] = r.reproduction_level
        row["submitted_at"] = r.submitted_at
        row["submission"] = r.submission_url
        out.append(row)
    return out
```

- [ ] **Step 4: Run, confirm pass**

Run: `pytest tests/test_ranking.py -v`
Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add ranking.py tests/test_ranking.py
git commit -m "arena: overview_table and per_dataset_table"
```

---

## Task 7: `ingest.py` — load_state with TTL, lock, warnings

**Files:**
- Create: `arena/ingest.py`
- Create: `arena/tests/test_ingest.py`

- [ ] **Step 1: Write failing tests**

```python
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import ingest
from schema import ArenaState, Row


@pytest.fixture(autouse=True)
def reset_cache():
    ingest._state = None
    ingest._loaded_at = None
    yield
    ingest._state = None
    ingest._loaded_at = None


def _patch_hf(monkeypatch, fixtures_dir: Path, dataset_files: dict[str, list[str]]):
    """Patch HF calls to read from fixtures.

    dataset_files: dataset_id -> list of submission paths relative to fixtures/submissions/.
    """
    def fake_fetch_manifest():
        import yaml
        return yaml.safe_load((fixtures_dir / "manifest.yaml").read_text())

    def fake_list(dataset_id, *, api=None):
        return [f"submissions/{name}" for name in dataset_files.get(dataset_id, [])]

    def fake_fetch_submission(dataset_id, path):
        from speech_spoof_bench.submission import parse_submission
        local = fixtures_dir / path  # path = "submissions/xxx.yaml"
        return parse_submission(local.read_text())

    monkeypatch.setattr(ingest, "_fetch_manifest", fake_fetch_manifest)
    monkeypatch.setattr(ingest, "_list_submission_files", fake_list)
    monkeypatch.setattr(ingest, "_fetch_submission_dict", fake_fetch_submission)


def test_load_state_with_valid_submission(monkeypatch, fixtures_dir):
    _patch_hf(monkeypatch, fixtures_dir, {
        "SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA": ["valid.yaml"],
        "SpeechAntiSpoofingBenchmarks/ASVspoof2021_LA": [],
        "SpeechAntiSpoofingBenchmarks/InTheWild": [],
    })
    state = ingest.load_state(force_refresh=True)
    assert len(state.rows) == 1
    assert state.rows[0].system_slug == "random-baseline"
    assert state.rows[0].scores["eer_percent"] == 49.87
    assert state.warnings == []


def test_load_state_skips_schema_invalid_and_records_warning(monkeypatch, fixtures_dir):
    _patch_hf(monkeypatch, fixtures_dir, {
        "SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA": ["valid.yaml", "schema-invalid.yaml"],
        "SpeechAntiSpoofingBenchmarks/ASVspoof2021_LA": [],
        "SpeechAntiSpoofingBenchmarks/InTheWild": [],
    })
    state = ingest.load_state(force_refresh=True)
    assert len(state.rows) == 1
    assert len(state.warnings) == 1
    w = state.warnings[0]
    assert "schema-invalid.yaml" in w.path
    assert "reproduction" in w.reason.lower() or "required" in w.reason.lower()


def test_load_state_cache_within_ttl(monkeypatch, fixtures_dir):
    _patch_hf(monkeypatch, fixtures_dir, {
        "SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA": ["valid.yaml"],
        "SpeechAntiSpoofingBenchmarks/ASVspoof2021_LA": [],
        "SpeechAntiSpoofingBenchmarks/InTheWild": [],
    })
    call_count = {"n": 0}
    real_fetch = ingest._fetch_manifest
    def counting_fetch():
        call_count["n"] += 1
        return real_fetch()
    monkeypatch.setattr(ingest, "_fetch_manifest", counting_fetch)

    s1 = ingest.load_state(force_refresh=True)
    s2 = ingest.load_state(force_refresh=False)
    assert s1 is s2
    assert call_count["n"] == 1


def test_load_state_force_refresh_bypasses_cache(monkeypatch, fixtures_dir):
    _patch_hf(monkeypatch, fixtures_dir, {
        "SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA": ["valid.yaml"],
        "SpeechAntiSpoofingBenchmarks/ASVspoof2021_LA": [],
        "SpeechAntiSpoofingBenchmarks/InTheWild": [],
    })
    call_count = {"n": 0}
    real_fetch = ingest._fetch_manifest
    def counting_fetch():
        call_count["n"] += 1
        return real_fetch()
    monkeypatch.setattr(ingest, "_fetch_manifest", counting_fetch)

    ingest.load_state(force_refresh=True)
    ingest.load_state(force_refresh=True)
    assert call_count["n"] == 2


def test_load_state_manifest_failure_returns_empty_state(monkeypatch):
    def boom():
        raise RuntimeError("hf down")
    monkeypatch.setattr(ingest, "_fetch_manifest", boom)
    state = ingest.load_state(force_refresh=True)
    assert state.rows == []
    assert len(state.warnings) == 1
    assert "hf down" in state.warnings[0].reason
```

- [ ] **Step 2: Run, confirm fail**

Run: `pytest tests/test_ingest.py -x`
Expected: ModuleNotFoundError on `ingest`.

- [ ] **Step 3: Implement `ingest.py`**

```python
"""Fetch manifest + submissions from HF Hub; cache with 30-min TTL."""

from __future__ import annotations

import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi
from speech_spoof_bench.manifest import fetch_manifest
from speech_spoof_bench.submission import (
    SubmissionValidationError,
    fetch_submission,
    list_submission_files,
)

from schema import ArenaState, Row, Warning

_TTL_SECONDS = 30 * 60
_FAILURE_TTL_SECONDS = 60

_state: ArenaState | None = None
_loaded_at: float | None = None
_lock = threading.Lock()


def load_state(force_refresh: bool = False) -> ArenaState:
    global _state, _loaded_at

    if not force_refresh and _state is not None and _loaded_at is not None:
        ttl = _FAILURE_TTL_SECONDS if not _state.rows and _state.warnings else _TTL_SECONDS
        if (time.monotonic() - _loaded_at) < ttl:
            return _state

    with _lock:
        if not force_refresh and _state is not None and _loaded_at is not None:
            ttl = _FAILURE_TTL_SECONDS if not _state.rows and _state.warnings else _TTL_SECONDS
            if (time.monotonic() - _loaded_at) < ttl:
                return _state
        new_state = _build_state()
        _state = new_state
        _loaded_at = time.monotonic()
        return new_state


def _build_state() -> ArenaState:
    warnings: list[Warning] = []
    try:
        manifest = _fetch_manifest()
    except Exception as exc:
        return ArenaState(
            manifest={},
            rows=[],
            loaded_at=datetime.utcnow(),
            warnings=[Warning(dataset_id="<manifest>", path="manifest.yaml", reason=str(exc))],
        )

    dataset_ids = [e["id"] for e in manifest["core_set"] + manifest["extended"]]
    api = HfApi()
    rows: list[Row] = []

    for dataset_id in dataset_ids:
        try:
            paths = _list_submission_files(dataset_id, api=api)
        except Exception as exc:
            warnings.append(Warning(dataset_id=dataset_id, path="<list>", reason=str(exc)))
            continue
        for path in paths:
            try:
                sub = _fetch_submission_dict(dataset_id, path)
            except SubmissionValidationError as exc:
                warnings.append(Warning(dataset_id=dataset_id, path=path, reason=str(exc)))
                continue
            except Exception as exc:
                warnings.append(Warning(dataset_id=dataset_id, path=path, reason=f"{type(exc).__name__}: {exc}"))
                continue
            rows.append(_to_row(sub, dataset_id, path))

    return ArenaState(
        manifest=manifest,
        rows=rows,
        loaded_at=datetime.utcnow(),
        warnings=warnings,
    )


# Indirection points so tests can monkeypatch.
def _fetch_manifest() -> dict[str, Any]:
    return fetch_manifest()


def _list_submission_files(dataset_id: str, *, api: HfApi) -> list[str]:
    return list_submission_files(dataset_id, api=api)


def _fetch_submission_dict(dataset_id: str, path: str) -> dict[str, Any]:
    return fetch_submission(dataset_id, path)


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
    )
```

- [ ] **Step 4: Run, confirm pass**

Run: `pytest tests/test_ingest.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add ingest.py tests/test_ingest.py
git commit -m "arena: load_state with TTL cache, warnings, and HF Hub fetch"
```

---

## Task 8: `app.py` — Gradio Blocks with three tabs

**Files:**
- Create: `arena/app.py`

- [ ] **Step 1: Write `app.py`**

```python
"""SpeechAntiSpoofingArena — Gradio entrypoint."""

from __future__ import annotations

import gradio as gr

import ingest
from ranking import overview_table, per_dataset_table
from schema import ArenaState


def _format_timestamp(state: ArenaState) -> str:
    return f"**Last refreshed:** {state.loaded_at.isoformat(timespec='seconds')}Z"


def _render() -> tuple[ArenaState, str]:
    state = ingest.load_state()
    return state, _format_timestamp(state)


def _refresh() -> tuple[ArenaState, str]:
    state = ingest.load_state(force_refresh=True)
    return state, _format_timestamp(state)


def _overview_tables(state: ArenaState) -> list:
    tiers = state.manifest.get("tiers", [])
    core = [e["id"] for e in state.manifest.get("core_set", [])]
    primary = state.manifest.get("metrics_in_use", ["eer_percent"])[0]
    tables = overview_table(state.rows, tiers, core, primary_metric=primary)
    return [tables.get(t["name"], []) for t in tiers]


def _per_dataset_choices(state: ArenaState) -> list[str]:
    return [e["id"] for e in state.manifest.get("core_set", []) + state.manifest.get("extended", [])]


def _per_dataset_data(state: ArenaState, dataset_id: str | None) -> list[dict]:
    if not dataset_id:
        return []
    metrics = state.manifest.get("metrics_in_use", ["eer_percent"])
    return per_dataset_table(state.rows, dataset_id=dataset_id, metrics_in_use=metrics)


def _about_text(state: ArenaState) -> str:
    m = state.manifest
    lines = [
        f"**Schema version:** {m.get('schema_version', '?')}",
        f"**Ranking version:** {m.get('ranking_version', '?')}",
        f"**Metrics in use:** {', '.join(m.get('metrics_in_use', []))}",
        f"**Last refreshed:** {state.loaded_at.isoformat(timespec='seconds')}Z",
        "",
        f"**Warnings ({len(state.warnings)}):**",
    ]
    if not state.warnings:
        lines.append("_No warnings._")
    else:
        lines.append("| Dataset | Path | Reason |")
        lines.append("|---|---|---|")
        for w in state.warnings:
            lines.append(f"| {w.dataset_id} | {w.path} | {w.reason} |")
    return "\n".join(lines)


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Speech Anti-Spoofing Arena") as demo:
        gr.Markdown("# 🎙️ Speech Anti-Spoofing Arena")
        state_box = gr.State()
        ts_md = gr.Markdown()
        refresh_btn = gr.Button("🔄 Refresh")

        with gr.Tabs():
            with gr.Tab("Overview"):
                overview_headers = gr.Markdown()
                # We render tier tables dynamically based on manifest tiers.
                # MVP: assume 3 tiers (gold/silver/bronze) and render 3 DataFrames.
                gold_md = gr.Markdown("## 🥇 Gold (coverage = 100%)")
                gold_df = gr.DataFrame(interactive=False, wrap=True)
                silver_md = gr.Markdown("## 🥈 Silver (coverage ≥ 50%)")
                silver_df = gr.DataFrame(interactive=False, wrap=True)
                bronze_md = gr.Markdown("## 🥉 Bronze (coverage ≥ 0%)")
                bronze_df = gr.DataFrame(interactive=False, wrap=True)

            with gr.Tab("Per dataset"):
                ds_dropdown = gr.Dropdown(label="Dataset", choices=[], value=None)
                per_ds_df = gr.DataFrame(interactive=False, wrap=True)

            with gr.Tab("About"):
                about_md = gr.Markdown()

        def _populate(state: ArenaState):
            gold, silver, bronze = _overview_tables(state)
            choices = _per_dataset_choices(state)
            first = choices[0] if choices else None
            per_ds = _per_dataset_data(state, first)
            return (
                gold, silver, bronze,
                gr.Dropdown(choices=choices, value=first),
                per_ds,
                _about_text(state),
            )

        def _initial():
            state, ts = _render()
            return (state, ts, *_populate(state))

        def _do_refresh():
            state, ts = _refresh()
            return (state, ts, *_populate(state))

        def _on_dataset_change(state: ArenaState, dataset_id: str | None):
            return _per_dataset_data(state, dataset_id)

        outputs = [state_box, ts_md, gold_df, silver_df, bronze_df, ds_dropdown, per_ds_df, about_md]
        demo.load(_initial, outputs=outputs)
        refresh_btn.click(_do_refresh, outputs=outputs)
        ds_dropdown.change(_on_dataset_change, inputs=[state_box, ds_dropdown], outputs=per_ds_df)

    return demo


demo = build_demo()

if __name__ == "__main__":
    demo.launch()
```

- [ ] **Step 2: Local smoke test (no HF traffic — uses real fetch)**

Run: `cd /home/kirill/speech-spoof-bench/arena && pip install -r requirements.txt && python -c "import app; print(app.demo)"`
Expected: prints a Gradio Blocks object, no traceback.

- [ ] **Step 3: Local end-to-end (hits HF Hub)**

Run: `cd /home/kirill/speech-spoof-bench/arena && python app.py`
Expected: Gradio serves on `http://127.0.0.1:7860`. Open it. Confirm:
- Overview → Gold tier shows the `random-baseline` row with `coverage: 1/1` and `org/...ASVspoof2019_LA: 49.87`.
- Per dataset → dropdown defaults to ASVspoof2019_LA; table has one row.
- About → schema_version 1, ranking_version v1, metrics `eer_percent`, no warnings.

Ctrl+C to stop. If anything renders wrong, stop and fix before continuing.

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "arena: Gradio app — Overview / Per dataset / About + Refresh"
```

---

## Task 9: Interactive UI review with the user (LOCAL)

**Goal:** Before pushing to HF, walk the user through the local Gradio UI and surface any visual / interaction issues. This task is a guided dialog, not code. The agent runs the app, instructs the user, waits for replies, and either fixes findings (looping back to earlier tasks) or proceeds.

**Files:** none.

- [ ] **Step 1: Launch the local app for the user**

Run in a background-friendly way (so the agent stays interactive):

```bash
cd /home/kirill/speech-spoof-bench/arena && python app.py
```

Tell the user: "The Gradio app is running at http://127.0.0.1:7860. Open it in a browser. I'll guide you through a checklist — reply after each item with `ok`, or describe what's wrong."

- [ ] **Step 2: Guide the user through Overview tab**

Ask the user (one at a time, wait for reply between each):

1. "On the **Overview** tab — do you see the page title 'Speech Anti-Spoofing Arena' at the top and a 'Last refreshed: …' timestamp?"
2. "Below it, three tier sections: **🥇 Gold**, **🥈 Silver**, **🥉 Bronze**. Are all three headings visible?"
3. "The Gold table should contain exactly one row — `random-baseline` — with columns: system, coverage (`1/1`), repro (`scoring`), and one column for `SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA` showing `~49.87`. Does it match?"
4. "Silver and Bronze tables should be empty (or render as 'No data'). Is that what you see?"
5. "Is anything visually broken — overlapping text, columns too narrow to read, weird wrapping, missing emoji?"

- [ ] **Step 3: Guide the user through Per dataset tab**

1. "Click the **Per dataset** tab. There's a Dataset dropdown — does it default to `SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA`?"
2. "Below it, a table with one row for `random-baseline`. Columns should be: system, `eer_percent` (≈49.87), reproduction (`scoring`), submitted_at, submission (a URL). Confirm all columns are present and the submission cell looks clickable / shows a URL?"
3. "Open the dropdown — are all three dataset ids listed (`ASVspoof2019_LA`, `ASVspoof2021_LA`, `InTheWild`)? Selecting one with no submissions should show an empty table, no error. Confirm?"

- [ ] **Step 4: Guide the user through About tab**

1. "Click the **About** tab. You should see: Schema version `1`, Ranking version `v1`, Metrics in use `eer_percent`, Last refreshed timestamp."
2. "Below that: 'Warnings (0): No warnings.' Confirm?"

- [ ] **Step 5: Guide the user through the Refresh button**

1. "Note the current timestamp. Click the 🔄 Refresh button at the top."
2. "After a few seconds, does the timestamp update? Do tab contents stay correct (no flicker into empty state, no error)?"

- [ ] **Step 6: Decide outcome**

Ask: "Anything off — visually, content-wise, or interaction-wise? Reply `all good` to proceed to deploying the Space, or list the issues."

Based on the reply:

- If `all good` → kill the local app (`Ctrl+C`) and proceed to Task 10.
- If issues are **content/logic** bugs (wrong numbers, missing rows, wrong tier) → loop back to the relevant task (5–7 for ranking/ingest, 8 for layout) and fix with a new TDD cycle. After fixing, restart this Task 9 from Step 1.
- If issues are **cosmetic** (widths, headings, ordering) → patch `app.py` only, commit as `arena: ui polish from review`, then re-run this Task 9 from Step 1.

- [ ] **Step 7: Commit any fixes from this review**

If fixes were applied:

```bash
cd /home/kirill/speech-spoof-bench/arena
git add -A
git commit -m "arena: address local UI review findings"
```

If no fixes were needed, skip this step.

---

## Task 10: Push to HF Space and verify cold start

**Files:** none (deploy step).

- [ ] **Step 1: Configure HF remote**

Run: `cd /home/kirill/speech-spoof-bench/arena && git remote -v`

If no `hf` remote: `git remote add hf https://huggingface.co/spaces/SpeechAntiSpoofingBenchmarks/SpeechAntiSpoofingArena`.

- [ ] **Step 2: Pin `speech-spoof-bench` to a sha in requirements.txt**

Replace `@main` with `@<current-sha>` from the pip package repo (so HF Space rebuilds are reproducible). Get the sha:

Run: `cd /home/kirill/speech-spoof-bench/speech-spoof-bench && git rev-parse HEAD`

Edit `arena/requirements.txt`, replace `@main` with the sha. Commit:

```bash
cd /home/kirill/speech-spoof-bench/arena
git add requirements.txt
git commit -m "arena: pin speech-spoof-bench to current sha"
```

- [ ] **Step 3: Push**

Run: `cd /home/kirill/speech-spoof-bench/arena && git push hf main`

Expected: HF responds with a build URL.

- [ ] **Step 4: Wait for build, open the Space**

Open `https://huggingface.co/spaces/SpeechAntiSpoofingBenchmarks/SpeechAntiSpoofingArena` in a browser. Wait for "Running".

- [ ] **Step 5: Visual check**

Confirm Phase 5 DoD bullets from spec §8:
- Cold start under 15s (eyeball the build log).
- Overview shows the random baseline in Gold.
- Per dataset → ASVspoof2019_LA → one row.
- About → no warnings.
- Refresh button works (updates the timestamp).

- [ ] **Step 6: Negative test — inject a broken submission**

This is a one-off, manual:

```bash
cd /home/kirill/mnt/drive3_8tb/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA
cat > submissions/broken.yaml <<'EOF'
schema_version: 4
system:
  slug: broken
EOF
git add submissions/broken.yaml && git commit -m "TEMP: arena warning smoke test"
git push
```

Click Refresh on the Space. Confirm About tab now lists a warning for `submissions/broken.yaml` and Overview is unchanged.

Then **remove the test file**:

```bash
git revert HEAD --no-edit
git push
```

Click Refresh on the Space — warning gone.

- [ ] **Step 7: Final commit log**

Run: `cd /home/kirill/speech-spoof-bench/arena && git log --oneline`
Confirm 6–7 clean commits matching the tasks above.

---

## Done-when

All checkboxes in Phase 5 DoD (spec §8) pass on the live Space.

## Out of scope (deferred, per spec §9)

System-detail page, paper rendering, Submit tab, Docker, webhook, `cache.json`, CI/CD, `nightly-revalidate`.
