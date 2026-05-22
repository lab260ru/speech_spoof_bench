# Phase 4 — Manifest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish the single-file `arena-manifest/manifest.yaml` and replace the Phase 2 manifest stubs in the pip package with a working fetch + validate + CLI (`manifest`, `list`).

**Architecture:** The manifest is a YAML file in a public HF dataset repo. The pip package downloads it via `huggingface_hub.hf_hub_download`, validates it against a bundled JSON Schema, and exposes two CLI commands plus a tiny accessor API.

**Tech Stack:** Python 3.10+, `huggingface_hub`, `pyyaml`, `jsonschema`, `pytest`. All already in `pyproject.toml`.

**Spec:** `docs/specs/2026-05-21-phase-4-manifest-design.md`

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `arena-manifest/manifest.yaml` | create (in `/home/kirill/speech-spoof-bench/arena-manifest/`) | The launch manifest content. |
| `arena-manifest/README.md` | modify | Replace "Phase 0 — not yet written" with file shape + PLAN §4 link. |
| `speech-spoof-bench/src/speech_spoof_bench/schema/__init__.py` | create | Empty marker so `schema/` is a package and ships via `find`. |
| `speech-spoof-bench/src/speech_spoof_bench/schema/manifest.schema.json` | create | JSON Schema for the manifest. |
| `speech-spoof-bench/src/speech_spoof_bench/manifest.py` | replace | `fetch_manifest`, `load_manifest`, accessor helpers, internal `_validate`. |
| `speech-spoof-bench/src/speech_spoof_bench/cli.py` | modify | Wire `manifest` subcommand; replace `list` stub. |
| `speech-spoof-bench/pyproject.toml` | modify | Add `package-data` so `*.json` ships with the wheel. |
| `speech-spoof-bench/tests/test_manifest.py` | create | Unit tests for schema, loader, accessors. |
| `speech-spoof-bench/tests/test_cli.py` | modify | Replace the Phase 2 `test_cli_list_raises_at_phase_2` with real `manifest` + `list` CLI tests. |

Working directories (note: there are two layers of `speech-spoof-bench/`):
- Project root: `/home/kirill/speech-spoof-bench/`
- Pip package repo: `/home/kirill/speech-spoof-bench/speech-spoof-bench/` (its own git repo)
- Arena manifest repo: `/home/kirill/speech-spoof-bench/arena-manifest/` (its own git repo)

Each `git commit` step states which repo to run it in.

---

## Task 1: Author and publish the manifest YAML

**Files:**
- Create: `/home/kirill/speech-spoof-bench/arena-manifest/manifest.yaml`
- Modify: `/home/kirill/speech-spoof-bench/arena-manifest/README.md`

- [ ] **Step 1: Confirm the LA dataset commit sha to pin**

Run:
```bash
cd /home/kirill/mnt/drive3_8tb/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA && git rev-parse HEAD
```
Expected: prints a 40-char hex sha. Use this value in Step 2 in place of `<LA_SHA>`. As of spec time the sha is `9b2040e8c57749dcd9a4f16ad61b4f47626b89ec` — re-check, do not assume.

- [ ] **Step 2: Write `manifest.yaml`**

Create `/home/kirill/speech-spoof-bench/arena-manifest/manifest.yaml` with this content (substituting the sha from Step 1):

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
    revision: <LA_SHA>

extended: []
```

- [ ] **Step 3: Update the manifest repo README**

Replace `/home/kirill/speech-spoof-bench/arena-manifest/README.md` body. Final content:

```markdown
---
license: mit
tags:
  - arena-manifest
---

# arena-manifest

Single-file manifest read by the SpeechAntiSpoofingBenchmarks Arena and the
`speech-spoof-bench` pip package. Defines tiers, the core/extended dataset
sets, and pinned dataset commit shas.

## Schema (informal)

- `ranking_version` (string) — bumped when ranking rules change.
- `schema_version` (int) — bumped when this file's shape changes. Currently `1`.
- `metrics_in_use` (list of metric ids) — informational; controls arena column order.
- `tiers` (ordered list, highest first) — `{name, min_coverage}`.
- `core_set` / `extended` — `{id, revision}` where `id` is `org/name` and
  `revision` is a git commit sha of the dataset repo (7–40 hex chars).

The authoritative JSON Schema ships with the `speech-spoof-bench` pip package
at `speech_spoof_bench/schema/manifest.schema.json`.

See [the project plan](https://github.com/lab260ru/speech_spoof_bench/blob/main/docs/roadmap/PLAN.md) §4.
```

- [ ] **Step 4: Commit and push the manifest repo**

Run in the `arena-manifest/` repo:
```bash
cd /home/kirill/speech-spoof-bench/arena-manifest
git add manifest.yaml README.md
git status --short
git commit -m "feat: add launch manifest (LA pinned, v1 ranking, gold/silver/bronze tiers)"
git push origin main
```
Expected: push succeeds; `huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/arena-manifest/blob/main/manifest.yaml` shows the file.

- [ ] **Step 5: Sanity-check by re-downloading**

Run:
```bash
python -c "from huggingface_hub import hf_hub_download; p = hf_hub_download(repo_id='SpeechAntiSpoofingBenchmarks/arena-manifest', repo_type='dataset', filename='manifest.yaml'); print(open(p).read())"
```
Expected: prints the manifest YAML body verbatim.

---

## Task 2: Add the JSON Schema

**Files:**
- Create: `speech-spoof-bench/src/speech_spoof_bench/schema/__init__.py`
- Create: `speech-spoof-bench/src/speech_spoof_bench/schema/manifest.schema.json`

All paths below are relative to `/home/kirill/speech-spoof-bench/speech-spoof-bench/`.

- [ ] **Step 1: Create the schema package marker**

Create `src/speech_spoof_bench/schema/__init__.py` as an empty file. (Needed so `setuptools.find` treats `schema/` as a package; the JSON file rides along via `package-data` in Task 6.)

- [ ] **Step 2: Write the JSON Schema**

Create `src/speech_spoof_bench/schema/manifest.schema.json`:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "arena-manifest",
  "type": "object",
  "additionalProperties": false,
  "required": ["ranking_version", "schema_version", "metrics_in_use", "tiers", "core_set", "extended"],
  "properties": {
    "ranking_version": {"type": "string", "minLength": 1},
    "schema_version": {"type": "integer", "const": 1},
    "metrics_in_use": {
      "type": "array",
      "minItems": 1,
      "items": {"type": "string", "minLength": 1}
    },
    "tiers": {
      "type": "array",
      "minItems": 1,
      "items": {
        "type": "object",
        "additionalProperties": false,
        "required": ["name", "min_coverage"],
        "properties": {
          "name": {"type": "string", "minLength": 1},
          "min_coverage": {"type": "number", "minimum": 0, "maximum": 1}
        }
      }
    },
    "core_set": {
      "type": "array",
      "minItems": 1,
      "items": {"$ref": "#/definitions/dataset_entry"}
    },
    "extended": {
      "type": "array",
      "items": {"$ref": "#/definitions/dataset_entry"}
    }
  },
  "definitions": {
    "dataset_entry": {
      "type": "object",
      "additionalProperties": false,
      "required": ["id", "revision"],
      "properties": {
        "id": {"type": "string", "pattern": "^[^/\\s]+/[^/\\s]+$"},
        "revision": {"type": "string", "pattern": "^[a-f0-9]{7,40}$"}
      }
    }
  }
}
```

- [ ] **Step 3: Verify the schema parses**

Run:
```bash
cd /home/kirill/speech-spoof-bench/speech-spoof-bench
python -c "import json; json.load(open('src/speech_spoof_bench/schema/manifest.schema.json'))"
```
Expected: exits 0, no output.

---

## Task 3: Implement `manifest.py` — TDD

**Files:**
- Create: `tests/test_manifest.py`
- Replace: `src/speech_spoof_bench/manifest.py`

All paths relative to `/home/kirill/speech-spoof-bench/speech-spoof-bench/`.

- [ ] **Step 1: Write failing tests**

Create `tests/test_manifest.py`:

```python
"""Tests for speech_spoof_bench.manifest."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from jsonschema import ValidationError

from speech_spoof_bench import manifest as mf


VALID = {
    "ranking_version": "v1",
    "schema_version": 1,
    "metrics_in_use": ["eer_percent"],
    "tiers": [
        {"name": "gold", "min_coverage": 1.0},
        {"name": "silver", "min_coverage": 0.5},
        {"name": "bronze", "min_coverage": 0.0},
    ],
    "core_set": [
        {"id": "Org/Dataset_A", "revision": "9b2040e8c57749dcd9a4f16ad61b4f47626b89ec"}
    ],
    "extended": [
        {"id": "Org/Dataset_B", "revision": "deadbee"}
    ],
}


def _write(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "manifest.yaml"
    p.write_text(yaml.safe_dump(data, sort_keys=False))
    return p


def test_load_manifest_accepts_valid(tmp_path):
    out = mf.load_manifest(_write(tmp_path, VALID))
    assert out["ranking_version"] == "v1"
    assert out["schema_version"] == 1


@pytest.mark.parametrize("mutator,reason", [
    (lambda d: d.pop("tiers"), "missing tiers"),
    (lambda d: d.pop("core_set"), "missing core_set"),
    (lambda d: d.update({"extra_top_level_key": 1}), "extra key"),
    (lambda d: d["core_set"].clear(), "empty core_set"),
    (lambda d: d["core_set"][0].update({"revision": "not-hex"}), "bad revision"),
    (lambda d: d["core_set"][0].update({"id": "no-slash"}), "bad id"),
    (lambda d: d["tiers"][0].update({"min_coverage": 1.5}), "min_coverage > 1"),
    (lambda d: d.update({"schema_version": 2}), "wrong schema_version"),
    (lambda d: d["metrics_in_use"].clear(), "empty metrics"),
])
def test_load_manifest_rejects_invalid(tmp_path, mutator, reason):
    import copy
    bad = copy.deepcopy(VALID)
    mutator(bad)
    with pytest.raises(ValidationError):
        mf.load_manifest(_write(tmp_path, bad))


def test_core_dataset_ids(tmp_path):
    m = mf.load_manifest(_write(tmp_path, VALID))
    assert mf.core_dataset_ids(m) == ["Org/Dataset_A"]


def test_all_dataset_ids_core_then_extended(tmp_path):
    m = mf.load_manifest(_write(tmp_path, VALID))
    assert mf.all_dataset_ids(m) == ["Org/Dataset_A", "Org/Dataset_B"]


def test_revision_for_known(tmp_path):
    m = mf.load_manifest(_write(tmp_path, VALID))
    assert mf.revision_for(m, "Org/Dataset_A").startswith("9b2040e8")
    assert mf.revision_for(m, "Org/Dataset_B") == "deadbee"


def test_revision_for_unknown_returns_none(tmp_path):
    m = mf.load_manifest(_write(tmp_path, VALID))
    assert mf.revision_for(m, "Org/Nope") is None


def test_fetch_manifest_uses_hf_hub(monkeypatch, tmp_path):
    """fetch_manifest delegates to hf_hub_download with the expected repo coords."""
    fake = _write(tmp_path, VALID)
    calls = {}

    def fake_download(*, repo_id, repo_type, filename):
        calls["repo_id"] = repo_id
        calls["repo_type"] = repo_type
        calls["filename"] = filename
        return str(fake)

    monkeypatch.setattr(mf, "hf_hub_download", fake_download)
    out = mf.fetch_manifest()
    assert out["ranking_version"] == "v1"
    assert calls == {
        "repo_id": "SpeechAntiSpoofingBenchmarks/arena-manifest",
        "repo_type": "dataset",
        "filename": "manifest.yaml",
    }
```

- [ ] **Step 2: Run the test file to confirm it fails**

Run:
```bash
cd /home/kirill/speech-spoof-bench/speech-spoof-bench
pytest tests/test_manifest.py -x
```
Expected: collection or import errors (because `load_manifest`, accessors, and `hf_hub_download` symbol don't exist on `manifest` yet) — all FAIL.

- [ ] **Step 3: Implement `manifest.py`**

Replace `src/speech_spoof_bench/manifest.py` entirely with:

```python
"""arena-manifest reader.

Fetches the single-file manifest from the public HF dataset repo
`SpeechAntiSpoofingBenchmarks/arena-manifest`, validates it against the
bundled JSON Schema, and exposes a few small accessors.
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any

import yaml
from huggingface_hub import hf_hub_download
from jsonschema import validate

MANIFEST_REPO = "SpeechAntiSpoofingBenchmarks/arena-manifest"
MANIFEST_FILENAME = "manifest.yaml"
SCHEMA_PACKAGE = "speech_spoof_bench.schema"
SCHEMA_FILENAME = "manifest.schema.json"


def _load_schema() -> dict[str, Any]:
    with resources.files(SCHEMA_PACKAGE).joinpath(SCHEMA_FILENAME).open("r") as f:
        return json.load(f)


def _parse_and_validate(text: str) -> dict[str, Any]:
    data = yaml.safe_load(text)
    validate(instance=data, schema=_load_schema())
    return data


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Load + validate a local manifest file. Used in tests and offline dev."""
    return _parse_and_validate(Path(path).read_text())


def fetch_manifest() -> dict[str, Any]:
    """Download manifest.yaml from HF, parse, validate, return dict.

    No auth required (public dataset repo).
    """
    local = hf_hub_download(
        repo_id=MANIFEST_REPO,
        repo_type="dataset",
        filename=MANIFEST_FILENAME,
    )
    return _parse_and_validate(Path(local).read_text())


def core_dataset_ids(manifest: dict[str, Any]) -> list[str]:
    return [entry["id"] for entry in manifest["core_set"]]


def all_dataset_ids(manifest: dict[str, Any]) -> list[str]:
    return [entry["id"] for entry in manifest["core_set"] + manifest["extended"]]


def revision_for(manifest: dict[str, Any], dataset_id: str) -> str | None:
    for entry in manifest["core_set"] + manifest["extended"]:
        if entry["id"] == dataset_id:
            return entry["revision"]
    return None
```

- [ ] **Step 4: Run the tests until they pass**

Run:
```bash
cd /home/kirill/speech-spoof-bench/speech-spoof-bench
pytest tests/test_manifest.py -v
```
Expected: all tests PASS.

- [ ] **Step 5: Commit**

Run in the pip-package repo:
```bash
cd /home/kirill/speech-spoof-bench/speech-spoof-bench
git add src/speech_spoof_bench/schema/__init__.py \
        src/speech_spoof_bench/schema/manifest.schema.json \
        src/speech_spoof_bench/manifest.py \
        tests/test_manifest.py
git commit -m "feat(manifest): fetch + validate arena-manifest, replace phase-2 stub"
```

---

## Task 4: Wire the `manifest` and `list` CLI subcommands — TDD

**Files:**
- Modify: `tests/test_cli.py`
- Modify: `src/speech_spoof_bench/cli.py`

All paths relative to `/home/kirill/speech-spoof-bench/speech-spoof-bench/`.

- [ ] **Step 1: Replace the Phase 2 `list` test with real CLI tests**

In `tests/test_cli.py`:

1. Delete the existing `test_cli_list_raises_at_phase_2` function (lines 47–51 of the current file).
2. Append the following at the bottom of the file:

```python
from speech_spoof_bench import manifest as _mf


_FAKE_MANIFEST = {
    "ranking_version": "v1",
    "schema_version": 1,
    "metrics_in_use": ["eer_percent"],
    "tiers": [
        {"name": "gold", "min_coverage": 1.0},
        {"name": "silver", "min_coverage": 0.5},
        {"name": "bronze", "min_coverage": 0.0},
    ],
    "core_set": [
        {"id": "Org/A", "revision": "9b2040e8c57749dcd9a4f16ad61b4f47626b89ec"}
    ],
    "extended": [
        {"id": "Org/B", "revision": "deadbeef"}
    ],
}


def test_cli_manifest_prints_yaml(monkeypatch, capsys, tmp_path):
    """`manifest` prints the raw file contents verbatim."""
    raw = (
        "ranking_version: v1\n"
        "schema_version: 1\n"
        "metrics_in_use:\n  - eer_percent\n"
        "tiers:\n  - {name: gold, min_coverage: 1.0}\n"
        "  - {name: silver, min_coverage: 0.5}\n"
        "  - {name: bronze, min_coverage: 0.0}\n"
        "core_set:\n  - id: Org/A\n    revision: 9b2040e8c57749dcd9a4f16ad61b4f47626b89ec\n"
        "extended: []\n"
    )
    fake = tmp_path / "manifest.yaml"
    fake.write_text(raw)

    def fake_download(**kwargs):
        return str(fake)

    monkeypatch.setattr(_mf, "hf_hub_download", fake_download)
    rc = main(["manifest"])
    assert rc == 0
    out = capsys.readouterr().out
    assert out.rstrip("\n") == raw.rstrip("\n")


def test_cli_list_prints_core_then_extended(monkeypatch, capsys):
    def fake_fetch():
        return _FAKE_MANIFEST

    monkeypatch.setattr(_mf, "fetch_manifest", fake_fetch)
    rc = main(["list"])
    assert rc == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines == ["[core] Org/A", "[ext]  Org/B"]
```

- [ ] **Step 2: Run the CLI tests to confirm they fail**

Run:
```bash
cd /home/kirill/speech-spoof-bench/speech-spoof-bench
pytest tests/test_cli.py::test_cli_manifest_prints_yaml tests/test_cli.py::test_cli_list_prints_core_then_extended -v
```
Expected: both FAIL — `manifest` subcommand doesn't exist; `list` still prints the phase-4 placeholder.

- [ ] **Step 3: Wire the CLI**

In `src/speech_spoof_bench/cli.py`:

(a) Replace `_cmd_list` body so it imports the manifest module lazily, fetches, and prints `[core]`/`[ext]` lines. Replace:

```python
def _cmd_list(args: argparse.Namespace) -> int:
    print("listing manifest datasets is part of phase 4", file=sys.stderr)
    return 2
```

with:

```python
def _cmd_list(args: argparse.Namespace) -> int:
    from . import manifest as mf

    m = mf.fetch_manifest()
    for entry in m["core_set"]:
        print(f"[core] {entry['id']}")
    for entry in m["extended"]:
        print(f"[ext]  {entry['id']}")
    return 0
```

(b) Add a new command handler. Insert above `def build_parser` (i.e. just below `_cmd_validate_dataset`):

```python
def _cmd_manifest(args: argparse.Namespace) -> int:
    """Print the raw manifest.yaml contents verbatim to stdout."""
    from pathlib import Path
    from . import manifest as mf

    # Reuse the repo coordinates from the manifest module, but skip the
    # parse/validate round-trip so the output equals the upstream file byte-for-byte.
    local = mf.hf_hub_download(
        repo_id=mf.MANIFEST_REPO,
        repo_type="dataset",
        filename=mf.MANIFEST_FILENAME,
    )
    sys.stdout.write(Path(local).read_text())
    return 0
```

Note: the handler calls `mf.hf_hub_download` (the module attribute), not a direct `from huggingface_hub import hf_hub_download`. The test monkeypatches that attribute, so calling through `mf.` is what makes the CLI test work.

(c) Register the subparser. Inside `build_parser`, just below the `lst = sub.add_parser("list", ...)` block, add:

```python
    mf = sub.add_parser("manifest",
                        help="print the arena-manifest YAML contents")
    mf.set_defaults(func=_cmd_manifest)
```

- [ ] **Step 4: Run the CLI tests until they pass**

Run:
```bash
cd /home/kirill/speech-spoof-bench/speech-spoof-bench
pytest tests/test_cli.py -v
```
Expected: all CLI tests PASS, including the new `manifest` and `list` cases. The deleted `test_cli_list_raises_at_phase_2` no longer exists.

- [ ] **Step 5: Commit**

```bash
cd /home/kirill/speech-spoof-bench/speech-spoof-bench
git add src/speech_spoof_bench/cli.py tests/test_cli.py
git commit -m "feat(cli): wire 'manifest' and 'list' subcommands against arena-manifest"
```

---

## Task 5: Ensure the JSON Schema ships with the package

**Files:**
- Modify: `pyproject.toml`

Without explicit package-data, setuptools will not include `*.json` files inside the package, which would break `manifest` imports for end users (it works in-place under `pip install -e .` because the source tree is used directly).

- [ ] **Step 1: Add package-data to `pyproject.toml`**

Append to `/home/kirill/speech-spoof-bench/speech-spoof-bench/pyproject.toml`:

```toml
[tool.setuptools.package-data]
"speech_spoof_bench.schema" = ["*.json"]
```

- [ ] **Step 2: Verify it builds and includes the JSON**

Run:
```bash
cd /home/kirill/speech-spoof-bench/speech-spoof-bench
python -m pip install --quiet build
python -m build --wheel --outdir /tmp/ssb_wheel
unzip -l /tmp/ssb_wheel/speech_spoof_bench-*.whl | grep manifest.schema.json
```
Expected: the `unzip -l` output contains a line ending in `speech_spoof_bench/schema/manifest.schema.json`.

- [ ] **Step 3: Clean up the temp wheel and commit**

```bash
rm -rf /tmp/ssb_wheel
cd /home/kirill/speech-spoof-bench/speech-spoof-bench
git add pyproject.toml
git commit -m "build: ship manifest JSON Schema with the wheel"
```

---

## Task 6: End-to-end DoD verification

No code changes — this is the Phase 4 acceptance test from ROADMAP.

- [ ] **Step 1: Run the full test suite once**

Run:
```bash
cd /home/kirill/speech-spoof-bench/speech-spoof-bench
pytest -q
```
Expected: all tests PASS, no errors.

- [ ] **Step 2: Run `speech-spoof-bench manifest` against the real HF repo**

Run:
```bash
cd /home/kirill/speech-spoof-bench/speech-spoof-bench
pip install -e . --quiet
speech-spoof-bench manifest
```
Expected: prints the YAML manifest body — same content as `arena-manifest/manifest.yaml`.

- [ ] **Step 3: Run `speech-spoof-bench list`**

Run:
```bash
speech-spoof-bench list
```
Expected: one line: `[core] SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA`. No `[ext]` lines (extended is empty).

- [ ] **Step 4: Tick the ROADMAP checkboxes**

In `/home/kirill/speech-spoof-bench/speech-spoof-bench/docs/roadmap/ROADMAP.md`, in the Phase 4 section, change the two unchecked items to checked:

- `- [ ] manifest.yaml per §4 ...` → `- [x] ...`
- `- [ ] Push to huggingface.co/datasets/...` → `- [x] ...`

Also tick the Phase 4 Done-when bullet if/when present.

Run:
```bash
cd /home/kirill/speech-spoof-bench/speech-spoof-bench
git add docs/roadmap/ROADMAP.md
git commit -m "docs(roadmap): mark Phase 4 complete"
```

---

## Notes / non-goals

- No `--manifest-path` CLI flag at launch. `load_manifest(path)` is the offline/test surface.
- No retry/backoff around `hf_hub_download` — let the network errors bubble. Adding retry is a separate, later concern.
- No Arena code is touched in this phase. Arena consumes the manifest in Phase 5.
- No live-HF unit test. The Task 6 manual run is the integration check; CI doesn't need a network round-trip for Phase 4.
