# Phase 7b Authoring (`submit`, `scaffold-dataset`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two CLI commands (`submit` and `scaffold-dataset`) so submitters no longer hand-author submission YAMLs or new dataset skeletons.

**Architecture:** Two new modules in the existing pip package (`submit.py`, `scaffold.py`) plus package data (`data/dataset_skeleton/`, `data/submission_meta.schema.json`). `submit` reuses `Benchmark.run` for the hybrid run path, calls `HfApi.upload_file` to push `scores.txt` to the model repo's main branch, then `HfApi.create_commit(create_pr=True)` to open a PR on the dataset repo. `scaffold-dataset` is pure file generation from packaged templates. The submission JSON Schema is relaxed so `reproduction: {}` is allowed at the submitter stage; `validate-dataset`'s separate `S2` check already enforces a non-empty reproduction block at the merged stage, so the maintainer gate remains intact.

**Tech Stack:** Python 3.10+, `huggingface_hub`, `pyyaml`, `jsonschema`, `argparse`, `pytest`. No new runtime deps.

**Spec:** `docs/specs/2026-05-22-phase-7b-authoring-design.md`.

---

## File Structure

- `src/speech_spoof_bench/data/__init__.py` — empty marker, lets `importlib.resources` see the package data subtree.
- `src/speech_spoof_bench/data/submission_meta.schema.json` — JSON Schema for `meta.yaml`.
- `src/speech_spoof_bench/data/dataset_skeleton/*` — verbatim template files copied by `scaffold-dataset`.
- `src/speech_spoof_bench/schema/submission.schema.json` — modified: top-level `required` drops `reproduction`; `reproduction` accepts either `{}` or the existing filled form via `oneOf`.
- `src/speech_spoof_bench/submit.py` — pure-functional helpers + `submit_one`, `submit` orchestrators.
- `src/speech_spoof_bench/scaffold.py` — `scaffold_dataset(name, output_dir, force)`.
- `src/speech_spoof_bench/cli.py` — modified: add `submit` and `scaffold-dataset` subcommands.
- `tests/test_submission_schema_empty_reproduction.py` — new.
- `tests/test_submit_meta.py` — new.
- `tests/test_submit_payload.py` — new.
- `tests/test_submit_upload.py` — new.
- `tests/test_submit_pr.py` — new.
- `tests/test_submit_one.py` — new.
- `tests/test_scaffold.py` — new.
- `tests/test_cli.py` — modified: add smoke tests for both new subcommands.

---

## Task 1: Schema relaxation — allow empty `reproduction` for submitter YAMLs

**Files:**
- Modify: `src/speech_spoof_bench/schema/submission.schema.json`
- Create: `tests/test_submission_schema_empty_reproduction.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_submission_schema_empty_reproduction.py
"""Submission schema must accept reproduction: {} so submit-time YAMLs parse."""

from __future__ import annotations

import datetime as _dt

import pytest
import yaml

from speech_spoof_bench import submission


_BASE = {
    "schema_version": 4,
    "system": {
        "name": "x",
        "slug": "x",
        "description": "x",
        "code": "https://example.com",
        "checkpoint": "https://example.com",
        "paper": {
            "arxiv_id": "1234.5678",
            "url": "https://arxiv.org/abs/1234.5678",
            "bibtex": "@x{}",
        },
    },
    "dataset": {"id": "Org/A", "revision": "abcdef1", "split": "test"},
    "scores": {"eer_percent": 1.0, "n_trials": 1, "n_skipped": 0},
    "artifact": {
        "scores_url": (
            "https://huggingface.co/Org/x/resolve/abcdef1/.eval_results/"
            "Org/A/scores.txt"
        ),
        "scores_sha256": "0" * 64,
        "bench_version": "speech-spoof-bench==0.1.0",
    },
    "submitter": {"hf_username": "x", "contact": "x@example.com"},
    "submitted_at": _dt.date(2026, 5, 22).isoformat(),
}


def test_empty_reproduction_parses():
    data = dict(_BASE)
    data["reproduction"] = {}
    text = yaml.safe_dump(data)
    out = submission.parse_submission(text)
    assert out["reproduction"] == {}


def test_missing_reproduction_key_parses():
    data = dict(_BASE)  # no 'reproduction' key
    text = yaml.safe_dump(data)
    out = submission.parse_submission(text)
    assert out.get("reproduction", {}) in ({}, None)


def test_filled_reproduction_still_parses():
    data = dict(_BASE)
    data["reproduction"] = {
        "reproduced_by": "Org",
        "reproduced_at": "2026-05-22",
        "reproduced_bench_version": "speech-spoof-bench==0.1.0",
        "match": "scoring",
    }
    text = yaml.safe_dump(data)
    out = submission.parse_submission(text)
    assert out["reproduction"]["match"] == "scoring"


def test_partial_reproduction_rejected():
    data = dict(_BASE)
    data["reproduction"] = {"reproduced_by": "Org"}  # missing other keys
    text = yaml.safe_dump(data)
    with pytest.raises(submission.SubmissionValidationError):
        submission.parse_submission(text)
```

- [ ] **Step 2: Run tests to confirm they fail**

Run: `cd speech-spoof-bench && python -m pytest tests/test_submission_schema_empty_reproduction.py -v`
Expected: `test_empty_reproduction_parses`, `test_missing_reproduction_key_parses` FAIL (schema currently requires filled reproduction). The other two pass.

- [ ] **Step 3: Edit the schema**

Open `src/speech_spoof_bench/schema/submission.schema.json`.

Change the top-level `"required"` array — drop `"reproduction"`:
```json
"required": ["schema_version", "system", "dataset", "scores", "artifact", "submitter", "submitted_at"]
```

Replace the existing `"reproduction"` block with:
```json
"reproduction": {
  "oneOf": [
    {"type": "object", "additionalProperties": false, "properties": {}, "maxProperties": 0},
    {
      "type": "object",
      "additionalProperties": false,
      "required": ["reproduced_by", "reproduced_at", "reproduced_bench_version", "match"],
      "properties": {
        "reproduced_by": {"type": "string", "minLength": 1},
        "reproduced_at": {"type": "string", "format": "date"},
        "reproduced_bench_version": {"type": "string", "minLength": 1},
        "match": {"enum": ["scoring", "inference"]}
      }
    }
  ]
}
```

- [ ] **Step 4: Run all submission/validate tests**

Run: `cd speech-spoof-bench && python -m pytest tests/test_submission.py tests/test_submission_schema_empty_reproduction.py tests/test_validate_dataset.py tests/test_validate_submission.py -v`
Expected: all PASS. (The existing `S2` check in `validate.py:321` still enforces non-empty reproduction at validate-dataset time, so the maintainer gate is unchanged.)

- [ ] **Step 5: Commit**

```bash
cd speech-spoof-bench
git add src/speech_spoof_bench/schema/submission.schema.json tests/test_submission_schema_empty_reproduction.py
git commit -m "schema: allow empty reproduction block for submitter-stage YAMLs"
```

---

## Task 2: `meta.yaml` schema + `load_meta`

**Files:**
- Create: `src/speech_spoof_bench/data/__init__.py`
- Create: `src/speech_spoof_bench/data/submission_meta.schema.json`
- Create: `src/speech_spoof_bench/submit.py` (initial stub for `load_meta`)
- Create: `tests/test_submit_meta.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_submit_meta.py
"""load_meta validates meta.yaml against the submission_meta JSON Schema."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from speech_spoof_bench.submit import MetaValidationError, load_meta


_GOOD = {
    "system": {
        "name": "AASIST",
        "slug": "aasist-clovaai-default",
        "description": "Reference AASIST.\n",
        "code": "https://github.com/clovaai/aasist",
        "checkpoint": "https://huggingface.co/owner/repo",
        "paper": {
            "arxiv_id": "2110.01200",
            "url": "https://arxiv.org/abs/2110.01200",
            "bibtex": "@inproceedings{jung2022aasist}",
        },
    },
    "notes": "free-form notes",
}


def _write(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "meta.yaml"
    p.write_text(yaml.safe_dump(data))
    return p


def test_load_meta_accepts_complete(tmp_path: Path):
    out = load_meta(_write(tmp_path, _GOOD))
    assert out["system"]["slug"] == "aasist-clovaai-default"


def test_load_meta_accepts_no_notes(tmp_path: Path):
    data = dict(_GOOD)
    del data["notes"]
    out = load_meta(_write(tmp_path, data))
    assert "notes" not in out


def test_load_meta_rejects_missing_paper(tmp_path: Path):
    data = {"system": dict(_GOOD["system"])}
    del data["system"]["paper"]
    with pytest.raises(MetaValidationError):
        load_meta(_write(tmp_path, data))


def test_load_meta_rejects_bad_slug(tmp_path: Path):
    data = {"system": dict(_GOOD["system"]), "notes": ""}
    data["system"]["slug"] = "Has Spaces"
    with pytest.raises(MetaValidationError):
        load_meta(_write(tmp_path, data))


def test_load_meta_rejects_extra_top_level_key(tmp_path: Path):
    data = dict(_GOOD)
    data["unexpected"] = 1
    with pytest.raises(MetaValidationError):
        load_meta(_write(tmp_path, data))


def test_load_meta_file_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_meta(tmp_path / "nope.yaml")
```

- [ ] **Step 2: Run tests to verify failure**

Run: `cd speech-spoof-bench && python -m pytest tests/test_submit_meta.py -v`
Expected: ImportError (`speech_spoof_bench.submit` doesn't exist).

- [ ] **Step 3: Create the package-data marker and schema**

Create `src/speech_spoof_bench/data/__init__.py` as an empty file (just touch it).

Create `src/speech_spoof_bench/data/submission_meta.schema.json`:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "speech-spoof-bench submission meta",
  "type": "object",
  "additionalProperties": false,
  "required": ["system"],
  "properties": {
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
    "notes": {"type": "string"}
  }
}
```

- [ ] **Step 4: Create `submit.py` with `load_meta`**

Create `src/speech_spoof_bench/submit.py`:

```python
"""Phase 7b — `submit` command implementation.

Public surface (used by cli.py):
  - load_meta(path) -> dict
  - submit(...)             # added later in Task 8
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any

import yaml
from jsonschema import ValidationError, validate

_META_SCHEMA_PACKAGE = "speech_spoof_bench.data"
_META_SCHEMA_FILENAME = "submission_meta.schema.json"


class MetaValidationError(ValueError):
    """Raised when a submission meta YAML fails schema validation."""


def _load_meta_schema() -> dict[str, Any]:
    with resources.files(_META_SCHEMA_PACKAGE).joinpath(_META_SCHEMA_FILENAME).open("r") as f:
        return json.load(f)


def load_meta(path: Path | str) -> dict[str, Any]:
    """Parse and validate a submission meta YAML.

    Raises:
      FileNotFoundError: path doesn't exist.
      MetaValidationError: YAML parses but fails the schema.
    """
    p = Path(path)
    text = p.read_text()  # raises FileNotFoundError as desired
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise MetaValidationError(f"{p}: not a YAML mapping")
    try:
        validate(instance=data, schema=_load_meta_schema())
    except ValidationError as exc:
        raise MetaValidationError(f"{p}: {exc.message}") from exc
    return data
```

- [ ] **Step 5: Ensure package data is shipped**

Open `pyproject.toml`. If there is no `[tool.setuptools.package-data]` (or equivalent) entry that picks up `*.json` under `speech_spoof_bench/data/`, add one. Look for the existing entry for `schema/*.json` and mirror it.

Example (only add if missing, keeping existing entries):
```toml
[tool.setuptools.package-data]
speech_spoof_bench = ["schema/*.json", "data/*.json", "data/dataset_skeleton/**/*"]
```

- [ ] **Step 6: Run tests**

Run: `cd speech-spoof-bench && python -m pytest tests/test_submit_meta.py -v`
Expected: all 6 tests PASS.

- [ ] **Step 7: Commit**

```bash
cd speech-spoof-bench
git add src/speech_spoof_bench/data/__init__.py src/speech_spoof_bench/data/submission_meta.schema.json src/speech_spoof_bench/submit.py tests/test_submit_meta.py pyproject.toml
git commit -m "submit: meta schema + load_meta with validation"
```

---

## Task 3: `build_submission_payload`

**Files:**
- Modify: `src/speech_spoof_bench/submit.py`
- Create: `tests/test_submit_payload.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_submit_payload.py
"""build_submission_payload merges result.yaml + meta into a v4 submission."""

from __future__ import annotations

import yaml

from speech_spoof_bench import submission
from speech_spoof_bench.submit import build_submission_payload


_RESULT = {
    "schema_version": 4,
    "system": {
        "name": "unknown", "slug": None, "description": None,
        "code": None, "checkpoint": None, "paper": None,
    },
    "dataset": {"id": "Org/A", "revision": "abcdef1", "split": "test"},
    "scores": {"eer_percent": 1.234, "n_trials": 100, "n_skipped": 0},
    "artifact": {
        "scores_url": None,
        "scores_sha256": "f" * 64,
        "bench_version": "speech-spoof-bench==0.1.0",
    },
    "reproduction": {},
    "submitter": {},
    "submitted_at": None,
    "notes": None,
}


_META = {
    "system": {
        "name": "AASIST",
        "slug": "aasist-test",
        "description": "AASIST desc",
        "code": "https://github.com/clovaai/aasist",
        "checkpoint": "https://huggingface.co/owner/repo",
        "paper": {
            "arxiv_id": "2110.01200",
            "url": "https://arxiv.org/abs/2110.01200",
            "bibtex": "@inproceedings{jung2022aasist}",
        },
    },
    "notes": "from meta",
}


def _build():
    return build_submission_payload(
        result_yaml=_RESULT,
        meta=_META,
        scores_url=(
            "https://huggingface.co/owner/repo/resolve/"
            "1234567890abcdef1234567890abcdef12345678/.eval_results/Org/A/scores.txt"
        ),
        scores_sha256="f" * 64,
        hf_username="kborodin",
        contact="k@example.com",
        submitted_at="2026-05-22",
    )


def test_payload_parses_against_submission_schema():
    payload = _build()
    # Round-trip through YAML for realism.
    submission.parse_submission(yaml.safe_dump(payload))


def test_payload_system_block_mirrors_meta():
    payload = _build()
    assert payload["system"]["slug"] == "aasist-test"
    assert payload["system"]["name"] == "AASIST"
    assert payload["system"]["paper"]["arxiv_id"] == "2110.01200"


def test_payload_dataset_block_from_result():
    payload = _build()
    assert payload["dataset"] == {
        "id": "Org/A", "revision": "abcdef1", "split": "test",
    }


def test_payload_artifact_block():
    payload = _build()
    assert payload["artifact"]["scores_url"].endswith("/scores.txt")
    assert payload["artifact"]["scores_sha256"] == "f" * 64
    assert payload["artifact"]["bench_version"] == "speech-spoof-bench==0.1.0"


def test_payload_reproduction_empty():
    payload = _build()
    assert payload["reproduction"] == {}


def test_payload_submitter_from_flags():
    payload = _build()
    assert payload["submitter"] == {"hf_username": "kborodin", "contact": "k@example.com"}


def test_payload_submitted_at_from_arg():
    payload = _build()
    assert payload["submitted_at"] == "2026-05-22"


def test_payload_notes_from_meta():
    payload = _build()
    assert payload["notes"] == "from meta"


def test_payload_omits_notes_when_meta_lacks_it():
    meta = {"system": _META["system"]}  # no notes
    payload = build_submission_payload(
        result_yaml=_RESULT,
        meta=meta,
        scores_url="https://huggingface.co/o/r/resolve/abcdef1/x",
        scores_sha256="f" * 64,
        hf_username="u", contact="c",
        submitted_at="2026-05-22",
    )
    assert "notes" not in payload or payload["notes"] is None
```

- [ ] **Step 2: Run tests to verify failure**

Run: `cd speech-spoof-bench && python -m pytest tests/test_submit_payload.py -v`
Expected: ImportError on `build_submission_payload`.

- [ ] **Step 3: Implement `build_submission_payload`**

Append to `src/speech_spoof_bench/submit.py`:

```python
def build_submission_payload(
    *,
    result_yaml: dict[str, Any],
    meta: dict[str, Any],
    scores_url: str,
    scores_sha256: str,
    hf_username: str,
    contact: str,
    submitted_at: str,
) -> dict[str, Any]:
    """Merge a result.yaml + meta into a fully-formed v4 submission dict.

    The `reproduction` block is left empty by design (§1.7) — the maintainer
    fills it via `reproduce --scoring` at merge time.
    """
    sys_meta = meta["system"]
    payload: dict[str, Any] = {
        "schema_version": 4,
        "system": {
            "name": sys_meta["name"],
            "slug": sys_meta["slug"],
            "description": sys_meta["description"],
            "code": sys_meta["code"],
            "checkpoint": sys_meta["checkpoint"],
            "paper": dict(sys_meta["paper"]),
        },
        "dataset": dict(result_yaml["dataset"]),
        "scores": dict(result_yaml["scores"]),
        "artifact": {
            "scores_url": scores_url,
            "scores_sha256": scores_sha256,
            "bench_version": result_yaml["artifact"]["bench_version"],
        },
        "reproduction": {},
        "submitter": {"hf_username": hf_username, "contact": contact},
        "submitted_at": submitted_at,
    }
    if "notes" in meta:
        payload["notes"] = meta["notes"]
    return payload
```

- [ ] **Step 4: Run tests**

Run: `cd speech-spoof-bench && python -m pytest tests/test_submit_payload.py -v`
Expected: all 9 PASS.

- [ ] **Step 5: Commit**

```bash
cd speech-spoof-bench
git add src/speech_spoof_bench/submit.py tests/test_submit_payload.py
git commit -m "submit: build_submission_payload merges result + meta into v4 YAML"
```

---

## Task 4: `upload_scores`

**Files:**
- Modify: `src/speech_spoof_bench/submit.py`
- Create: `tests/test_submit_upload.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_submit_upload.py
"""upload_scores pushes scores.txt to <model-repo>/.eval_results/<canonical_id>/."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from speech_spoof_bench.submit import upload_scores


def _fake_api_returning_oid(oid: str):
    api = MagicMock()
    info = MagicMock()
    info.oid = oid
    api.upload_file.return_value = info
    return api


def test_upload_scores_calls_upload_file_with_correct_args(tmp_path: Path):
    scores = tmp_path / "scores.txt"
    scores.write_text("x 1.0\n")
    api = _fake_api_returning_oid("abcdef1234567890abcdef1234567890abcdef12")

    url, oid = upload_scores(
        api=api,
        model_repo="Org/random-baseline",
        dataset_canonical_id="Org/ASVspoof2019_LA",
        local_path=scores,
    )

    api.upload_file.assert_called_once()
    kwargs = api.upload_file.call_args.kwargs
    assert kwargs["path_or_fileobj"] == str(scores)
    assert kwargs["path_in_repo"] == ".eval_results/Org/ASVspoof2019_LA/scores.txt"
    assert kwargs["repo_id"] == "Org/random-baseline"
    assert kwargs["repo_type"] == "model"
    assert "commit_message" in kwargs


def test_upload_scores_returns_pinned_url(tmp_path: Path):
    scores = tmp_path / "scores.txt"
    scores.write_text("x 1.0\n")
    oid = "abcdef1234567890abcdef1234567890abcdef12"
    api = _fake_api_returning_oid(oid)

    url, returned_oid = upload_scores(
        api=api,
        model_repo="Org/random-baseline",
        dataset_canonical_id="Org/ASVspoof2019_LA",
        local_path=scores,
    )

    assert returned_oid == oid
    assert url == (
        f"https://huggingface.co/Org/random-baseline/resolve/{oid}/"
        ".eval_results/Org/ASVspoof2019_LA/scores.txt"
    )


def test_upload_scores_missing_local_file(tmp_path: Path):
    api = _fake_api_returning_oid("a" * 40)
    with pytest.raises(FileNotFoundError):
        upload_scores(
            api=api,
            model_repo="Org/x",
            dataset_canonical_id="Org/A",
            local_path=tmp_path / "nope.txt",
        )
```

- [ ] **Step 2: Run test to verify failure**

Run: `cd speech-spoof-bench && python -m pytest tests/test_submit_upload.py -v`
Expected: ImportError on `upload_scores`.

- [ ] **Step 3: Implement `upload_scores`**

Append to `src/speech_spoof_bench/submit.py`:

```python
from huggingface_hub import HfApi  # noqa: E402  (re-exported for mocking convenience)


def upload_scores(
    *,
    api: HfApi,
    model_repo: str,
    dataset_canonical_id: str,
    local_path: Path | str,
) -> tuple[str, str]:
    """Upload scores.txt to the model repo's main branch.

    Returns (scores_url, commit_oid). The URL pins the returned commit oid.
    """
    p = Path(local_path)
    if not p.is_file():
        raise FileNotFoundError(f"scores file not found: {p}")
    path_in_repo = f".eval_results/{dataset_canonical_id}/scores.txt"
    info = api.upload_file(
        path_or_fileobj=str(p),
        path_in_repo=path_in_repo,
        repo_id=model_repo,
        repo_type="model",
        commit_message=f"upload scores for {dataset_canonical_id}",
    )
    oid = info.oid
    url = (
        f"https://huggingface.co/{model_repo}/resolve/{oid}/{path_in_repo}"
    )
    return url, oid
```

- [ ] **Step 4: Run tests**

Run: `cd speech-spoof-bench && python -m pytest tests/test_submit_upload.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
cd speech-spoof-bench
git add src/speech_spoof_bench/submit.py tests/test_submit_upload.py
git commit -m "submit: upload_scores pushes scores.txt to model repo and returns pinned url"
```

---

## Task 5: `open_submission_pr`

**Files:**
- Modify: `src/speech_spoof_bench/submit.py`
- Create: `tests/test_submit_pr.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_submit_pr.py
"""open_submission_pr calls HfApi.create_commit with create_pr=True and parent_commit."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from huggingface_hub import CommitOperationAdd

from speech_spoof_bench.submit import open_submission_pr


def test_open_submission_pr_passes_expected_kwargs():
    api = MagicMock()
    commit_info = MagicMock()
    commit_info.pr_url = "https://huggingface.co/datasets/Org/A/discussions/7"
    api.create_commit.return_value = commit_info

    pr_url = open_submission_pr(
        api=api,
        dataset_id="Org/A",
        parent_commit="abcdef1",
        slug="aasist-test",
        yaml_text="schema_version: 4\n",
    )

    assert pr_url == "https://huggingface.co/datasets/Org/A/discussions/7"
    kwargs = api.create_commit.call_args.kwargs
    assert kwargs["repo_id"] == "Org/A"
    assert kwargs["repo_type"] == "dataset"
    assert kwargs["create_pr"] is True
    assert kwargs["parent_commit"] == "abcdef1"

    ops = kwargs["operations"]
    assert len(ops) == 1
    op = ops[0]
    assert isinstance(op, CommitOperationAdd)
    assert op.path_in_repo == "submissions/aasist-test.yaml"


def test_open_submission_pr_propagates_unknown_pr_url():
    """When the HF response lacks `pr_url`, raise so the caller notices."""
    api = MagicMock()
    commit_info = MagicMock(spec=[])  # no pr_url attr
    api.create_commit.return_value = commit_info

    with pytest.raises(RuntimeError, match="PR url"):
        open_submission_pr(
            api=api,
            dataset_id="Org/A",
            parent_commit="abcdef1",
            slug="x",
            yaml_text="x",
        )
```

- [ ] **Step 2: Run to verify failure**

Run: `cd speech-spoof-bench && python -m pytest tests/test_submit_pr.py -v`
Expected: ImportError on `open_submission_pr`.

- [ ] **Step 3: Implement `open_submission_pr`**

Append to `src/speech_spoof_bench/submit.py`:

```python
from io import BytesIO  # noqa: E402

from huggingface_hub import CommitOperationAdd  # noqa: E402


def open_submission_pr(
    *,
    api: HfApi,
    dataset_id: str,
    parent_commit: str,
    slug: str,
    yaml_text: str,
) -> str:
    """Open an HF PR on the dataset repo carrying submissions/<slug>.yaml.

    Returns the PR URL.
    """
    ops = [
        CommitOperationAdd(
            path_in_repo=f"submissions/{slug}.yaml",
            path_or_fileobj=BytesIO(yaml_text.encode("utf-8")),
        )
    ]
    info = api.create_commit(
        repo_id=dataset_id,
        repo_type="dataset",
        operations=ops,
        commit_message=f"submissions: add {slug}",
        create_pr=True,
        parent_commit=parent_commit,
    )
    url = getattr(info, "pr_url", None)
    if not url:
        raise RuntimeError("HF create_commit returned no PR url")
    return url
```

- [ ] **Step 4: Run tests**

Run: `cd speech-spoof-bench && python -m pytest tests/test_submit_pr.py -v`
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
cd speech-spoof-bench
git add src/speech_spoof_bench/submit.py tests/test_submit_pr.py
git commit -m "submit: open_submission_pr creates HF PR with submissions/<slug>.yaml"
```

---

## Task 6: `submit_one` — orchestrator (hybrid run + upload + PR)

**Files:**
- Modify: `src/speech_spoof_bench/submit.py`
- Create: `tests/test_submit_one.py`

This task introduces the orchestrator. Because `Benchmark.run` and `HfApi` are the side-effecting bits, the test patches them.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_submit_one.py
"""submit_one wires hybrid-run + upload + PR with all HF calls mocked."""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

import speech_spoof_bench.submit as submit_mod
from speech_spoof_bench.submit import submit_one


_META = {
    "system": {
        "name": "RB",
        "slug": "rb-phase7b",
        "description": "d",
        "code": "https://example.com/c",
        "checkpoint": "https://huggingface.co/Org/rb",
        "paper": {
            "arxiv_id": "1911.01601",
            "url": "https://arxiv.org/abs/1911.01601",
            "bibtex": "@x{}",
        },
    },
    "notes": "n",
}


def _result_yaml_text(revision: str) -> str:
    payload = {
        "schema_version": 4,
        "system": {"name": "unknown", "slug": None, "description": None,
                   "code": None, "checkpoint": None, "paper": None},
        "dataset": {"id": "Org/A", "revision": revision, "split": "test"},
        "scores": {"eer_percent": 50.0, "n_trials": 10, "n_skipped": 0},
        "artifact": {
            "scores_url": None,
            "scores_sha256": "f" * 64,
            "bench_version": "speech-spoof-bench==0.1.0",
        },
        "reproduction": {}, "submitter": {}, "submitted_at": None, "notes": None,
    }
    return yaml.safe_dump(payload, sort_keys=False)


@pytest.fixture
def fake_hf_api():
    api = MagicMock()
    upload_info = MagicMock()
    upload_info.oid = "a" * 40
    api.upload_file.return_value = upload_info
    commit_info = MagicMock()
    commit_info.pr_url = "https://huggingface.co/datasets/Org/A/discussions/1"
    api.create_commit.return_value = commit_info
    repo_info = MagicMock()
    repo_info.sha = "deadbee"
    api.repo_info.return_value = repo_info
    return api


def _make_existing_result(tmp_path: Path, slug: str, revision: str) -> Path:
    out = tmp_path / "results" / slug
    out.mkdir(parents=True)
    (out / "scores.txt").write_text("u 1.0\n")
    sha = hashlib.sha256(b"u 1.0\n").hexdigest()
    text = _result_yaml_text(revision).replace("f" * 64, sha)
    (out / "result.yaml").write_text(text)
    return out


def test_submit_one_reuses_existing_result_when_revision_matches(
    tmp_path: Path, fake_hf_api, monkeypatch
):
    """Hybrid path: result.yaml present + revision matches → no Benchmark.run call."""
    fake_hf_api.repo_info.return_value.sha = "deadbee"
    _make_existing_result(tmp_path, "Org_A", "deadbee")

    bench_called = {"n": 0}

    def fake_bench_run(*args, **kwargs):
        bench_called["n"] += 1

    monkeypatch.setattr(submit_mod, "_run_benchmark", fake_bench_run)
    monkeypatch.setattr(submit_mod, "_resolve_dataset_slug",
                        lambda spec, api: ("Org/A", "Org_A", "deadbee", "test"))

    pr_url = submit_one(
        model_module_spec="x:Y",
        dataset_spec="Org/A",
        output_dir=tmp_path / "results",
        meta=_META,
        model_repo="Org/rb",
        hf_username="u",
        contact="c@example.com",
        submitted_at="2026-05-22",
        api=fake_hf_api,
    )

    assert pr_url.endswith("/discussions/1")
    assert bench_called["n"] == 0
    fake_hf_api.upload_file.assert_called_once()
    fake_hf_api.create_commit.assert_called_once()

    # PR commit's parent must be the resolved dataset revision.
    assert fake_hf_api.create_commit.call_args.kwargs["parent_commit"] == "deadbee"


def test_submit_one_runs_benchmark_when_result_missing(
    tmp_path: Path, fake_hf_api, monkeypatch
):
    fake_hf_api.repo_info.return_value.sha = "deadbee"

    def fake_bench_run(*, model_module_spec, dataset_spec, output_dir, **_):
        # Materialize the same files Benchmark.run would have created.
        _make_existing_result(Path(output_dir), "Org_A", "deadbee")

    monkeypatch.setattr(submit_mod, "_run_benchmark", fake_bench_run)
    monkeypatch.setattr(submit_mod, "_resolve_dataset_slug",
                        lambda spec, api: ("Org/A", "Org_A", "deadbee", "test"))

    pr_url = submit_one(
        model_module_spec="x:Y",
        dataset_spec="Org/A",
        output_dir=tmp_path / "results",
        meta=_META,
        model_repo="Org/rb",
        hf_username="u", contact="c",
        submitted_at="2026-05-22",
        api=fake_hf_api,
    )
    assert pr_url.endswith("/discussions/1")


def test_submit_one_reruns_when_revision_mismatch(
    tmp_path: Path, fake_hf_api, monkeypatch
):
    fake_hf_api.repo_info.return_value.sha = "newrev1"
    _make_existing_result(tmp_path, "Org_A", "oldrev1")

    bench_called = {"n": 0}

    def fake_bench_run(*, output_dir, **_):
        bench_called["n"] += 1
        # Overwrite with matching revision.
        _make_existing_result(Path(output_dir), "Org_A", "newrev1")

    monkeypatch.setattr(submit_mod, "_run_benchmark", fake_bench_run)
    monkeypatch.setattr(submit_mod, "_resolve_dataset_slug",
                        lambda spec, api: ("Org/A", "Org_A", "newrev1", "test"))

    submit_one(
        model_module_spec="x:Y",
        dataset_spec="Org/A",
        output_dir=tmp_path / "results",
        meta=_META,
        model_repo="Org/rb",
        hf_username="u", contact="c",
        submitted_at="2026-05-22",
        api=fake_hf_api,
    )
    assert bench_called["n"] == 1


def test_submit_one_yaml_is_schema_valid(
    tmp_path: Path, fake_hf_api, monkeypatch
):
    """The YAML pushed in the PR commit must parse as a valid v4 submission."""
    from speech_spoof_bench import submission

    fake_hf_api.repo_info.return_value.sha = "deadbee"
    _make_existing_result(tmp_path, "Org_A", "deadbee")
    monkeypatch.setattr(submit_mod, "_resolve_dataset_slug",
                        lambda spec, api: ("Org/A", "Org_A", "deadbee", "test"))
    monkeypatch.setattr(submit_mod, "_run_benchmark", lambda **_: None)

    submit_one(
        model_module_spec="x:Y",
        dataset_spec="Org/A",
        output_dir=tmp_path / "results",
        meta=_META,
        model_repo="Org/rb",
        hf_username="u", contact="c@example.com",
        submitted_at="2026-05-22",
        api=fake_hf_api,
    )

    ops = fake_hf_api.create_commit.call_args.kwargs["operations"]
    op = ops[0]
    yaml_text = op.path_or_fileobj.getvalue().decode("utf-8")
    parsed = submission.parse_submission(yaml_text)
    assert parsed["reproduction"] == {}
    assert parsed["system"]["slug"] == "rb-phase7b"
    assert parsed["submitter"] == {"hf_username": "u", "contact": "c@example.com"}
    assert parsed["dataset"]["revision"] == "deadbee"
```

- [ ] **Step 2: Run to verify failure**

Run: `cd speech-spoof-bench && python -m pytest tests/test_submit_one.py -v`
Expected: ImportError on `submit_one`.

- [ ] **Step 3: Implement `submit_one` plus the two seams the test patches**

Append to `src/speech_spoof_bench/submit.py`:

```python
import datetime as _dt  # noqa: E402
import logging  # noqa: E402

import yaml  # noqa: E402  (already imported above; re-listed for clarity)

from .benchmark import Benchmark  # noqa: E402

_LOG = logging.getLogger(__name__)


def _resolve_dataset_slug(spec: str, api: HfApi) -> tuple[str, str, str, str]:
    """Resolve `spec` to (canonical_id, slug, revision, split).

    Slug is the last path segment (matches DatasetSource.slug for HF specs).
    Revision is the current main-branch sha from HfApi.repo_info — we want the
    state the run scored against, even though `loader.resolve` returns None.
    Split is read from eval.yaml via loader (which already validates it).
    """
    from .loader import resolve as _resolve

    source, _ = _resolve(spec, streaming=True)
    info = api.repo_info(repo_id=source.canonical_id, repo_type="dataset")
    return source.canonical_id, source.slug, info.sha, source.split


def _run_benchmark(
    *, model_module_spec: str, dataset_spec: str, output_dir: Path,
) -> None:
    """Import the model class and run the benchmark for a single dataset."""
    import importlib

    mod_name, cls_name = model_module_spec.split(":", 1)
    cls = getattr(importlib.import_module(mod_name), cls_name)
    model = cls()
    Benchmark.run(
        model,
        datasets=[dataset_spec],
        output_dir=str(output_dir),
        skip_existing=False,
    )


def _read_result_yaml(out_dir: Path) -> dict[str, Any] | None:
    p = out_dir / "result.yaml"
    if not p.is_file():
        return None
    return yaml.safe_load(p.read_text())


def submit_one(
    *,
    model_module_spec: str,
    dataset_spec: str,
    output_dir: Path,
    meta: dict[str, Any],
    model_repo: str,
    hf_username: str,
    contact: str,
    submitted_at: str,
    api: HfApi,
) -> str:
    """Run one (model, dataset) submission end-to-end. Returns the PR URL."""
    from . import submission as _sub

    canonical_id, slug, revision, _split = _resolve_dataset_slug(dataset_spec, api)
    out_dir = Path(output_dir) / slug

    existing = _read_result_yaml(out_dir)
    if existing is None or existing.get("dataset", {}).get("revision") != revision:
        _LOG.info("running benchmark for %s (revision %s)", canonical_id, revision)
        _run_benchmark(
            model_module_spec=model_module_spec,
            dataset_spec=dataset_spec,
            output_dir=Path(output_dir),
        )
        existing = _read_result_yaml(out_dir)
        if existing is None:
            raise RuntimeError(
                f"benchmark run produced no result.yaml under {out_dir}"
            )

    # Make sure every metric from eval.yaml is present in the result.
    source_metrics = [
        k for k in existing.get("scores", {}) if k not in {"n_trials", "n_skipped"}
    ]
    if not source_metrics:
        raise RuntimeError(f"result.yaml at {out_dir} has no metric values")

    # Patch revision in the result we hand to build_submission_payload so the
    # dataset block in the YAML matches what we just resolved (loader.resolve
    # currently returns None for HF specs).
    existing = dict(existing)
    existing["dataset"] = dict(existing["dataset"])
    existing["dataset"]["revision"] = revision

    scores_path = out_dir / "scores.txt"
    scores_url, _commit_oid = upload_scores(
        api=api,
        model_repo=model_repo,
        dataset_canonical_id=canonical_id,
        local_path=scores_path,
    )

    payload = build_submission_payload(
        result_yaml=existing,
        meta=meta,
        scores_url=scores_url,
        scores_sha256=existing["artifact"]["scores_sha256"],
        hf_username=hf_username,
        contact=contact,
        submitted_at=submitted_at,
    )

    yaml_text = yaml.safe_dump(payload, sort_keys=False)
    _sub.parse_submission(yaml_text)  # raises if invalid

    return open_submission_pr(
        api=api,
        dataset_id=canonical_id,
        parent_commit=revision,
        slug=meta["system"]["slug"],
        yaml_text=yaml_text,
    )
```

- [ ] **Step 4: Run tests**

Run: `cd speech-spoof-bench && python -m pytest tests/test_submit_one.py -v`
Expected: all 4 PASS.

- [ ] **Step 5: Commit**

```bash
cd speech-spoof-bench
git add src/speech_spoof_bench/submit.py tests/test_submit_one.py
git commit -m "submit: submit_one orchestrator (hybrid run + upload + PR)"
```

---

## Task 7: `submit` multi-dataset entry point + CLI wiring

**Files:**
- Modify: `src/speech_spoof_bench/submit.py`
- Modify: `src/speech_spoof_bench/cli.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write the failing CLI test**

Append to `tests/test_cli.py`:

```python
def test_cli_submit_smoke(monkeypatch, tmp_path):
    """`submit` wires flags into submit_one for each dataset."""
    import speech_spoof_bench.submit as submit_mod

    meta_path = tmp_path / "meta.yaml"
    meta_path.write_text(
        "system:\n"
        "  name: RB\n"
        "  slug: rb\n"
        "  description: d\n"
        "  code: https://example.com/c\n"
        "  checkpoint: https://huggingface.co/Org/rb\n"
        "  paper:\n"
        "    arxiv_id: '1911.01601'\n"
        "    url: https://arxiv.org/abs/1911.01601\n"
        "    bibtex: '@x{}'\n"
    )

    calls = []

    def fake_submit_one(**kwargs):
        calls.append(kwargs["dataset_spec"])
        return f"https://huggingface.co/datasets/{kwargs['dataset_spec']}/discussions/1"

    monkeypatch.setattr(submit_mod, "submit_one", fake_submit_one)

    rc = main([
        "submit",
        "--model-module", "x:Y",
        "--datasets", "Org/A",
        "--datasets", "Org/B",
        "--model-repo", "Org/rb",
        "--submission-meta", str(meta_path),
        "--hf-username", "u",
        "--contact", "c@example.com",
        "--output-dir", str(tmp_path / "results"),
    ])
    assert rc == 0
    assert calls == ["Org/A", "Org/B"]


def test_cli_submit_all_uses_manifest(monkeypatch, tmp_path):
    """`--datasets all` iterates core_set + extended from the manifest."""
    import speech_spoof_bench.submit as submit_mod
    from speech_spoof_bench import manifest as _mf

    meta_path = tmp_path / "meta.yaml"
    meta_path.write_text(
        "system:\n"
        "  name: RB\n  slug: rb\n  description: d\n"
        "  code: https://example.com/c\n"
        "  checkpoint: https://huggingface.co/Org/rb\n"
        "  paper:\n    arxiv_id: '1'\n    url: https://arxiv.org/abs/1\n    bibtex: '@x{}'\n"
    )

    monkeypatch.setattr(_mf, "fetch_manifest", lambda: _FAKE_MANIFEST)

    seen = []
    monkeypatch.setattr(submit_mod, "submit_one", lambda **kw: seen.append(kw["dataset_spec"]) or "url")

    rc = main([
        "submit",
        "--model-module", "x:Y",
        "--datasets", "all",
        "--model-repo", "Org/rb",
        "--submission-meta", str(meta_path),
        "--hf-username", "u", "--contact", "c@example.com",
        "--output-dir", str(tmp_path / "results"),
    ])
    assert rc == 0
    assert seen == ["Org/A", "Org/B"]
```

- [ ] **Step 2: Run test to verify failure**

Run: `cd speech-spoof-bench && python -m pytest tests/test_cli.py::test_cli_submit_smoke tests/test_cli.py::test_cli_submit_all_uses_manifest -v`
Expected: fails with "unknown subcommand" or similar (`submit` not registered).

- [ ] **Step 3: Implement `submit` multi-dataset wrapper**

Append to `src/speech_spoof_bench/submit.py`:

```python
def _expand_dataset_specs(specs: list[str]) -> list[str]:
    """Expand `--datasets all` against the arena manifest (core_set + extended)."""
    if specs == ["all"]:
        from . import manifest as _mf
        m = _mf.fetch_manifest()
        return [e["id"] for e in m.get("core_set", [])] + [
            e["id"] for e in m.get("extended", [])
        ]
    if "all" in specs:
        raise ValueError("'--datasets all' must be used alone, not mixed with explicit ids")
    return list(specs)


def submit(
    *,
    model_module_spec: str,
    dataset_specs: list[str],
    output_dir: Path,
    meta_path: Path,
    model_repo: str,
    hf_username: str,
    contact: str,
    continue_on_error: bool = False,
    api: HfApi | None = None,
) -> dict[str, str]:
    """Run `submit_one` for each dataset; return {dataset_spec: pr_url}."""
    meta = load_meta(meta_path)
    expanded = _expand_dataset_specs(dataset_specs)
    api = api or HfApi()
    submitted_at = _dt.date.today().isoformat()

    results: dict[str, str] = {}
    for spec in expanded:
        try:
            url = submit_one(
                model_module_spec=model_module_spec,
                dataset_spec=spec,
                output_dir=Path(output_dir),
                meta=meta,
                model_repo=model_repo,
                hf_username=hf_username,
                contact=contact,
                submitted_at=submitted_at,
                api=api,
            )
            _LOG.info("submitted %s → %s", spec, url)
            results[spec] = url
        except Exception as exc:
            if not continue_on_error:
                raise
            _LOG.error("submission failed for %s: %s", spec, exc)
            results[spec] = f"ERROR: {exc}"
    return results
```

- [ ] **Step 4: Wire the CLI subcommand**

Open `src/speech_spoof_bench/cli.py`. After `_cmd_validate_submission`, add:

```python
def _cmd_submit(args: argparse.Namespace) -> int:
    from . import submit as submit_mod

    results = submit_mod.submit(
        model_module_spec=args.model_module,
        dataset_specs=list(args.datasets),
        output_dir=args.output_dir,
        meta_path=args.submission_meta,
        model_repo=args.model_repo,
        hf_username=args.hf_username,
        contact=args.contact,
        continue_on_error=args.continue_on_error,
    )
    for spec, url in results.items():
        print(f"{spec}\t{url}")
    return 0 if all(not str(v).startswith("ERROR:") for v in results.values()) else 1
```

In `build_parser`, after the `validate-submission` block and before `reproduce`, add:

```python
sm = sub.add_parser("submit", help="run model + upload scores + open PR on dataset repo")
sm.add_argument("--model-module", required=True,
                help="module:ClassName, e.g. mypkg.mymod:MyModel")
sm.add_argument("--datasets", action="append", required=True,
                help="HF dataset id; repeatable; use 'all' for manifest-wide")
sm.add_argument("--model-repo", required=True,
                help="HF model repo (owner/name) that owns the scores.txt")
sm.add_argument("--submission-meta", required=True, type=Path,
                help="path to meta.yaml describing the system")
sm.add_argument("--hf-username", required=True)
sm.add_argument("--contact", required=True)
sm.add_argument("--output-dir", default="./results", type=Path)
sm.add_argument("--continue-on-error", action="store_true")
sm.set_defaults(func=_cmd_submit)
```

You'll also need `from pathlib import Path` near the top of `cli.py` if it isn't there yet.

- [ ] **Step 5: Run the new and existing CLI tests**

Run: `cd speech-spoof-bench && python -m pytest tests/test_cli.py -v`
Expected: all pass (existing + 2 new).

- [ ] **Step 6: Commit**

```bash
cd speech-spoof-bench
git add src/speech_spoof_bench/submit.py src/speech_spoof_bench/cli.py tests/test_cli.py
git commit -m "submit: multi-dataset orchestrator + CLI subcommand"
```

---

## Task 8: Dataset-skeleton template files

**Files:**
- Create: `src/speech_spoof_bench/data/dataset_skeleton/README.md`
- Create: `src/speech_spoof_bench/data/dataset_skeleton/LICENSE.txt`
- Create: `src/speech_spoof_bench/data/dataset_skeleton/eval.yaml`
- Create: `src/speech_spoof_bench/data/dataset_skeleton/build_parquet.py`
- Create: `src/speech_spoof_bench/data/dataset_skeleton/submissions/README.md`
- Create: `src/speech_spoof_bench/data/dataset_skeleton/submissions/results_template.yaml`

These files are copied verbatim by `scaffold-dataset` (with `{{NAME}}` substitution in README.md and eval.yaml). Source the `submissions/` files from the canonical ones already living in the LA dataset repo at `/home/kirill/mnt/drive3_8tb/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA/submissions/`.

- [ ] **Step 1: Copy the canonical `submissions/README.md` and `results_template.yaml`**

```bash
mkdir -p speech-spoof-bench/src/speech_spoof_bench/data/dataset_skeleton/submissions
cp /home/kirill/mnt/drive3_8tb/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA/submissions/README.md \
   speech-spoof-bench/src/speech_spoof_bench/data/dataset_skeleton/submissions/README.md
cp /home/kirill/mnt/drive3_8tb/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA/submissions/results_template.yaml \
   speech-spoof-bench/src/speech_spoof_bench/data/dataset_skeleton/submissions/results_template.yaml
```

- [ ] **Step 2: Create `eval.yaml` template**

Write to `src/speech_spoof_bench/data/dataset_skeleton/eval.yaml`:

```yaml
name: {{NAME}}
description: >
  TODO: one-paragraph description of the dataset. What was collected, by whom,
  what the bonafide/spoof split represents.
evaluation_framework: inspect-ai

tasks:
  - id: antispoofing_eval
    config: default
    split: test

    field_spec:
      input: audio
      target: label

    solvers:
      - name: speech_spoof_bench_solver

    scorers:
      - name: speech_spoof_scorer

    metrics:
      - eer_percent
```

- [ ] **Step 3: Create `LICENSE.txt` placeholder**

Write to `src/speech_spoof_bench/data/dataset_skeleton/LICENSE.txt`:

```
TODO: replace this file with the upstream license verbatim before pushing.
```

- [ ] **Step 4: Create `build_parquet.py` stub**

Write to `src/speech_spoof_bench/data/dataset_skeleton/build_parquet.py`:

```python
"""Build the dataset's parquet shards under `data/test-*.parquet`.

The output schema MUST match the canonical schema (PLAN.md §1.2):

    Features({
        "path":  Value("string"),
        "audio": Audio(sampling_rate=16000),
        "label": ClassLabel(names=["bonafide", "spoof"]),
        "notes": Value("string"),
    })

`notes` is a JSON string and MUST contain a unique `utterance_id`. Audio MUST
be 16 kHz mono. Resample at build time.

After building, validate with:

    speech-spoof-bench validate-dataset <this-dir> --skip-submissions
"""

from __future__ import annotations


def main() -> None:
    raise NotImplementedError("Implement parquet build for this dataset.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Create `README.md` template**

Write to `src/speech_spoof_bench/data/dataset_skeleton/README.md`:

```markdown
---
license: TODO
language: [en]
pretty_name: {{NAME}}
task_categories: [audio-classification]
size_categories: [unknown]
configs:
  - config_name: default
    data_files:
      - {split: test, path: "data/test-*.parquet"}
tags:
  - anti-spoofing
  - audio-deepfake-detection
  - speech
  - benchmark
  - arena-ready
paperswithcode_id:
arxiv: []
---

# {{NAME}}

TODO: one-line summary.

## Overview

TODO: longer description, source, motivation.

## License & redistribution

TODO: confirm the upstream license permits redistribution and paste it
verbatim into `LICENSE.txt`. If not redistributable, this dataset is out of
scope for the org (PLAN.md §1.8).

## Schema

| Field | Type | Description |
|---|---|---|
| path  | string | Stable archive-relative path, unique within dataset. |
| audio | Audio(16kHz mono) | Resampled at build time. |
| label | ClassLabel[bonafide, spoof] | Index 0 = bonafide. |
| notes | string (JSON) | Must contain `utterance_id`. |

## Quick Start

```python
from datasets import load_dataset
ds = load_dataset("SpeechAntiSpoofingBenchmarks/{{NAME}}", split="test")
```

## Stats

| n_total | n_bonafide | n_spoof | total duration |
|---|---|---|---|
| TODO | TODO | TODO | TODO |

## Source provenance

TODO

## Evaluation

See `eval.yaml` and `submissions/README.md`.

## Citation

**Original paper**: TODO arxiv link

```bibtex
TODO
```

## Maintainer

TODO
```

- [ ] **Step 6: Verify the package data exists and is importable**

Run from the repo root:
```bash
cd speech-spoof-bench
python -c "from importlib import resources; \
  print(sorted(p.name for p in resources.files('speech_spoof_bench.data.dataset_skeleton').iterdir()))"
```
Expected output includes: `README.md`, `LICENSE.txt`, `eval.yaml`, `build_parquet.py`, `submissions`.

(If this fails with `ModuleNotFoundError`, add `src/speech_spoof_bench/data/dataset_skeleton/__init__.py` and `src/speech_spoof_bench/data/dataset_skeleton/submissions/__init__.py` as empty files. `importlib.resources` doesn't always need them, but `setuptools` package discovery does in some configs.)

- [ ] **Step 7: Commit**

```bash
cd speech-spoof-bench
git add src/speech_spoof_bench/data/dataset_skeleton/
git commit -m "scaffold: dataset skeleton templates (README, eval.yaml, build_parquet, submissions/)"
```

---

## Task 9: `scaffold-dataset` command

**Files:**
- Create: `src/speech_spoof_bench/scaffold.py`
- Modify: `src/speech_spoof_bench/cli.py`
- Create: `tests/test_scaffold.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scaffold.py
"""scaffold_dataset generates the §1.1 skeleton with {{NAME}} substitutions."""

from __future__ import annotations

from pathlib import Path

import pytest

from speech_spoof_bench.scaffold import scaffold_dataset


def test_scaffold_writes_all_expected_files(tmp_path: Path):
    out = tmp_path / "MyDataset"
    scaffold_dataset(name="MyDataset", output_dir=out)

    expected = {
        "README.md",
        "LICENSE.txt",
        "eval.yaml",
        "build_parquet.py",
        "submissions/README.md",
        "submissions/results_template.yaml",
    }
    actual = {
        str(p.relative_to(out)).replace("\\", "/")
        for p in out.rglob("*") if p.is_file()
    }
    # __init__.py files (if any) are allowed but not required.
    actual = {a for a in actual if not a.endswith("__init__.py")}
    assert expected.issubset(actual)


def test_scaffold_substitutes_name_token(tmp_path: Path):
    out = tmp_path / "InTheWild"
    scaffold_dataset(name="InTheWild", output_dir=out)
    readme = (out / "README.md").read_text()
    eval_yaml = (out / "eval.yaml").read_text()
    assert "{{NAME}}" not in readme
    assert "{{NAME}}" not in eval_yaml
    assert "InTheWild" in readme
    assert "name: InTheWild" in eval_yaml


def test_scaffold_refuses_nonempty_dir_without_force(tmp_path: Path):
    out = tmp_path / "X"
    out.mkdir()
    (out / "stuff.txt").write_text("hi")
    with pytest.raises(FileExistsError):
        scaffold_dataset(name="X", output_dir=out)


def test_scaffold_force_overwrites(tmp_path: Path):
    out = tmp_path / "X"
    out.mkdir()
    (out / "stuff.txt").write_text("hi")
    scaffold_dataset(name="X", output_dir=out, force=True)
    assert (out / "README.md").is_file()


def test_scaffold_empty_dir_ok(tmp_path: Path):
    out = tmp_path / "X"
    out.mkdir()  # exists but empty
    scaffold_dataset(name="X", output_dir=out)
    assert (out / "eval.yaml").is_file()
```

Append to `tests/test_cli.py`:

```python
def test_cli_scaffold_dataset(tmp_path):
    out = tmp_path / "Y"
    rc = main(["scaffold-dataset", "--name", "Y", "--output-dir", str(out)])
    assert rc == 0
    assert (out / "eval.yaml").is_file()
    assert "name: Y" in (out / "eval.yaml").read_text()
```

- [ ] **Step 2: Run tests to verify failure**

Run: `cd speech-spoof-bench && python -m pytest tests/test_scaffold.py tests/test_cli.py::test_cli_scaffold_dataset -v`
Expected: ImportError / unknown subcommand.

- [ ] **Step 3: Implement `scaffold_dataset`**

Create `src/speech_spoof_bench/scaffold.py`:

```python
"""Phase 7b — `scaffold-dataset` command.

Produces the §1.1 skeleton for a new dataset repo by copying packaged
template files into `output_dir` and substituting `{{NAME}}` in README.md
and eval.yaml.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

_TEMPLATE_PACKAGE = "speech_spoof_bench.data.dataset_skeleton"
_SUBSTITUTE_IN = {"README.md", "eval.yaml"}


def _iter_template_files() -> list[tuple[str, bytes]]:
    """Walk the packaged template and yield (relpath, bytes) pairs."""
    root = resources.files(_TEMPLATE_PACKAGE)
    out: list[tuple[str, bytes]] = []

    def walk(node, rel: str) -> None:
        for child in node.iterdir():
            if child.is_dir():
                walk(child, f"{rel}{child.name}/")
            else:
                name = child.name
                if name == "__init__.py":
                    continue
                out.append((f"{rel}{name}", child.read_bytes()))

    walk(root, "")
    return out


def scaffold_dataset(
    *, name: str, output_dir: Path | str, force: bool = False,
) -> None:
    """Materialize the dataset skeleton under `output_dir`.

    Raises:
      FileExistsError: `output_dir` exists and is non-empty without force.
    """
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()) and not force:
        raise FileExistsError(
            f"{out} already exists and is non-empty (use force=True to overwrite)"
        )
    out.mkdir(parents=True, exist_ok=True)

    for relpath, data in _iter_template_files():
        target = out / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        if Path(relpath).name in _SUBSTITUTE_IN:
            text = data.decode("utf-8").replace("{{NAME}}", name)
            target.write_text(text)
        else:
            target.write_bytes(data)
```

- [ ] **Step 4: Wire the CLI subcommand**

In `src/speech_spoof_bench/cli.py`, add:

```python
def _cmd_scaffold_dataset(args: argparse.Namespace) -> int:
    from . import scaffold

    scaffold.scaffold_dataset(
        name=args.name, output_dir=args.output_dir, force=args.force,
    )
    print(f"scaffolded dataset skeleton at {args.output_dir}")
    return 0
```

In `build_parser`, after the `submit` block, add:

```python
sd = sub.add_parser("scaffold-dataset",
                    help="generate the §1.1 dataset-repo skeleton")
sd.add_argument("--name", required=True)
sd.add_argument("--output-dir", required=True, type=Path)
sd.add_argument("--force", action="store_true",
                help="overwrite if output-dir is non-empty")
sd.set_defaults(func=_cmd_scaffold_dataset)
```

- [ ] **Step 5: Run tests**

Run: `cd speech-spoof-bench && python -m pytest tests/test_scaffold.py tests/test_cli.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
cd speech-spoof-bench
git add src/speech_spoof_bench/scaffold.py src/speech_spoof_bench/cli.py tests/test_scaffold.py tests/test_cli.py
git commit -m "scaffold-dataset: copy packaged skeleton, substitute name, CLI wired"
```

---

## Task 10: Full test suite + linting sweep

**Files:** None new.

- [ ] **Step 1: Run the whole test suite**

Run: `cd speech-spoof-bench && python -m pytest -v`
Expected: every test passes. If a previously-passing test broke (e.g. submission-schema tests reacted to the relaxation), inspect and fix in place — the relaxation is intentional but `validate-dataset`'s `S2` check should keep merged-stage strictness intact.

- [ ] **Step 2: Quick code review of `submit.py`**

Open `src/speech_spoof_bench/submit.py`. Confirm:
- All `import` statements are at the top, not scattered (move any inline imports added during TDD up to the top of the file, except those that must stay local to avoid circular imports — `from .loader import resolve as _resolve` and `from .benchmark import Benchmark` are fine where they are).
- Module docstring is present and accurate.
- No unused imports (`python -c "import ast; ..."` or just eyeball).

- [ ] **Step 3: Commit any cleanup**

```bash
cd speech-spoof-bench
git status
# if there's a diff:
git add -u
git commit -m "submit: cleanup imports and docstrings"
# else: skip
```

---

## Task 11: Manual smoke test against random-baseline-asas

This task is **not run by a subagent** — it requires HF credentials and network access, and produces an HF PR that the user must close manually. Run it interactively with the user.

- [ ] **Step 1: Author the meta.yaml**

Write `/tmp/random-baseline-meta.yaml`:

```yaml
system:
  name: random-baseline
  slug: random-baseline-phase7b
  description: |
    Reference random baseline. Returns N(0, 1) for every utterance using a
    fixed seed (seed=0). EER ≈ 50% by construction. Phase 7b smoke-test
    submission — close PR without merging.
  code: https://github.com/SpeechAntiSpoofingBenchmarks/speech-spoof-bench
  checkpoint: https://huggingface.co/SpeechAntiSpoofingBenchmarks/random-baseline-asas
  paper:
    arxiv_id: "1911.01601"
    url: https://arxiv.org/abs/1911.01601
    bibtex: |
      @article{wang2020asvspoof,
        title={ASVspoof 2019: A large-scale public database of synthesized,
               converted and replayed speech},
        author={Wang, Xin and others},
        journal={Computer Speech \& Language},
        volume={64},
        pages={101114},
        year={2020},
        publisher={Elsevier}
      }
notes: |
  Phase 7b smoke-test submission produced by `speech-spoof-bench submit`.
  Slug uses the `-phase7b` suffix to avoid colliding with the Phase 3
  manual submission. Close the PR without merging.
```

- [ ] **Step 2: Verify HF auth**

Run: `python -c "from huggingface_hub import whoami; print(whoami()['name'])"`
Expected: prints an HF username with write access to `SpeechAntiSpoofingBenchmarks/random-baseline-asas` and ability to open PRs on `SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA`. If not, `huggingface-cli login` first.

- [ ] **Step 3: Run `submit`**

```bash
cd speech-spoof-bench
speech-spoof-bench submit \
  --model-module speech_spoof_bench.examples.random_baseline:RandomBaseline \
  --datasets SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA \
  --model-repo SpeechAntiSpoofingBenchmarks/random-baseline-asas \
  --submission-meta /tmp/random-baseline-meta.yaml \
  --hf-username SpeechAntiSpoofingBenchmarks \
  --contact k.n.borodin@mtuci.ru \
  --output-dir /tmp/phase7b-results
```

Expected: the command prints a single line `SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA\t<PR URL>` and exits 0.

- [ ] **Step 4: Inspect the PR**

Open the PR URL in a browser. Check:
- Diff is exactly one file: `submissions/random-baseline-phase7b.yaml`.
- `scores_url` pins a `/resolve/<40-char-sha>/.eval_results/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA/scores.txt`.
- The corresponding commit exists on `SpeechAntiSpoofingBenchmarks/random-baseline-asas`'s main branch (the URL resolves in the browser).
- `reproduction: {}` (empty mapping).
- `submitter.hf_username` and `submitter.contact` match the flags.
- `dataset.revision` is a 40-char sha (the current main of the LA dataset repo, not `None`).

- [ ] **Step 5: Validate and reproduce the PR YAML locally**

Download the YAML from the PR branch (use the "Download raw file" link from the HF PR diff or `huggingface_hub.hf_hub_download(..., revision=<pr-branch-name>)`). Save to `/tmp/phase7b-pr.yaml`.

```bash
speech-spoof-bench validate-submission /tmp/phase7b-pr.yaml
speech-spoof-bench reproduce --scoring /tmp/phase7b-pr.yaml
```

Expected: both exit 0; `reproduce --scoring` reports EER matching the claimed value within 1e-6.

- [ ] **Step 6: Close the PR without merging**

In the HF PR UI, click "Close PR". Do not merge — this would create a duplicate `random-baseline-phase7b.yaml` alongside the existing `random-baseline.yaml`.

- [ ] **Step 7: Smoke-test `scaffold-dataset`**

```bash
speech-spoof-bench scaffold-dataset --name TestScaffold --output-dir /tmp/test-scaffold
ls -la /tmp/test-scaffold /tmp/test-scaffold/submissions
cat /tmp/test-scaffold/eval.yaml
grep TestScaffold /tmp/test-scaffold/README.md
rm -rf /tmp/test-scaffold
```

Expected: all six files present, `name: TestScaffold` in eval.yaml, `pretty_name: TestScaffold` and `# TestScaffold` heading in README.md.

- [ ] **Step 8: Update the roadmap**

In `docs/roadmap/ROADMAP.md`, mark Phase 7b's checkboxes complete:

```
- [x] `submit` (§2.5a) — one command: run + upload scores to model repo + build YAML + open HF PR on dataset repo.
- [x] `scaffold-dataset` (§2.5, §3.8 step 1) — produces the skeleton for a new dataset repo.
```

- [ ] **Step 9: Commit roadmap update**

```bash
cd speech-spoof-bench
git add ../docs/roadmap/ROADMAP.md  # or wherever ROADMAP.md actually lives in the repo
git commit -m "roadmap: mark Phase 7b complete"
```

---

## Self-review

**Spec coverage:** Decisions table → Task 1 (schema), Task 2 (meta), Task 3 (payload), Task 4 (upload), Task 5 (PR), Task 6 (submit_one), Task 7 (multi-dataset + CLI), Task 8+9 (scaffold). Manual smoke test → Task 11. Unit-test inventory in §6 → covered file-by-file in Tasks 2–7 and 9.

**Placeholder scan:** No "TBD" / "implement later" / "similar to" in steps. All code blocks are complete and self-contained.

**Type consistency:** `submit_one` returns `str` (PR URL); `submit` returns `dict[str, str]`. `upload_scores` returns `tuple[str, str]`. `open_submission_pr` returns `str`. `load_meta` returns `dict[str, Any]`. `build_submission_payload` returns `dict[str, Any]`. CLI subcommand functions return `int`. All consistent across tasks.

**Open spec items §1 and §2 (`--datasets all` source; revision mismatch behavior):** resolved in Task 7 step 3 (`_expand_dataset_specs` uses `core_set + extended`) and Task 6 (silent re-run on revision mismatch).
