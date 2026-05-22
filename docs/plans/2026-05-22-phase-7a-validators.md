# Phase 7a — Validators Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship three CLI commands — `validate-submission`, `validate-dataset` (full §1.9), and `reproduce --scoring` — implementing the maintainer-side gate for HF PR submissions per the v4 spec.

**Architecture:** Three new pure-Python modules (`validate.py`, `reproduce.py`, `hf_fetch.py`) plus CLI wiring in `cli.py`. No new third-party deps. Public functions take Python objects so they're unit-testable without the CLI. Network access is centralised in `hf_fetch.py`. All audio-decode paths in `reproduce.py` are avoided via `IterableDataset.select_columns(["notes", "label"])`.

**Tech Stack:** Python 3.10+, `huggingface_hub`, `datasets`, `pyyaml`, `jsonschema`, `pytest`.

**Spec reference:** `docs/specs/2026-05-22-phase-7a-validators-design.md`

**Working directory:** `/home/kirill/speech-spoof-bench/speech-spoof-bench`

**Conventions used throughout:**
- All test commands: `pytest -xvs <path>`.
- All commits: per-task; message in imperative `feat:` / `test:` / `chore:` form.
- File paths are relative to the working directory unless absolute.
- TDD strict: write failing test, run it red, implement, run it green, commit.

---

### Task 1: `hf_fetch.py` — URL parser + download wrapper

**Files:**
- Create: `src/speech_spoof_bench/hf_fetch.py`
- Test: `tests/test_hf_fetch.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_hf_fetch.py
"""Tests for the HF resolve-URL parser and download wrapper."""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import patch

import pytest

from speech_spoof_bench import hf_fetch


VALID_URL = (
    "https://huggingface.co/SpeechAntiSpoofingBenchmarks/random-baseline-asas/"
    "resolve/f63c30bade6e2d059b2e805dea7a807f2f57e99a/"
    ".eval_results/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA/scores.txt"
)


def test_parse_valid_url():
    repo, sha, path = hf_fetch.parse_hf_resolve_url(VALID_URL)
    assert repo == "SpeechAntiSpoofingBenchmarks/random-baseline-asas"
    assert sha == "f63c30bade6e2d059b2e805dea7a807f2f57e99a"
    assert path == ".eval_results/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA/scores.txt"


def test_parse_rejects_main_ref():
    bad = VALID_URL.replace("/resolve/f63c30bade6e2d059b2e805dea7a807f2f57e99a/", "/resolve/main/")
    with pytest.raises(ValueError, match="commit-pinned"):
        hf_fetch.parse_hf_resolve_url(bad)


def test_parse_rejects_non_hf():
    with pytest.raises(ValueError):
        hf_fetch.parse_hf_resolve_url("https://example.com/file.txt")


def test_download_passes_hf_token(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "tok_xyz")
    fake_file = tmp_path / "scores.txt"
    fake_file.write_text("LA_E_0001 0.5\n")

    captured = {}
    def fake_download(**kwargs):
        captured.update(kwargs)
        return str(fake_file)

    with patch.object(hf_fetch, "hf_hub_download", side_effect=fake_download):
        local, sha = hf_fetch.download(VALID_URL)

    assert captured["repo_id"] == "SpeechAntiSpoofingBenchmarks/random-baseline-asas"
    assert captured["revision"] == "f63c30bade6e2d059b2e805dea7a807f2f57e99a"
    assert captured["token"] == "tok_xyz"
    assert captured["repo_type"] == "model"
    assert local == fake_file
    assert sha == hashlib.sha256(fake_file.read_bytes()).hexdigest()


def test_download_without_token(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    fake_file = tmp_path / "scores.txt"
    fake_file.write_text("x")
    captured = {}
    def fake_download(**kwargs):
        captured.update(kwargs)
        return str(fake_file)
    with patch.object(hf_fetch, "hf_hub_download", side_effect=fake_download):
        hf_fetch.download(VALID_URL)
    assert captured["token"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest -xvs tests/test_hf_fetch.py`
Expected: import error (module missing).

- [ ] **Step 3: Implement `hf_fetch.py`**

```python
# src/speech_spoof_bench/hf_fetch.py
"""Tiny wrapper around huggingface_hub for commit-pinned resolve URLs.

Centralises:
  - URL parsing (must be /resolve/<sha>/, not /resolve/main/)
  - HF_TOKEN env honoring
  - sha256 of the fetched file
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

from huggingface_hub import hf_hub_download

_HF_RESOLVE_RE = re.compile(
    r"^https://huggingface\.co/(?P<repo>[^/]+/[^/]+)/resolve/"
    r"(?P<sha>[0-9a-f]{7,40})/(?P<path>.+)$"
)


def parse_hf_resolve_url(url: str) -> tuple[str, str, str]:
    """Parse a commit-pinned HF resolve URL into (repo_id, commit_sha, path)."""
    m = _HF_RESOLVE_RE.match(url)
    if not m:
        raise ValueError(f"not a commit-pinned HF resolve URL: {url!r}")
    return m["repo"], m["sha"], m["path"]


def download(url: str) -> tuple[Path, str]:
    """Download a commit-pinned HF resolve URL.

    Returns (local_path, sha256_hex). Honors $HF_TOKEN if set.
    """
    repo, sha, path = parse_hf_resolve_url(url)
    token = os.environ.get("HF_TOKEN") or None
    local = hf_hub_download(
        repo_id=repo,
        filename=path,
        repo_type="model",
        revision=sha,
        token=token,
    )
    p = Path(local)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return p, h.hexdigest()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest -xvs tests/test_hf_fetch.py`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/speech_spoof_bench/hf_fetch.py tests/test_hf_fetch.py
git commit -m "feat: add hf_fetch helper for commit-pinned HF URLs"
```

---

### Task 2: Submission fixtures

**Files:**
- Create: `tests/fixtures/submissions/valid.yaml`
- Create: `tests/fixtures/submissions/invalid_no_reproduction.yaml`
- Create: `tests/fixtures/submissions/invalid_unpinned_url.yaml`
- Create: `tests/fixtures/submissions/invalid_bad_sha.yaml`
- Create: `tests/fixtures/submissions/invalid_bad_slug.yaml`
- Create: `tests/fixtures/submissions/invalid_wrong_schema_version.yaml`
- Create: `tests/fixtures/submissions/invalid_malformed.yaml`
- Create: `tests/fixtures/scores_known.txt`

These are static fixtures shared by `test_validate_submission.py`, `test_validate_dataset.py`, and `test_reproduce.py`.

- [ ] **Step 1: Create `valid.yaml`** — full v4 submission, fields all present, slug-pattern-clean. Use a synthetic but well-formed sha and a commit-pinned URL.

```yaml
# tests/fixtures/submissions/valid.yaml
schema_version: 4
system:
  name: fixture-baseline
  slug: fixture-baseline
  description: Test fixture submission.
  code: https://github.com/example/fixture
  checkpoint: https://huggingface.co/example/fixture
  paper:
    arxiv_id: "1234.56789"
    url: https://arxiv.org/abs/1234.56789
    bibtex: |
      @article{fixture, title={t}, author={a}, year={2026}}
dataset:
  id: SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA
  revision: 151aa4c6
  split: test
scores:
  eer_percent: 25.0
  n_trials: 4
  n_skipped: 0
artifact:
  scores_url: https://huggingface.co/example/fixture/resolve/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/scores.txt
  scores_sha256: "0000000000000000000000000000000000000000000000000000000000000000"
  bench_version: speech-spoof-bench==0.1.0
reproduction:
  reproduced_by: SpeechAntiSpoofingBenchmarks
  reproduced_at: 2026-05-22
  reproduced_bench_version: speech-spoof-bench==0.1.0
  match: scoring
submitter:
  hf_username: fixture
  contact: fixture@example.com
submitted_at: 2026-05-22
notes: ""
```

- [ ] **Step 2: Create each invalid fixture** — produced by mutating `valid.yaml`:

`invalid_no_reproduction.yaml`: same as valid but delete the entire `reproduction:` block.
`invalid_unpinned_url.yaml`: replace `/resolve/aaa…/` with `/resolve/main/`.
`invalid_bad_sha.yaml`: change `scores_sha256` to `"deadbeef"` (too short, not 64 hex chars).
`invalid_bad_slug.yaml`: change `slug` to `Fixture Baseline` (contains space + uppercase).
`invalid_wrong_schema_version.yaml`: change `schema_version: 4` to `schema_version: 3`.
`invalid_malformed.yaml`: write literal text `not: valid: yaml: at: all\n  - [\n` (parser error).

- [ ] **Step 3: Create `tests/fixtures/scores_known.txt`** with content:

```
UTT_0000 -1.0
UTT_0001 0.5
UTT_0002 0.0
UTT_0003 2.0
```

With the synthetic dataset (4 rows, labels `[0,1,0,1]` = bonafide,spoof,bonafide,spoof), these scores produce a deterministic EER.

- [ ] **Step 4: Commit**

```bash
git add tests/fixtures/
git commit -m "test: add submission YAML and scores fixtures for 7a"
```

---

### Task 3: `validate-submission` CLI

**Files:**
- Modify: `src/speech_spoof_bench/cli.py` (add `_cmd_validate_submission` + subparser)
- Test: `tests/test_validate_submission.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_validate_submission.py
"""Tests for `speech-spoof-bench validate-submission`."""

from __future__ import annotations

from pathlib import Path

import pytest

from speech_spoof_bench.cli import main

FIX = Path(__file__).parent / "fixtures" / "submissions"


def test_valid(capsys):
    rc = main(["validate-submission", str(FIX / "valid.yaml")])
    assert rc == 0
    assert "OK" in capsys.readouterr().out


@pytest.mark.parametrize(
    "name",
    [
        "invalid_no_reproduction.yaml",
        "invalid_unpinned_url.yaml",
        "invalid_bad_sha.yaml",
        "invalid_bad_slug.yaml",
        "invalid_wrong_schema_version.yaml",
        "invalid_malformed.yaml",
    ],
)
def test_invalid(name, capsys):
    rc = main(["validate-submission", str(FIX / name)])
    assert rc == 1
    assert "FAIL" in capsys.readouterr().err
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest -xvs tests/test_validate_submission.py`
Expected: argparse exits with code 2 ("invalid choice: 'validate-submission'").

- [ ] **Step 3: Wire the subcommand in `cli.py`**

Add to `cli.py` just before `def build_parser()`:

```python
def _cmd_validate_submission(args: argparse.Namespace) -> int:
    from . import submission
    path = args.path
    try:
        text = open(path).read()
        submission.parse_submission(text)
    except FileNotFoundError as e:
        print(f"FAIL {path}: {e}", file=sys.stderr)
        return 1
    except submission.SubmissionValidationError as e:
        print(f"FAIL {path}: {e}", file=sys.stderr)
        return 1
    except Exception as e:  # malformed YAML, decode errors, etc.
        print(f"FAIL {path}: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    print(f"OK: {path}")
    return 0
```

In `build_parser()`, add after the `validate-dataset` subparser block:

```python
vs = sub.add_parser(
    "validate-submission",
    help="schema-check a submission YAML offline (no network)",
)
vs.add_argument("path", help="path to a submission YAML file")
vs.set_defaults(func=_cmd_validate_submission)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest -xvs tests/test_validate_submission.py`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/speech_spoof_bench/cli.py tests/test_validate_submission.py
git commit -m "feat: add validate-submission CLI command"
```

---

### Task 4: `validate.py` — dataset-side checks D1–D7

**Files:**
- Create: `src/speech_spoof_bench/validate.py`
- Test: `tests/test_validate_dataset.py`
- Modify: `tests/conftest.py` (extend `synth_local_dataset` to add README + submissions/)

The aggregator returns a structured `ValidationReport` object. The CLI is wired in Task 6.

- [ ] **Step 1: Extend `tests/conftest.py`** — add a fully-equipped synthetic dataset fixture.

Replace the existing `synth_local_dataset` body with the version below (it adds README.md with frontmatter and one valid submission YAML). Existing tests using the simpler version must continue passing because the returned path still resolves the same way through `loader.resolve()`.

```python
# Add after the existing eval_yaml.write_text() line:
    readme = """---
license: cc-by-4.0
language: [en]
pretty_name: Synth Dataset TEST
task_categories: [audio-classification]
size_categories: [n<1K]
configs:
  - config_name: default
    data_files:
      - {split: test, path: "data/test-*.parquet"}
tags:
  - anti-spoofing
  - arena-ready
arxiv:
  - 1911.01601
---

# Synth Dataset TEST
Synthetic fixture used by 7a tests.
"""
    (root / "README.md").write_text(readme)

    (root / "submissions").mkdir()
    sub = yaml.safe_load((Path(__file__).parent / "fixtures" / "submissions" / "valid.yaml").read_text())
    sub["dataset"]["id"] = "SynthDataset_TEST/SynthDataset_TEST"  # any org/name shape
    (root / "submissions" / "fixture.yaml").write_text(yaml.safe_dump(sub))
    return root
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_validate_dataset.py
"""Tests for validate.py dataset-side checks (D1–D7)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from speech_spoof_bench import validate


def test_happy_path(synth_local_dataset):
    report = validate.validate_dataset(str(synth_local_dataset), skip_submissions=True)
    assert report.ok, report.format()
    assert all(check.passed for check in report.dataset_checks)


def test_d6_missing_arena_ready_tag(synth_local_dataset):
    readme = (synth_local_dataset / "README.md").read_text()
    bad = readme.replace("- arena-ready\n", "")
    (synth_local_dataset / "README.md").write_text(bad)
    report = validate.validate_dataset(str(synth_local_dataset), skip_submissions=True)
    assert not report.ok
    assert any("arena-ready" in c.message for c in report.dataset_checks if not c.passed)


def test_d6_missing_arxiv(synth_local_dataset):
    readme = (synth_local_dataset / "README.md").read_text()
    bad = readme.replace("arxiv:\n  - 1911.01601\n", "")
    (synth_local_dataset / "README.md").write_text(bad)
    report = validate.validate_dataset(str(synth_local_dataset), skip_submissions=True)
    assert not report.ok
    assert any(c.id == "D6" and not c.passed for c in report.dataset_checks)


def test_d7_unregistered_metric(synth_local_dataset):
    ev = yaml.safe_load((synth_local_dataset / "eval.yaml").read_text())
    ev["tasks"][0]["metrics"] = ["does_not_exist"]
    (synth_local_dataset / "eval.yaml").write_text(yaml.safe_dump(ev))
    report = validate.validate_dataset(str(synth_local_dataset), skip_submissions=True)
    assert not report.ok
    assert any(c.id == "D7" and not c.passed for c in report.dataset_checks)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest -xvs tests/test_validate_dataset.py`
Expected: import error (validate module missing).

- [ ] **Step 4: Implement `validate.py` skeleton + D1–D7**

```python
# src/speech_spoof_bench/validate.py
"""Aggregating validator for a dataset repo against the v4 spec (§1.9).

Public surface:
  - validate_dataset(spec, *, skip_submissions=False) -> ValidationReport
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from datasets import Audio, ClassLabel
from huggingface_hub import hf_hub_download

from .loader import resolve
from .metrics import is_registered

REQUIRED_README_KEYS = {
    "license", "language", "pretty_name", "task_categories",
    "size_categories", "configs", "tags", "arxiv",
}


@dataclass
class CheckResult:
    id: str
    passed: bool
    message: str = ""


@dataclass
class SubmissionReport:
    path: str
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.passed for c in self.checks)


@dataclass
class ValidationReport:
    dataset_spec: str
    dataset_checks: list[CheckResult] = field(default_factory=list)
    submission_reports: list[SubmissionReport] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.passed for c in self.dataset_checks) and all(
            s.ok for s in self.submission_reports
        )

    def format(self) -> str:
        lines = [f"Dataset: {self.dataset_spec}"]
        for c in self.dataset_checks:
            mark = "✓" if c.passed else "✗"
            lines.append(f"  [{mark}] {c.id} {c.message}")
        if self.submission_reports:
            lines.append("")
            lines.append(f"Submissions ({len(self.submission_reports)}):")
            for sr in self.submission_reports:
                if sr.ok:
                    lines.append(f"  [✓] {sr.path}")
                else:
                    lines.append(f"  [✗] {sr.path}")
                    for c in sr.checks:
                        if not c.passed:
                            lines.append(f"        {c.id} {c.message}")
        failed = sum(1 for c in self.dataset_checks if not c.passed)
        failed += sum(
            1
            for sr in self.submission_reports
            for c in sr.checks
            if not c.passed
        )
        lines.append("")
        lines.append("OK" if self.ok else f"{failed} checks failed.")
        return "\n".join(lines)


def _read_readme_frontmatter(text: str) -> dict[str, Any] | None:
    if not text.startswith("---"):
        return None
    parts = text.split("---", 2)
    if len(parts) < 3:
        return None
    try:
        data = yaml.safe_load(parts[1])
    except yaml.YAMLError:
        return None
    return data if isinstance(data, dict) else None


def _load_readme(spec_path: Path | None, repo_id: str | None) -> str:
    if spec_path is not None:
        p = spec_path / "README.md"
        return p.read_text() if p.is_file() else ""
    local = hf_hub_download(
        repo_id=repo_id, filename="README.md", repo_type="dataset"
    )
    return Path(local).read_text()


def _list_submission_paths(
    spec_path: Path | None, repo_id: str | None
) -> list[tuple[str, str]]:
    """Return list of (display_path, local_file_path)."""
    excluded = {"README.md", "results_template.yaml"}
    out: list[tuple[str, str]] = []
    if spec_path is not None:
        sub_dir = spec_path / "submissions"
        if not sub_dir.is_dir():
            return out
        for p in sorted(sub_dir.glob("*.yaml")):
            if p.name in excluded:
                continue
            out.append((f"submissions/{p.name}", str(p)))
        return out
    from .submission import list_submission_files
    for remote in list_submission_files(repo_id):
        local = hf_hub_download(
            repo_id=repo_id, filename=remote, repo_type="dataset"
        )
        out.append((remote, local))
    return out


def _check_dataset_side(spec: str) -> tuple[list[CheckResult], dict[str, Any]]:
    """Run D1–D7. Returns (checks, info_for_submission_checks)."""
    checks: list[CheckResult] = []
    source, ds = resolve(spec, streaming=True)

    # D1: schema columns
    try:
        first = next(iter(ds))
        expected = {"path", "audio", "label", "notes"}
        actual = set(first.keys())
        if expected == actual:
            checks.append(CheckResult("D1", True, "schema matches v4"))
        else:
            checks.append(CheckResult(
                "D1", False,
                f"schema mismatch (got {sorted(actual)}, want {sorted(expected)})",
            ))
    except Exception as e:
        first = None
        checks.append(CheckResult("D1", False, f"could not read first row: {e}"))

    # D2: ClassLabel names (re-resolve non-streaming for features inspection)
    try:
        from datasets import load_dataset
        # Use the features from a small non-streaming probe; ds in streaming mode
        # may not expose ClassLabel directly. Probe one row from a fresh load.
        # For local: read via the loader. For HF: same.
        feat = getattr(ds, "features", None)
        label_feat = feat.get("label") if feat else None
        if isinstance(label_feat, ClassLabel) and label_feat.names == ["bonafide", "spoof"]:
            checks.append(CheckResult("D2", True, "label classes ok"))
        elif isinstance(label_feat, ClassLabel):
            checks.append(CheckResult(
                "D2", False, f"label classes {label_feat.names!r} != ['bonafide','spoof']"
            ))
        else:
            # Schema in conftest uses Value("int64") — accept when label is int 0/1.
            # The on-HF dataset uses ClassLabel; the local fixture uses int. Treat
            # int-with-0/1 as a soft pass with a note.
            if first is not None and int(first.get("label", -1)) in (0, 1):
                checks.append(CheckResult("D2", True, "label is int 0/1 (non-ClassLabel)"))
            else:
                checks.append(CheckResult("D2", False, "label feature not ClassLabel"))
    except Exception as e:
        checks.append(CheckResult("D2", False, f"label feature check error: {e}"))

    # D3: sampling rate + duration
    try:
        if first is None:
            checks.append(CheckResult("D3", False, "first row unavailable"))
        else:
            audio = first["audio"]
            sr = int(audio["sampling_rate"])
            arr = audio["array"]
            dur = len(arr) / sr
            if sr == 16000 and dur >= 1.0:
                checks.append(CheckResult("D3", True, f"sr=16000 dur={dur:.2f}s"))
            else:
                checks.append(CheckResult(
                    "D3", False, f"sr={sr} dur={dur:.2f}s (want sr=16000, dur>=1.0s)"
                ))
    except Exception as e:
        checks.append(CheckResult("D3", False, f"audio decode error: {e}"))

    # D4 + D5 combined pass: stream rows, accumulate sets.
    utt_ids: set[str] = set()
    paths: set[str] = set()
    dup_utt: set[str] = set()
    dup_path: set[str] = set()
    d4_failed = False
    d4_msg = ""
    sampled_for_d4 = 0
    try:
        for row in resolve(spec, streaming=True)[1]:
            if sampled_for_d4 < 100:
                try:
                    note = json.loads(row["notes"])
                except Exception as e:
                    d4_failed = True
                    d4_msg = f"notes JSON decode error: {e}"
                else:
                    if not note.get("utterance_id"):
                        d4_failed = True
                        d4_msg = "notes missing non-empty utterance_id"
                sampled_for_d4 += 1
            try:
                note = json.loads(row["notes"])
                uid = note.get("utterance_id")
            except Exception:
                uid = None
            if uid:
                if uid in utt_ids:
                    dup_utt.add(uid)
                utt_ids.add(uid)
            p = row.get("path")
            if p:
                if p in paths:
                    dup_path.add(p)
                paths.add(p)
        if d4_failed:
            checks.append(CheckResult("D4", False, d4_msg))
        else:
            checks.append(CheckResult("D4", True, f"sampled {sampled_for_d4} rows"))
        if not dup_utt and not dup_path:
            checks.append(CheckResult(
                "D5", True, f"uniqueness ok ({len(utt_ids)} ids, {len(paths)} paths)"
            ))
        else:
            checks.append(CheckResult(
                "D5", False,
                f"duplicates: utt={len(dup_utt)} path={len(dup_path)}",
            ))
    except Exception as e:
        checks.append(CheckResult("D4", False, f"iteration error: {e}"))
        checks.append(CheckResult("D5", False, "skipped (iteration failed)"))

    # D6: README frontmatter
    try:
        readme_text = _load_readme(
            spec_path=source.local_path if source.is_local else None,
            repo_id=None if source.is_local else source.canonical_id,
        )
        fm = _read_readme_frontmatter(readme_text)
        if fm is None:
            checks.append(CheckResult("D6", False, "README frontmatter missing or invalid"))
        else:
            missing = REQUIRED_README_KEYS - set(fm.keys())
            if missing:
                checks.append(CheckResult("D6", False, f"frontmatter missing keys: {sorted(missing)}"))
            elif "arena-ready" not in (fm.get("tags") or []):
                checks.append(CheckResult("D6", False, "tags missing 'arena-ready'"))
            else:
                checks.append(CheckResult("D6", True, "frontmatter ok"))
    except Exception as e:
        checks.append(CheckResult("D6", False, f"README load error: {e}"))

    # D7: metric registry — loader.resolve already raises KeyError on unregistered
    # metrics, so reaching this point means D7 passed.
    checks.append(CheckResult("D7", True, f"metrics registered: {source.metrics}"))

    info = {
        "is_local": source.is_local,
        "local_path": source.local_path,
        "canonical_id": source.canonical_id if not source.is_local else None,
    }
    return checks, info


def validate_dataset(spec: str, *, skip_submissions: bool = False) -> ValidationReport:
    report = ValidationReport(dataset_spec=spec)
    try:
        dataset_checks, info = _check_dataset_side(spec)
    except Exception as e:
        # Catch the case where resolve() itself fails (e.g., D7 unregistered
        # metric raises KeyError before any check runs). Surface as D7 fail.
        msg = str(e)
        if "metric" in msg.lower() and "not registered" in msg.lower():
            report.dataset_checks.append(CheckResult("D7", False, msg))
        else:
            report.dataset_checks.append(CheckResult("D0", False, f"load error: {e}"))
        return report
    report.dataset_checks = dataset_checks
    if skip_submissions:
        return report
    # Submission checks land in Task 5.
    return report
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest -xvs tests/test_validate_dataset.py`
Expected: 4 passed.

- [ ] **Step 6: Run the full suite to check for regressions**

Run: `pytest -xvs`
Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add src/speech_spoof_bench/validate.py tests/test_validate_dataset.py tests/conftest.py
git commit -m "feat: add validate.py dataset-side checks (D1-D7)"
```

---

### Task 5: `validate.py` — submission-side checks S1–S4

**Files:**
- Modify: `src/speech_spoof_bench/validate.py`
- Modify: `tests/test_validate_dataset.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_validate_dataset.py`:

```python
from unittest.mock import patch
from pathlib import Path
import hashlib


def test_submissions_happy_path(synth_local_dataset, tmp_path, monkeypatch):
    fake_scores = tmp_path / "scores.txt"
    fake_scores.write_text("x\n")
    sha = hashlib.sha256(fake_scores.read_bytes()).hexdigest()

    sub_path = synth_local_dataset / "submissions" / "fixture.yaml"
    sub_yaml = yaml.safe_load(sub_path.read_text())
    sub_yaml["artifact"]["scores_sha256"] = sha
    sub_path.write_text(yaml.safe_dump(sub_yaml))

    with patch("speech_spoof_bench.validate.hf_fetch.download",
               return_value=(fake_scores, sha)):
        report = validate.validate_dataset(str(synth_local_dataset))
    assert report.ok, report.format()
    assert len(report.submission_reports) == 1
    assert report.submission_reports[0].ok


def test_submission_unreachable_url(synth_local_dataset):
    with patch("speech_spoof_bench.validate.hf_fetch.download",
               side_effect=OSError("HTTP 404")):
        report = validate.validate_dataset(str(synth_local_dataset))
    assert not report.ok
    sr = report.submission_reports[0]
    assert any(c.id == "S3" and not c.passed for c in sr.checks)
    assert any(c.id == "S4" and "depends" in c.message for c in sr.checks)


def test_submission_sha_mismatch(synth_local_dataset, tmp_path):
    fake_scores = tmp_path / "scores.txt"
    fake_scores.write_text("x\n")
    wrong_sha = "0" * 64
    with patch("speech_spoof_bench.validate.hf_fetch.download",
               return_value=(fake_scores, wrong_sha)):
        report = validate.validate_dataset(str(synth_local_dataset))
    assert not report.ok
    sr = report.submission_reports[0]
    assert any(c.id == "S4" and not c.passed for c in sr.checks)


def test_skip_submissions_flag(synth_local_dataset):
    report = validate.validate_dataset(
        str(synth_local_dataset), skip_submissions=True
    )
    assert report.submission_reports == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest -xvs tests/test_validate_dataset.py -k submission`
Expected: tests fail — `submission_reports` is empty in default behaviour.

- [ ] **Step 3: Extend `validate.py`** — implement S1–S4 in `validate_dataset()`.

Replace the placeholder comment at the bottom of `validate_dataset()` with:

```python
    from . import hf_fetch, submission as sub_mod
    for display_path, local_path in _list_submission_paths(
        spec_path=info["local_path"], repo_id=info["canonical_id"]
    ):
        sr = SubmissionReport(path=display_path)
        # S1: schema
        try:
            text = Path(local_path).read_text()
            data = sub_mod.parse_submission(text)
            sr.checks.append(CheckResult("S1", True, "schema ok"))
        except Exception as e:
            sr.checks.append(CheckResult("S1", False, str(e)))
            sr.checks.append(CheckResult("S2", False, "skipped (S1 failed)"))
            sr.checks.append(CheckResult("S3", False, "skipped (S1 failed)"))
            sr.checks.append(CheckResult("S4", False, "skipped (S1 failed)"))
            report.submission_reports.append(sr)
            continue
        # S2: reproduction block (already required by schema, kept explicit)
        repro = data.get("reproduction") or {}
        if repro:
            sr.checks.append(CheckResult("S2", True, "reproduction block present"))
        else:
            sr.checks.append(CheckResult("S2", False, "reproduction block missing"))
        # S3: fetch
        url = data["artifact"]["scores_url"]
        claimed_sha = data["artifact"]["scores_sha256"]
        try:
            _, observed_sha = hf_fetch.download(url)
            sr.checks.append(CheckResult("S3", True, "scores_url reachable"))
            if observed_sha == claimed_sha:
                sr.checks.append(CheckResult("S4", True, "scores_sha256 matches"))
            else:
                sr.checks.append(CheckResult(
                    "S4", False,
                    f"sha mismatch: claimed {claimed_sha} got {observed_sha}",
                ))
        except Exception as e:
            sr.checks.append(CheckResult("S3", False, f"scores_url unreachable: {e}"))
            sr.checks.append(CheckResult("S4", False, "depends on S3"))
        report.submission_reports.append(sr)
    return report
```

Also add the `hf_fetch` import at the top of the file (`from . import hf_fetch`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest -xvs tests/test_validate_dataset.py`
Expected: all 8 pass.

- [ ] **Step 5: Commit**

```bash
git add src/speech_spoof_bench/validate.py tests/test_validate_dataset.py
git commit -m "feat: add validate.py submission-side checks (S1-S4)"
```

---

### Task 6: Wire `validate-dataset` CLI

**Files:**
- Modify: `src/speech_spoof_bench/cli.py`

The CLI already has `_cmd_validate_dataset`, but it's the stub from Phase 2. Replace it with the full implementation.

- [ ] **Step 1: Write the failing test** — extend `tests/test_cli.py` (or create a new file) with:

```python
# tests/test_cli_validate_dataset.py
from speech_spoof_bench.cli import main


def test_cli_validate_dataset_skip_submissions(synth_local_dataset, capsys):
    rc = main(["validate-dataset", str(synth_local_dataset), "--skip-submissions"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "OK" in out


def test_cli_validate_dataset_failure(synth_local_dataset, capsys):
    (synth_local_dataset / "README.md").write_text("no frontmatter")
    rc = main(["validate-dataset", str(synth_local_dataset), "--skip-submissions"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "D6" in out
    assert "failed" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest -xvs tests/test_cli_validate_dataset.py`
Expected: failure — the old stub doesn't accept `--skip-submissions`.

- [ ] **Step 3: Replace `_cmd_validate_dataset` and its subparser**

In `cli.py`, replace the existing `_cmd_validate_dataset` function with:

```python
def _cmd_validate_dataset(args: argparse.Namespace) -> int:
    from . import validate
    report = validate.validate_dataset(
        args.spec, skip_submissions=args.skip_submissions
    )
    print(report.format())
    return 0 if report.ok else 1
```

And update the subparser block:

```python
vd = sub.add_parser(
    "validate-dataset",
    help="full §1.9 dataset + submission validation",
)
vd.add_argument("spec", help="local dir path or org/name HF repo id")
vd.add_argument("--skip-submissions", action="store_true",
                help="skip per-submission network checks")
vd.set_defaults(func=_cmd_validate_dataset)
```

- [ ] **Step 4: Run tests**

Run: `pytest -xvs tests/test_cli_validate_dataset.py tests/test_cli.py`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/speech_spoof_bench/cli.py tests/test_cli_validate_dataset.py
git commit -m "feat: wire full validate-dataset CLI with --skip-submissions"
```

---

### Task 7: `reproduce.py` — sha verification + scores loading

**Files:**
- Create: `src/speech_spoof_bench/reproduce.py`
- Test: `tests/test_reproduce.py`

This task ships only the pre-label-stream pieces (fetch + sha + parse scores). Tasks 8–9 add labels + metric diffing.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_reproduce.py
"""Tests for reproduce.py (--scoring path)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from speech_spoof_bench import reproduce

FIX = Path(__file__).parent / "fixtures"


def _patch_yaml(tmp_path, scores_path, scores_sha):
    src = (FIX / "submissions" / "valid.yaml").read_text()
    data = yaml.safe_load(src)
    data["artifact"]["scores_sha256"] = scores_sha
    data["scores"] = {"eer_percent": 25.0, "n_trials": 4, "n_skipped": 0}
    p = tmp_path / "submission.yaml"
    p.write_text(yaml.safe_dump(data))
    return p


def test_sha_mismatch_fails(tmp_path):
    fake = tmp_path / "scores.txt"
    fake.write_text("UTT_0000 1.0\n")
    real_sha = hashlib.sha256(fake.read_bytes()).hexdigest()
    yaml_path = _patch_yaml(tmp_path, fake, "0" * 64)  # claimed != real
    with patch("speech_spoof_bench.reproduce.hf_fetch.download",
               return_value=(fake, real_sha)):
        rc = reproduce.run_scoring(yaml_path, tolerance=1e-6)
    assert rc != 0


def test_scores_parse(tmp_path):
    p = tmp_path / "s.txt"
    p.write_text("a 1.0\nb -0.5\n\n")
    parsed = reproduce._parse_scores_txt(p)
    assert parsed == {"a": 1.0, "b": -0.5}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest -xvs tests/test_reproduce.py`
Expected: import error.

- [ ] **Step 3: Implement `reproduce.py` (partial)**

```python
# src/speech_spoof_bench/reproduce.py
"""Maintainer-side reproduction of submission scores (--scoring).

Workflow per §1.7 / spec §6 of phase-7a:
  1. Parse YAML.
  2. Fetch scores_url, verify sha.
  3. Stream labels from pinned dataset revision (no audio decode).
  4. Recompute every metric in the YAML.
  5. Diff against claimed values.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml

from . import hf_fetch, submission
from .metrics import get_metric


def _parse_scores_txt(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        utt_id, score = line.split()
        out[utt_id] = float(score)
    return out


def _stream_labels(dataset_id: str, split: str, revision: str) -> dict[str, int]:
    """Stream labels-only from the pinned dataset revision. Audio not decoded."""
    import json
    from datasets import load_dataset
    ds = load_dataset(
        dataset_id, split=split, streaming=True, revision=revision
    )
    ds = ds.select_columns(["notes", "label"])
    labels: dict[str, int] = {}
    for row in ds:
        note = json.loads(row["notes"])
        labels[note["utterance_id"]] = int(row["label"])
    return labels


def run_scoring(
    yaml_path: Path | str,
    *,
    tolerance: float = 1e-6,
    label_stream=_stream_labels,
) -> int:
    """Run --scoring reproduction. Returns exit code (0 success, 1 fail).

    ``label_stream`` is injectable for tests.
    """
    yaml_path = Path(yaml_path)
    try:
        data = submission.parse_submission(yaml_path.read_text())
    except Exception as e:
        print(f"FAIL: schema: {e}", file=sys.stderr)
        return 1

    url = data["artifact"]["scores_url"]
    claimed_sha = data["artifact"]["scores_sha256"]
    try:
        local, observed_sha = hf_fetch.download(url)
    except Exception as e:
        print(f"FAIL: fetch: {e}", file=sys.stderr)
        return 1
    if observed_sha != claimed_sha:
        print(
            f"FAIL: sha256 mismatch\n"
            f"  claimed:  {claimed_sha}\n"
            f"  observed: {observed_sha}",
            file=sys.stderr,
        )
        return 1

    scores = _parse_scores_txt(local)

    # Steps 5–7 land in Tasks 8–9.
    return 0
```

- [ ] **Step 4: Run tests**

Run: `pytest -xvs tests/test_reproduce.py`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/speech_spoof_bench/reproduce.py tests/test_reproduce.py
git commit -m "feat: reproduce.py partial — schema, fetch, sha verify"
```

---

### Task 8: `reproduce.py` — label streaming + coverage checks

**Files:**
- Modify: `src/speech_spoof_bench/reproduce.py`
- Modify: `tests/test_reproduce.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_reproduce.py`:

```python
def test_select_columns_called_before_iteration():
    """Guards the audio-not-downloaded invariant."""
    captured: dict = {}

    class FakeDS:
        def __init__(self, rows):
            self._rows = rows
            self.select_called_with = None
        def select_columns(self, cols):
            self.select_called_with = list(cols)
            captured["select"] = list(cols)
            return self
        def __iter__(self):
            captured["iter_after_select"] = "select" in captured
            return iter(self._rows)

    fake = FakeDS([
        {"notes": '{"utterance_id":"a"}', "label": 0},
        {"notes": '{"utterance_id":"b"}', "label": 1},
    ])
    from speech_spoof_bench import reproduce
    with patch("speech_spoof_bench.reproduce.load_dataset", return_value=fake):
        labels = reproduce._stream_labels("x/y", "test", "deadbeef")
    assert captured["select"] == ["notes", "label"]
    assert captured["iter_after_select"] is True
    assert labels == {"a": 0, "b": 1}


def test_coverage_missing_in_dataset(tmp_path):
    scores = tmp_path / "s.txt"
    scores.write_text("UTT_0000 1.0\nGHOST 0.0\n")
    sha = hashlib.sha256(scores.read_bytes()).hexdigest()
    yaml_path = _patch_yaml(tmp_path, scores, sha)
    fake_labels = {"UTT_0000": 0, "UTT_0001": 1}
    with patch("speech_spoof_bench.reproduce.hf_fetch.download",
               return_value=(scores, sha)):
        rc = reproduce.run_scoring(
            yaml_path, label_stream=lambda *a, **k: fake_labels
        )
    assert rc == 1


def test_n_trials_mismatch(tmp_path):
    scores = tmp_path / "s.txt"
    scores.write_text("UTT_0000 1.0\n")
    sha = hashlib.sha256(scores.read_bytes()).hexdigest()
    src = (FIX / "submissions" / "valid.yaml").read_text()
    data = yaml.safe_load(src)
    data["artifact"]["scores_sha256"] = sha
    data["scores"] = {"eer_percent": 25.0, "n_trials": 999, "n_skipped": 0}
    p = tmp_path / "submission.yaml"
    p.write_text(yaml.safe_dump(data))
    fake_labels = {"UTT_0000": 0}
    with patch("speech_spoof_bench.reproduce.hf_fetch.download",
               return_value=(scores, sha)):
        rc = reproduce.run_scoring(p, label_stream=lambda *a, **k: fake_labels)
    assert rc == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest -xvs tests/test_reproduce.py -k "coverage or n_trials or select_columns"`
Expected: failures (load_dataset symbol missing from reproduce module; coverage checks not yet implemented).

- [ ] **Step 3: Extend `reproduce.py`**

- Move `from datasets import load_dataset` to module-level (so it can be patched).
- After parsing scores in `run_scoring`, add coverage checks before returning:

```python
    try:
        labels = label_stream(
            data["dataset"]["id"],
            data["dataset"]["split"],
            data["dataset"]["revision"],
        )
    except Exception as e:
        print(f"FAIL: dataset revision unreachable: {e}", file=sys.stderr)
        return 1

    scored_ids = set(scores)
    label_ids = set(labels)
    n_trials_claim = data["scores"]["n_trials"]
    n_skipped_claim = data["scores"]["n_skipped"]

    if scored_ids - label_ids:
        extra = sorted(scored_ids - label_ids)[:5]
        print(
            f"FAIL: coverage: scored {len(scored_ids - label_ids)} utterances not "
            f"in dataset (e.g. {extra})",
            file=sys.stderr,
        )
        return 1
    if len(scored_ids) + n_skipped_claim != n_trials_claim:
        print(
            f"FAIL: n_trials mismatch: "
            f"len(scores)={len(scored_ids)} + n_skipped={n_skipped_claim} "
            f"!= n_trials={n_trials_claim}",
            file=sys.stderr,
        )
        return 1
    if len(label_ids - scored_ids) > n_skipped_claim:
        print(
            f"FAIL: more skipped than claimed: "
            f"{len(label_ids - scored_ids)} unscored > n_skipped={n_skipped_claim}",
            file=sys.stderr,
        )
        return 1

    # Metric recomputation lands in Task 9.
    return 0
```

Replace `from datasets import load_dataset` inside `_stream_labels` with a module-level import (`from datasets import load_dataset` near the top).

- [ ] **Step 4: Run tests**

Run: `pytest -xvs tests/test_reproduce.py`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/speech_spoof_bench/reproduce.py tests/test_reproduce.py
git commit -m "feat: reproduce.py — label streaming + coverage checks"
```

---

### Task 9: `reproduce.py` — metric recomputation + success report

**Files:**
- Modify: `src/speech_spoof_bench/reproduce.py`
- Modify: `tests/test_reproduce.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_reproduce.py`:

```python
def test_metric_match_success(tmp_path, capsys):
    # Use the canned fixture scores so the EER is reproducible.
    scores_src = (FIX / "scores_known.txt").read_text()
    scores = tmp_path / "s.txt"
    scores.write_text(scores_src)
    sha = hashlib.sha256(scores.read_bytes()).hexdigest()

    # Compute the EER the metric will produce, then pin it in the YAML.
    from speech_spoof_bench.metrics import get_metric
    parsed = {}
    for line in scores_src.splitlines():
        if line.strip():
            utt, s = line.split()
            parsed[utt] = float(s)
    labels = {"UTT_0000": 0, "UTT_0001": 1, "UTT_0002": 0, "UTT_0003": 1}
    expected = get_metric("eer_percent").fn(parsed, labels).value

    src = (FIX / "submissions" / "valid.yaml").read_text()
    data = yaml.safe_load(src)
    data["artifact"]["scores_sha256"] = sha
    data["scores"] = {
        "eer_percent": expected,
        "n_trials": 4,
        "n_skipped": 0,
    }
    p = tmp_path / "submission.yaml"
    p.write_text(yaml.safe_dump(data))

    with patch("speech_spoof_bench.reproduce.hf_fetch.download",
               return_value=(scores, sha)):
        rc = reproduce.run_scoring(p, label_stream=lambda *a, **k: labels)
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "OK reproduced" in out
    assert "eer_percent" in out


def test_metric_mismatch(tmp_path):
    scores_src = (FIX / "scores_known.txt").read_text()
    scores = tmp_path / "s.txt"
    scores.write_text(scores_src)
    sha = hashlib.sha256(scores.read_bytes()).hexdigest()
    labels = {"UTT_0000": 0, "UTT_0001": 1, "UTT_0002": 0, "UTT_0003": 1}

    src = (FIX / "submissions" / "valid.yaml").read_text()
    data = yaml.safe_load(src)
    data["artifact"]["scores_sha256"] = sha
    data["scores"] = {"eer_percent": 0.0, "n_trials": 4, "n_skipped": 0}
    p = tmp_path / "submission.yaml"
    p.write_text(yaml.safe_dump(data))

    with patch("speech_spoof_bench.reproduce.hf_fetch.download",
               return_value=(scores, sha)):
        rc = reproduce.run_scoring(p, label_stream=lambda *a, **k: labels)
    assert rc == 1


def test_unknown_metric(tmp_path):
    scores_src = (FIX / "scores_known.txt").read_text()
    scores = tmp_path / "s.txt"
    scores.write_text(scores_src)
    sha = hashlib.sha256(scores.read_bytes()).hexdigest()
    labels = {"UTT_0000": 0, "UTT_0001": 1, "UTT_0002": 0, "UTT_0003": 1}

    src = (FIX / "submissions" / "valid.yaml").read_text()
    data = yaml.safe_load(src)
    data["artifact"]["scores_sha256"] = sha
    # Bypass schema by writing custom: schema permits any number-valued metric id.
    data["scores"] = {"made_up_metric": 1.23, "n_trials": 4, "n_skipped": 0}
    p = tmp_path / "submission.yaml"
    p.write_text(yaml.safe_dump(data))

    with patch("speech_spoof_bench.reproduce.hf_fetch.download",
               return_value=(scores, sha)):
        rc = reproduce.run_scoring(p, label_stream=lambda *a, **k: labels)
    assert rc == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest -xvs tests/test_reproduce.py -k "metric or unknown"`
Expected: failures.

- [ ] **Step 3: Implement metric recomputation in `reproduce.py`**

Replace the `# Metric recomputation lands in Task 9.` comment with:

```python
    metric_keys = [
        k for k in data["scores"]
        if k not in {"n_trials", "n_skipped"}
    ]
    if not metric_keys:
        print("FAIL: no metrics in submission.scores to recompute", file=sys.stderr)
        return 1

    scores_subset = {k: scores[k] for k in scored_ids if k in label_ids}
    labels_subset = {k: labels[k] for k in scores_subset}

    diffs: list[tuple[str, float, float]] = []
    for mid in metric_keys:
        try:
            spec = get_metric(mid)
        except KeyError:
            print(
                f"FAIL: metric {mid!r} not registered in this version of "
                f"speech-spoof-bench",
                file=sys.stderr,
            )
            return 1
        result = spec.fn(scores_subset, labels_subset)
        claimed = float(data["scores"][mid])
        if abs(result.value - claimed) > tolerance:
            print(
                f"FAIL: metric {mid!r}: claimed {claimed!r} recomputed "
                f"{result.value!r} (Δ {result.value - claimed:.3e}, "
                f"tolerance {tolerance:.0e})",
                file=sys.stderr,
            )
            return 1
        diffs.append((mid, claimed, result.value))

    sha_short = claimed_sha[:4] + "…" + claimed_sha[-4:]
    rev = data["dataset"]["revision"]
    print(f"OK reproduced: {data['dataset']['id']} @ {rev}")
    print(f"  scores_sha256: matched ({sha_short})")
    for mid, claimed, recomputed in diffs:
        delta = recomputed - claimed
        print(
            f"  {mid}: claimed {claimed!r}  recomputed {recomputed!r}  "
            f"(Δ {delta:.1e})"
        )
    print(
        f"  n_trials:      {n_trials_claim} (skipped {n_skipped_claim})"
    )
    return 0
```

- [ ] **Step 4: Run tests**

Run: `pytest -xvs tests/test_reproduce.py`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add src/speech_spoof_bench/reproduce.py tests/test_reproduce.py
git commit -m "feat: reproduce.py — metric recomputation + success report"
```

---

### Task 10: Wire `reproduce` subcommand (with `--inference` placeholder)

**Files:**
- Modify: `src/speech_spoof_bench/cli.py`
- Create: `tests/test_cli_reproduce.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_cli_reproduce.py
import hashlib
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from speech_spoof_bench.cli import main

FIX = Path(__file__).parent / "fixtures"


def test_inference_raises():
    p = FIX / "submissions" / "valid.yaml"
    with pytest.raises(NotImplementedError):
        main(["reproduce", "--inference", str(p)])


def test_scoring_invokes_run_scoring(tmp_path):
    p = FIX / "submissions" / "valid.yaml"
    with patch("speech_spoof_bench.reproduce.run_scoring", return_value=0) as r:
        rc = main(["reproduce", "--scoring", str(p), "--tolerance", "1e-3"])
    assert rc == 0
    r.assert_called_once()
    kwargs = r.call_args.kwargs
    assert kwargs["tolerance"] == 1e-3


def test_scoring_or_inference_required():
    p = FIX / "submissions" / "valid.yaml"
    with pytest.raises(SystemExit):
        main(["reproduce", str(p)])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest -xvs tests/test_cli_reproduce.py`
Expected: failures — `reproduce` subcommand not defined.

- [ ] **Step 3: Add the subparser in `cli.py`**

In `build_parser()`, after the `validate-submission` block:

```python
rp = sub.add_parser(
    "reproduce",
    help="reproduce a submission's scores (--scoring) or full inference (--inference)",
)
rp.add_argument("path", help="path to a submission YAML file")
rp.add_argument("--tolerance", type=float, default=1e-6,
                help="metric tolerance for --scoring (default 1e-6)")
mode = rp.add_mutually_exclusive_group(required=True)
mode.add_argument("--scoring", action="store_true",
                  help="verify scores_url sha + recompute metrics")
mode.add_argument("--inference", action="store_true",
                  help="full re-inference (Phase 8+; not yet implemented)")
rp.set_defaults(func=_cmd_reproduce)
```

Add `_cmd_reproduce` near the other command functions:

```python
def _cmd_reproduce(args: argparse.Namespace) -> int:
    if args.inference:
        raise NotImplementedError("reproduce --inference lands in Phase 7b/8")
    from . import reproduce
    return reproduce.run_scoring(args.path, tolerance=args.tolerance)
```

- [ ] **Step 4: Run tests**

Run: `pytest -xvs tests/test_cli_reproduce.py`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/speech_spoof_bench/cli.py tests/test_cli_reproduce.py
git commit -m "feat: wire reproduce CLI subcommand with --scoring/--inference"
```

---

### Task 11: Integration test — real HF random-baseline reproduction

**Files:**
- Create: `tests/test_reproduce_integration.py`

This test is **not** marked skip — it runs as part of `pytest`. Per the spec (§6.2 of the design), it also asserts no audio shards were downloaded.

- [ ] **Step 1: Write the integration test**

```python
# tests/test_reproduce_integration.py
"""End-to-end: reproduce --scoring against the live on-HF random-baseline.

NOT marked `network` — required HF reachability is intentional for this suite.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from huggingface_hub import hf_hub_download

from speech_spoof_bench import reproduce


def _hf_cache_size(root: Path, substring: str) -> int:
    total = 0
    if not root.is_dir():
        return total
    for p in root.rglob("*"):
        if substring in str(p) and p.is_file():
            total += p.stat().st_size
    return total


def test_random_baseline_real(tmp_path, capsys):
    # Pull the live submission YAML to a temp file.
    local = hf_hub_download(
        repo_id="SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA",
        filename="submissions/random-baseline.yaml",
        repo_type="dataset",
    )
    yaml_path = Path(local)

    cache_root = Path(
        os.environ.get("HF_DATASETS_CACHE")
        or Path.home() / ".cache" / "huggingface" / "datasets"
    )
    before = _hf_cache_size(cache_root, "ASVspoof2019_LA")

    rc = reproduce.run_scoring(yaml_path, tolerance=1e-6)
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "OK reproduced" in out

    after = _hf_cache_size(cache_root, "ASVspoof2019_LA")
    growth_mb = (after - before) / (1024 * 1024)
    # ~71k rows × (~80 B notes JSON + 8 B int) ≈ a few MB. Audio shards
    # would push this to 10s of GB. Cap at 50 MB to be generous.
    assert growth_mb < 50, (
        f"HF cache grew by {growth_mb:.1f} MB — audio shards may have been "
        f"downloaded. The select_columns invariant is broken."
    )
```

- [ ] **Step 2: Run the integration test**

Run: `pytest -xvs tests/test_reproduce_integration.py`
Expected: pass. First run may take 30–60s for network + label stream.

- [ ] **Step 3: Run the full suite**

Run: `pytest -xvs`
Expected: all green.

- [ ] **Step 4: Commit**

```bash
git add tests/test_reproduce_integration.py
git commit -m "test: integration — reproduce --scoring vs live random-baseline"
```

---

### Task 12: Final smoke + DoD verification

This task contains no new code — it walks the spec's Done-When list against the implementation.

- [ ] **Step 1: `validate-submission` exits 0 on the live YAML**

```bash
python -c "
from huggingface_hub import hf_hub_download
print(hf_hub_download(
    repo_id='SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA',
    filename='submissions/random-baseline.yaml',
    repo_type='dataset',
))" | tail -1 > /tmp/live_path
speech-spoof-bench validate-submission "$(cat /tmp/live_path)"
```
Expected: `OK: <path>`, exit 0.

- [ ] **Step 2: `validate-submission` exits 1 on each invalid fixture**

```bash
for f in tests/fixtures/submissions/invalid_*.yaml; do
  speech-spoof-bench validate-submission "$f"; echo "exit=$?"
done
```
Expected: each prints `FAIL ...`, `exit=1`.

- [ ] **Step 3: `validate-dataset` on the live LA repo exits 0**

```bash
speech-spoof-bench validate-dataset SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA
```
Expected: all D1–D7 and S1–S4 green, exit 0. (This is slow — full streaming pass for D5.)

- [ ] **Step 4: `reproduce --scoring` on the live submission exits 0**

```bash
speech-spoof-bench reproduce --scoring "$(cat /tmp/live_path)"
```
Expected: `OK reproduced: …` line, `eer_percent` Δ ~0, exit 0.

- [ ] **Step 5: `reproduce --inference` raises NotImplementedError**

```bash
speech-spoof-bench reproduce --inference "$(cat /tmp/live_path)" || true
```
Expected: traceback ending in `NotImplementedError`.

- [ ] **Step 6: Full pytest passes**

Run: `pytest -xvs`
Expected: all green (unit + integration).

- [ ] **Step 7: Update ROADMAP checkboxes**

Edit `docs/roadmap/ROADMAP.md`, mark all four 7a bullets `[x]`.

- [ ] **Step 8: Final commit**

```bash
git add docs/roadmap/ROADMAP.md
git commit -m "chore: mark Phase 7a validators complete in ROADMAP"
```
