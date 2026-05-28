# Phase 9 — Badge layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** After a submission PR is merged on an HF dataset repo, post a second bot comment on the discussion containing a paste-ready `result.yaml`, a static shields.io badge line, and a `huggingface-cli upload` one-liner. Pasting both into the model repo renders a backlink badge on the HF model page.

**Architecture:** Webhook on `refs/heads/main` with PR metadata → dispatches a new GitHub Actions workflow → CLI command fetches the merged submission YAML(s) at the merge sha, reads the dataset's primary metric from `eval.yaml`, builds the comment via pure string-builders in `badge.py`, posts via `HfApi.comment_discussion`. Idempotent via an HTML-comment sentinel.

**Tech Stack:** Python 3.10+, `huggingface_hub`, `jsonschema`, `pyyaml`, `pytest`, `unittest.mock`, FastAPI (webhook), GitHub Actions.

**Spec:** `docs/specs/2026-05-28-phase-9-badge-design.md`

**Sub-repos involved** (each has its own `.git`):
- `speech-spoof-bench/` — pip package + workflows (most of this plan)
- `arena/` — webhook handler (one task: webhook dispatch + tests)

Always `cd` into the right sub-repo before `git add`/`git commit`.

---

## Slice 1 — `badge.py` and `result.schema.json` (pure, no I/O)

### Task 1: Add `result.schema.json`

**Files:**
- Create: `speech-spoof-bench/src/speech_spoof_bench/schema/result.schema.json`
- Create: `speech-spoof-bench/tests/test_badge_result_schema.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_badge_result_schema.py
"""Validates result.schema.json's strictness boundary."""
from __future__ import annotations

import json
from importlib import resources

import pytest
from jsonschema import ValidationError, validate


def _schema():
    with resources.files("speech_spoof_bench.schema").joinpath("result.schema.json").open("r") as f:
        return json.load(f)


def _good():
    return {
        "schema_version": 1,
        "system": {
            "name": "AASIST",
            "slug": "aasist",
            "paper": {"arxiv_id": "2110.01200"},
        },
        "dataset": {
            "id": "Org/Foo",
            "revision": "abc1234",
            "split": "test",
        },
        "scores": {
            "n_trials": 1,
            "n_skipped": 0,
            "eer_percent": 1.23,
        },
        "arena": {
            "url": "https://huggingface.co/spaces/Org/Arena",
            "system_url": "https://huggingface.co/spaces/Org/Arena?system=aasist",
        },
        "artifact": {
            "scores_url": "https://huggingface.co/u/r/resolve/abc1234/.eval_results/Org/Foo/scores.txt",
        },
    }


def test_good_passes():
    validate(instance=_good(), schema=_schema())


def test_wrong_schema_version_rejected():
    d = _good(); d["schema_version"] = 2
    with pytest.raises(ValidationError):
        validate(instance=d, schema=_schema())


def test_extra_top_level_key_rejected():
    d = _good(); d["surprise"] = 1
    with pytest.raises(ValidationError):
        validate(instance=d, schema=_schema())


def test_missing_arena_system_url_rejected():
    d = _good(); del d["arena"]["system_url"]
    with pytest.raises(ValidationError):
        validate(instance=d, schema=_schema())


def test_non_https_arena_url_rejected():
    d = _good(); d["arena"]["url"] = "http://insecure.example"
    with pytest.raises(ValidationError):
        validate(instance=d, schema=_schema())


def test_scores_must_have_at_least_one_metric_besides_counts():
    d = _good(); d["scores"] = {"n_trials": 1, "n_skipped": 0}
    with pytest.raises(ValidationError):
        validate(instance=d, schema=_schema())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd speech-spoof-bench && pytest tests/test_badge_result_schema.py -v`
Expected: FAIL — file does not exist yet.

- [ ] **Step 3: Create the schema file**

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "speech-spoof-bench result.yaml (badge layer)",
  "type": "object",
  "additionalProperties": false,
  "required": ["schema_version", "system", "dataset", "scores", "arena", "artifact"],
  "properties": {
    "schema_version": {"type": "integer", "const": 1},
    "system": {
      "type": "object",
      "additionalProperties": false,
      "required": ["name", "slug", "paper"],
      "properties": {
        "name": {"type": "string", "minLength": 1},
        "slug": {"type": "string", "pattern": "^[a-z0-9][a-z0-9-]*$"},
        "paper": {
          "type": "object",
          "additionalProperties": false,
          "required": ["arxiv_id"],
          "properties": {
            "arxiv_id": {"type": "string", "minLength": 1}
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
        "revision": {"type": "string", "pattern": "^[0-9a-f]{7,40}$"},
        "split": {"type": "string", "minLength": 1}
      }
    },
    "scores": {
      "type": "object",
      "required": ["n_trials", "n_skipped"],
      "minProperties": 3,
      "properties": {
        "n_trials": {"type": "integer", "minimum": 0},
        "n_skipped": {"type": "integer", "minimum": 0}
      },
      "patternProperties": {
        "^(?!n_trials$|n_skipped$).+$": {"type": "number"}
      },
      "additionalProperties": false
    },
    "arena": {
      "type": "object",
      "additionalProperties": false,
      "required": ["url", "system_url"],
      "properties": {
        "url": {"type": "string", "pattern": "^https://"},
        "system_url": {"type": "string", "pattern": "^https://"}
      }
    },
    "artifact": {
      "type": "object",
      "additionalProperties": false,
      "required": ["scores_url"],
      "properties": {
        "scores_url": {
          "type": "string",
          "pattern": "^https://huggingface\\.co/[^/]+/[^/]+/resolve/[0-9a-f]{7,40}/"
        }
      }
    }
  }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd speech-spoof-bench && pytest tests/test_badge_result_schema.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
cd speech-spoof-bench
git add src/speech_spoof_bench/schema/result.schema.json tests/test_badge_result_schema.py
git commit -m "feat(badge): result.schema.json + strictness tests"
```

---

### Task 2: `badge._color_for_eer` thresholds

**Files:**
- Create: `speech-spoof-bench/src/speech_spoof_bench/badge.py`
- Create: `speech-spoof-bench/tests/test_badge_color.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_badge_color.py
import pytest
from speech_spoof_bench import badge


@pytest.mark.parametrize("eer,expected", [
    (0.0, "brightgreen"),
    (1.99, "brightgreen"),
    (2.0, "green"),         # >= 2.0 → green
    (4.99, "green"),
    (5.0, "yellow"),        # >= 5.0 → yellow
    (9.99, "yellow"),
    (10.0, "lightgrey"),    # >= 10.0 → lightgrey
    (50.0, "lightgrey"),
])
def test_color_for_eer(eer, expected):
    assert badge._color_for_eer(eer) == expected
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd speech-spoof-bench && pytest tests/test_badge_color.py -v`
Expected: FAIL — `module 'speech_spoof_bench' has no attribute 'badge'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/speech_spoof_bench/badge.py
"""Badge-layer string builders (Phase 9).

Pure functions: input parsed dicts, output strings. No I/O.
"""

from __future__ import annotations


class BadgeError(Exception):
    """Raised on input that cannot produce a valid result.yaml or comment."""


def _color_for_eer(eer_percent: float) -> str:
    if eer_percent < 2.0:
        return "brightgreen"
    if eer_percent < 5.0:
        return "green"
    if eer_percent < 10.0:
        return "yellow"
    return "lightgrey"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd speech-spoof-bench && pytest tests/test_badge_color.py -v`
Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
cd speech-spoof-bench
git add src/speech_spoof_bench/badge.py tests/test_badge_color.py
git commit -m "feat(badge): _color_for_eer thresholds"
```

---

### Task 3: `badge.build_result_yaml`

**Files:**
- Modify: `speech-spoof-bench/src/speech_spoof_bench/badge.py`
- Create: `speech-spoof-bench/tests/test_badge_build_result_yaml.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_badge_build_result_yaml.py
"""build_result_yaml is a pure renderer; snapshot the YAML string."""
from __future__ import annotations

import pytest
import yaml

from speech_spoof_bench import badge


_ARENA_URL = "https://huggingface.co/spaces/Org/Arena"

_SUBMISSION = {
    "schema_version": 4,
    "system": {
        "name": "AASIST",
        "slug": "aasist",
        "description": "x",
        "code": "https://x",
        "checkpoint": "https://x",
        "paper": {
            "arxiv_id": "2110.01200",
            "url": "https://arxiv.org/abs/2110.01200",
            "bibtex": "@x{1, }",
        },
    },
    "dataset": {"id": "Org/Foo", "revision": "abc1234", "split": "test"},
    "scores": {"eer_percent": 1.23, "n_trials": 71237, "n_skipped": 0},
    "artifact": {
        "scores_url": "https://huggingface.co/u/r/resolve/abc1234/.eval_results/Org/Foo/scores.txt",
        "scores_sha256": "0" * 64,
        "bench_version": "speech-spoof-bench==0.1.0",
    },
    "submitter": {"hf_username": "u", "contact": "u@example.com"},
    "submitted_at": "2026-05-23",
}


def test_returns_string_parseable_as_mapping():
    out = badge.build_result_yaml(_SUBMISSION, arena_url=_ARENA_URL)
    assert isinstance(out, str)
    parsed = yaml.safe_load(out)
    assert isinstance(parsed, dict)


def test_required_top_level_keys_present():
    parsed = yaml.safe_load(badge.build_result_yaml(_SUBMISSION, arena_url=_ARENA_URL))
    assert set(parsed.keys()) == {
        "schema_version", "system", "dataset", "scores", "arena", "artifact",
    }


def test_drops_submission_only_fields():
    parsed = yaml.safe_load(badge.build_result_yaml(_SUBMISSION, arena_url=_ARENA_URL))
    assert "submitter" not in parsed
    assert "reproduction" not in parsed
    assert "submitted_at" not in parsed
    # artifact loses scores_sha256 and bench_version too
    assert set(parsed["artifact"].keys()) == {"scores_url"}
    # system keeps name/slug/paper.arxiv_id only
    assert set(parsed["system"].keys()) == {"name", "slug", "paper"}
    assert set(parsed["system"]["paper"].keys()) == {"arxiv_id"}


def test_scores_preserves_counts_and_metrics():
    parsed = yaml.safe_load(badge.build_result_yaml(_SUBMISSION, arena_url=_ARENA_URL))
    assert parsed["scores"] == {"eer_percent": 1.23, "n_trials": 71237, "n_skipped": 0}


def test_arena_urls_constructed_from_slug():
    parsed = yaml.safe_load(badge.build_result_yaml(_SUBMISSION, arena_url=_ARENA_URL))
    assert parsed["arena"]["url"] == _ARENA_URL
    assert parsed["arena"]["system_url"] == f"{_ARENA_URL}?system=aasist"


def test_schema_version_is_one_not_four():
    parsed = yaml.safe_load(badge.build_result_yaml(_SUBMISSION, arena_url=_ARENA_URL))
    assert parsed["schema_version"] == 1  # result.yaml is its own schema, starting at 1


def test_invalid_submission_raises_badge_error():
    bad = {"schema_version": 4}  # missing required fields
    with pytest.raises(badge.BadgeError):
        badge.build_result_yaml(bad, arena_url=_ARENA_URL)


def test_output_passes_result_schema():
    """The whole point: rendered output must validate against result.schema.json."""
    import json
    from importlib import resources
    from jsonschema import validate

    with resources.files("speech_spoof_bench.schema").joinpath("result.schema.json").open("r") as f:
        schema = json.load(f)

    out = badge.build_result_yaml(_SUBMISSION, arena_url=_ARENA_URL)
    parsed = yaml.safe_load(out)
    validate(instance=parsed, schema=schema)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd speech-spoof-bench && pytest tests/test_badge_build_result_yaml.py -v`
Expected: FAIL — `module 'speech_spoof_bench.badge' has no attribute 'build_result_yaml'`.

- [ ] **Step 3: Write minimal implementation**

Add to `src/speech_spoof_bench/badge.py`:

```python
import json
from importlib import resources

import yaml
from jsonschema import ValidationError, validate


def _load_result_schema() -> dict:
    with resources.files("speech_spoof_bench.schema").joinpath("result.schema.json").open("r") as f:
        return json.load(f)


def _project_submission_for_result(submission: dict, *, arena_url: str) -> dict:
    """Pure projection: submission dict → result.yaml dict.

    Raises BadgeError if required fields are missing.
    """
    try:
        system = submission["system"]
        dataset = submission["dataset"]
        scores = submission["scores"]
        artifact = submission["artifact"]
        slug = system["slug"]
    except KeyError as exc:
        raise BadgeError(f"submission missing required key: {exc.args[0]}") from exc

    return {
        "schema_version": 1,
        "system": {
            "name": system["name"],
            "slug": slug,
            "paper": {"arxiv_id": system["paper"]["arxiv_id"]},
        },
        "dataset": {
            "id": dataset["id"],
            "revision": dataset["revision"],
            "split": dataset["split"],
        },
        "scores": dict(scores),  # metric values + n_trials + n_skipped
        "arena": {
            "url": arena_url,
            "system_url": f"{arena_url}?system={slug}",
        },
        "artifact": {
            "scores_url": artifact["scores_url"],
        },
    }


def build_result_yaml(submission: dict, *, arena_url: str) -> str:
    """Render the paste-ready result.yaml string. Validates output against
    result.schema.json before returning.
    """
    projected = _project_submission_for_result(submission, arena_url=arena_url)
    try:
        validate(instance=projected, schema=_load_result_schema())
    except ValidationError as exc:
        raise BadgeError(f"result.yaml failed schema validation: {exc.message}") from exc
    return yaml.safe_dump(projected, sort_keys=False, default_flow_style=False)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd speech-spoof-bench && pytest tests/test_badge_build_result_yaml.py -v`
Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
cd speech-spoof-bench
git add src/speech_spoof_bench/badge.py tests/test_badge_build_result_yaml.py
git commit -m "feat(badge): build_result_yaml projects submission → result.yaml"
```

---

### Task 4: `badge.build_paste_comment`

**Files:**
- Modify: `speech-spoof-bench/src/speech_spoof_bench/badge.py`
- Create: `speech-spoof-bench/tests/test_badge_build_paste_comment.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_badge_build_paste_comment.py
"""build_paste_comment renders the full Markdown body."""
from __future__ import annotations

import pytest

from speech_spoof_bench import badge

_ARENA_URL = "https://huggingface.co/spaces/Org/Arena"

_SUBMISSION = {
    "schema_version": 4,
    "system": {
        "name": "AASIST",
        "slug": "aasist",
        "description": "x",
        "code": "https://x",
        "checkpoint": "https://x",
        "paper": {"arxiv_id": "2110.01200", "url": "https://x", "bibtex": "@x{1, }"},
    },
    "dataset": {"id": "Org/ASVspoof2019_LA", "revision": "abc1234", "split": "test"},
    "scores": {"eer_percent": 1.23, "n_trials": 71237, "n_skipped": 0},
    "artifact": {
        "scores_url": "https://huggingface.co/u/r/resolve/abc1234/.eval_results/Org/ASVspoof2019_LA/scores.txt",
        "scores_sha256": "0" * 64,
        "bench_version": "speech-spoof-bench==0.1.0",
    },
    "submitter": {"hf_username": "u", "contact": "u@example.com"},
    "submitted_at": "2026-05-23",
}


def _build():
    return badge.build_paste_comment(
        _SUBMISSION,
        arena_url=_ARENA_URL,
        dataset_canonical_id="Org/ASVspoof2019_LA",
        primary_metric="eer_percent",
        submission_path="submissions/aasist.yaml",
        merge_sha="deadbeefcafe1234",
        gh_run_url="https://github.com/lab260ru/speech_spoof_bench/actions/runs/9",
    )


def test_includes_sentinel_with_sha_and_path():
    body = _build()
    assert "<!-- ssb:badge --> sha=deadbeefcafe1234 path=submissions/aasist.yaml" in body


def test_includes_result_yaml_block():
    body = _build()
    assert "schema_version: 1" in body
    assert "slug: aasist" in body


def test_includes_upload_one_liner():
    body = _build()
    assert "huggingface-cli upload" in body
    assert ".eval_results/Org/ASVspoof2019_LA/result.yaml" in body


def test_includes_shields_url_with_correct_encoding():
    body = _build()
    # Dataset underscores doubled, % encoded as %25, value baked in.
    assert "https://img.shields.io/badge/EER%25%20on%20ASVspoof2019__LA-1.23%25-brightgreen" in body


def test_badge_links_back_to_arena_with_slug():
    body = _build()
    assert f"]({_ARENA_URL}?system=aasist)" in body


def test_includes_ci_run_footer():
    body = _build()
    assert "https://github.com/lab260ru/speech_spoof_bench/actions/runs/9" in body


def test_raises_when_primary_metric_absent_from_scores():
    sub = {**_SUBMISSION, "scores": {"n_trials": 1, "n_skipped": 0}}
    with pytest.raises(badge.BadgeError):
        badge.build_paste_comment(
            sub, arena_url=_ARENA_URL,
            dataset_canonical_id="Org/Foo", primary_metric="eer_percent",
            submission_path="submissions/x.yaml", merge_sha="abc1234",
            gh_run_url="https://gh/run",
        )


def test_uses_color_for_high_eer():
    sub = {**_SUBMISSION, "scores": {"eer_percent": 20.0, "n_trials": 1, "n_skipped": 0}}
    body = badge.build_paste_comment(
        sub, arena_url=_ARENA_URL,
        dataset_canonical_id="Org/ASVspoof2019_LA", primary_metric="eer_percent",
        submission_path="submissions/x.yaml", merge_sha="abc1234",
        gh_run_url="https://gh/run",
    )
    assert "-lightgrey)" in body
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd speech-spoof-bench && pytest tests/test_badge_build_paste_comment.py -v`
Expected: FAIL — `module has no attribute 'build_paste_comment'`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/speech_spoof_bench/badge.py`:

```python
from urllib.parse import quote


def _shields_url(metric: str, dataset_name: str, value: float, color: str) -> str:
    """Build a static shields.io badge URL.

    Shields rules: `_` → real underscore, `__` → literal `_`. So pre-double
    underscores in our segment text, then URL-encode the result.
    """
    label = quote(f"{metric.upper().replace('_', ' ')} on {dataset_name}".replace("_", "__"))
    msg = quote(f"{value}%".replace("_", "__"))
    return f"https://img.shields.io/badge/{label}-{msg}-{color}"


def _metric_display_for(metric: str) -> str:
    """`eer_percent` → `EER` (the value already carries the % suffix)."""
    if metric == "eer_percent":
        return "EER%"
    return metric.upper().replace("_", " ")


def build_paste_comment(
    submission: dict,
    *,
    arena_url: str,
    dataset_canonical_id: str,
    primary_metric: str,
    submission_path: str,
    merge_sha: str,
    gh_run_url: str,
) -> str:
    """Render the full Markdown comment body for the post-merge badge step."""
    scores = submission.get("scores", {})
    if primary_metric not in scores:
        raise BadgeError(
            f"primary metric {primary_metric!r} missing from submission scores"
        )
    value = scores[primary_metric]
    slug = submission["system"]["slug"]
    dataset_name = dataset_canonical_id.split("/", 1)[1]
    color = _color_for_eer(value) if primary_metric == "eer_percent" else "blue"

    result_yaml = build_result_yaml(submission, arena_url=arena_url)
    metric_label = _metric_display_for(primary_metric)
    shields_url = _shields_url(metric_label, dataset_name, value, color)
    badge_md = (
        f"[![{metric_label} {value} on {dataset_name}]"
        f"({shields_url})]"
        f"({arena_url}?system={slug})"
    )

    return (
        f"**speech-spoof-bench** — submission merged ✅\n\n"
        f"System `{slug}` is now live on the [Arena]({arena_url}?system={slug}).\n\n"
        f"To display a backlink badge on your model page, take the two steps below.\n\n"
        f"### 1. Add `result.yaml` to your model repo\n\n"
        f"```yaml\n{result_yaml}```\n\n"
        f"Upload it with:\n\n"
        f"```bash\n"
        f"huggingface-cli upload <your-model-repo> result.yaml \\\n"
        f"  .eval_results/{dataset_canonical_id}/result.yaml\n"
        f"```\n\n"
        f"### 2. Add the badge line to your README\n\n"
        f"```markdown\n{badge_md}\n```\n\n"
        f"<!-- ssb:badge --> sha={merge_sha} path={submission_path}\n\n"
        f"---\n"
        f"_🤖 [view CI run]({gh_run_url})_\n"
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd speech-spoof-bench && pytest tests/test_badge_build_paste_comment.py -v`
Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
cd speech-spoof-bench
git add src/speech_spoof_bench/badge.py tests/test_badge_build_paste_comment.py
git commit -m "feat(badge): build_paste_comment renders full Markdown body"
```

---

## Slice 2 — `ci/post_merge_badge.py` + CLI wiring

### Task 5: `post_merge_badge.run` happy path (one submission)

**Files:**
- Create: `speech-spoof-bench/src/speech_spoof_bench/ci/post_merge_badge.py`
- Create: `speech-spoof-bench/tests/ci/test_post_merge_badge_happy.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/ci/test_post_merge_badge_happy.py
"""Happy path: one new submission YAML in the merge sha → one comment posted."""
from __future__ import annotations

from unittest.mock import MagicMock

from speech_spoof_bench.ci import post_merge_badge


def _good_yaml(slug="aasist", dataset="Org/ASVspoof2019_LA"):
    sha64 = "0" * 64
    return f"""schema_version: 4
system: {{name: AASIST, slug: {slug}, description: x, code: https://x, checkpoint: https://x, paper: {{arxiv_id: "2110.01200", url: https://x, bibtex: "@x{{1, }}"}}}}
dataset: {{id: {dataset}, revision: abc1234, split: test}}
scores: {{eer_percent: 1.23, n_trials: 1, n_skipped: 0}}
artifact: {{scores_url: "https://huggingface.co/u/r/resolve/abc1234/.eval_results/{dataset}/scores.txt", scores_sha256: "{sha64}", bench_version: "speech-spoof-bench==0.1.0"}}
submitter: {{hf_username: u, contact: u@example.com}}
submitted_at: 2026-05-23
"""


def _eval_yaml():
    return "name: ASVspoof2019 LA\ntasks:\n  - split: test\n    metrics: [eer_percent]\n"


def test_one_new_submission_posts_one_comment(monkeypatch, tmp_path):
    api = MagicMock()
    # main has just README; sha adds aasist.yaml.
    api.list_repo_files.side_effect = [
        ["submissions/README.md"],                          # main
        ["submissions/aasist.yaml", "submissions/README.md"],  # at sha
    ]
    # No prior comments on the discussion.
    api.get_discussion_details.return_value = MagicMock(events=[])

    def fake_dl(repo_id, filename, revision, repo_type):
        p = tmp_path / filename.replace("/", "_")
        if filename == "eval.yaml":
            p.write_text(_eval_yaml())
        else:
            p.write_text(_good_yaml())
        return str(p)
    monkeypatch.setattr(post_merge_badge, "_download_at_revision", fake_dl)

    posted = []
    def fake_post(repo, pr, body):
        posted.append((repo, pr, body))
    monkeypatch.setattr(post_merge_badge, "_post_comment", fake_post)

    rc = post_merge_badge.run(
        repo="Org/ASVspoof2019_LA", pr=42, sha="deadbeefcafe1234",
        api=api, gh_run_url="https://gh/run",
    )
    assert rc == 0
    assert len(posted) == 1
    repo, pr, body = posted[0]
    assert repo == "Org/ASVspoof2019_LA" and pr == 42
    assert "speech-spoof-bench" in body and "submission merged" in body
    assert "<!-- ssb:badge --> sha=deadbeefcafe1234 path=submissions/aasist.yaml" in body
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd speech-spoof-bench && pytest tests/ci/test_post_merge_badge_happy.py -v`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Write minimal implementation**

```python
# src/speech_spoof_bench/ci/post_merge_badge.py
"""`speech-spoof-bench ci post-merge-badge` — post the paste-ready badge
snippet to the merged HF discussion."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from huggingface_hub import HfApi, hf_hub_download

from .. import badge, submission

logger = logging.getLogger(__name__)

ARENA_URL = "https://huggingface.co/spaces/SpeechAntiSpoofingBenchmarks/SpeechAntiSpoofingArena"


@dataclass
class Outcome:
    path: str
    posted: bool
    notes: str


def _download_at_revision(repo_id: str, filename: str, revision: str, repo_type: str) -> str:
    return hf_hub_download(repo_id=repo_id, filename=filename,
                           revision=revision, repo_type=repo_type)


def _changed_submissions(api: HfApi, repo: str, sha: str) -> list[str]:
    main_files = set(api.list_repo_files(repo_id=repo, repo_type="dataset"))
    sha_files = set(api.list_repo_files(repo_id=repo, revision=sha, repo_type="dataset"))
    candidates = {
        f for f in sha_files
        if f.startswith("submissions/") and f.endswith(".yaml")
        and f.rsplit("/", 1)[-1] not in {"README.md", "results_template.yaml"}
    }
    added = candidates - main_files
    return sorted(added)


def _primary_metric_at(api: HfApi, repo: str, revision: str) -> str:
    local = _download_at_revision(repo, "eval.yaml", revision=revision, repo_type="dataset")
    data = yaml.safe_load(Path(local).read_text())
    tasks = (data or {}).get("tasks") or []
    if not tasks:
        raise badge.BadgeError(f"{repo}@{revision}: eval.yaml has no tasks")
    metrics = tasks[0].get("metrics") or []
    if not metrics:
        raise badge.BadgeError(f"{repo}@{revision}: eval.yaml task[0] has no metrics")
    return str(metrics[0])


def _sentinel_for(sha: str, path: str) -> str:
    return f"<!-- ssb:badge --> sha={sha} path={path}"


def _already_posted(api: HfApi, repo: str, pr: int, sentinel: str) -> bool:
    try:
        details = api.get_discussion_details(
            repo_id=repo, repo_type="dataset", discussion_num=pr,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("get_discussion_details failed for %s#%d: %s", repo, pr, exc)
        return False
    for ev in getattr(details, "events", []) or []:
        body = getattr(ev, "content", "") or ""
        if sentinel in body:
            return True
    return False


def _post_comment(repo: str, pr: int, body: str) -> None:
    token = os.environ.get("HF_BOT_TOKEN")
    if not token:
        logger.warning("HF_BOT_TOKEN not set; printing comment instead of posting")
        print(body)
        return
    api = HfApi(token=token)
    api.comment_discussion(repo_id=repo, repo_type="dataset",
                           discussion_num=pr, comment=body)


def run(*, repo: str, pr: int, sha: str,
        api: HfApi | None = None,
        gh_run_url: str | None = None) -> int:
    api = api or HfApi()
    gh_run_url = gh_run_url or os.environ.get(
        "GH_RUN_URL", "https://github.com/lab260ru/speech_spoof_bench/actions"
    )

    paths = _changed_submissions(api, repo, sha)
    if not paths:
        logger.info("no new submissions in %s@%s; nothing to do", repo, sha)
        return 0

    errors = 0
    for path in paths:
        sentinel = _sentinel_for(sha, path)
        if _already_posted(api, repo, pr, sentinel):
            logger.info("badge comment already present for %s; skipping", path)
            continue
        try:
            local = _download_at_revision(repo, path, revision=sha, repo_type="dataset")
            data = submission.parse_submission(Path(local).read_text())
            dataset_id = data["dataset"]["id"]
            dataset_rev = data["dataset"]["revision"]
            primary = _primary_metric_at(api, dataset_id, dataset_rev)
            body = badge.build_paste_comment(
                data,
                arena_url=ARENA_URL,
                dataset_canonical_id=dataset_id,
                primary_metric=primary,
                submission_path=path,
                merge_sha=sha,
                gh_run_url=gh_run_url,
            )
            _post_comment(repo, pr, body)
        except Exception as exc:  # noqa: BLE001
            logger.error("badge generation failed for %s: %s", path, exc)
            errors += 1
    return 0 if errors == 0 else 1
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd speech-spoof-bench && pytest tests/ci/test_post_merge_badge_happy.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd speech-spoof-bench
git add src/speech_spoof_bench/ci/post_merge_badge.py tests/ci/test_post_merge_badge_happy.py
git commit -m "feat(ci): post_merge_badge.run happy path"
```

---

### Task 6: Multiple submissions in the same merge

**Files:**
- Create: `speech-spoof-bench/tests/ci/test_post_merge_badge_multi.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/ci/test_post_merge_badge_multi.py
"""Two new submission YAMLs in one merge sha → two comments."""
from __future__ import annotations

from unittest.mock import MagicMock

from speech_spoof_bench.ci import post_merge_badge

from .test_post_merge_badge_happy import _good_yaml, _eval_yaml


def test_two_new_submissions_two_comments(monkeypatch, tmp_path):
    api = MagicMock()
    api.list_repo_files.side_effect = [
        ["submissions/README.md"],
        ["submissions/aasist.yaml", "submissions/rawnet.yaml", "submissions/README.md"],
    ]
    api.get_discussion_details.return_value = MagicMock(events=[])

    def fake_dl(repo_id, filename, revision, repo_type):
        p = tmp_path / f"{revision}_{filename.replace('/', '_')}"
        if filename == "eval.yaml":
            p.write_text(_eval_yaml())
        elif "aasist" in filename:
            p.write_text(_good_yaml(slug="aasist"))
        else:
            p.write_text(_good_yaml(slug="rawnet"))
        return str(p)
    monkeypatch.setattr(post_merge_badge, "_download_at_revision", fake_dl)

    posted = []
    monkeypatch.setattr(post_merge_badge, "_post_comment",
                        lambda r, p, b: posted.append((r, p, b)))

    rc = post_merge_badge.run(
        repo="Org/ASVspoof2019_LA", pr=42, sha="abc1234",
        api=api, gh_run_url="https://gh/run",
    )
    assert rc == 0
    assert len(posted) == 2
    slugs = {("aasist" if "slug: aasist" in b else "rawnet") for _, _, b in posted}
    assert slugs == {"aasist", "rawnet"}
```

- [ ] **Step 2: Run test to verify it passes**

Run: `cd speech-spoof-bench && pytest tests/ci/test_post_merge_badge_multi.py -v`
Expected: PASS — `_changed_submissions` already returns multiple, the loop handles them.

- [ ] **Step 3: Commit**

```bash
cd speech-spoof-bench
git add tests/ci/test_post_merge_badge_multi.py
git commit -m "test(ci): post-merge-badge handles multiple submissions per merge"
```

---

### Task 7: Idempotency via sentinel

**Files:**
- Create: `speech-spoof-bench/tests/ci/test_post_merge_badge_idempotent.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/ci/test_post_merge_badge_idempotent.py
"""A re-run against the same sha must NOT post a duplicate comment."""
from __future__ import annotations

from unittest.mock import MagicMock

from speech_spoof_bench.ci import post_merge_badge

from .test_post_merge_badge_happy import _good_yaml, _eval_yaml


def test_existing_sentinel_skips_post(monkeypatch, tmp_path):
    api = MagicMock()
    api.list_repo_files.side_effect = [
        ["submissions/README.md"],
        ["submissions/aasist.yaml", "submissions/README.md"],
    ]
    # Discussion already carries a comment with the sentinel for this sha+path.
    prior = MagicMock()
    prior.content = (
        "**speech-spoof-bench** — submission merged ✅\n"
        "<!-- ssb:badge --> sha=abc1234 path=submissions/aasist.yaml\n"
    )
    api.get_discussion_details.return_value = MagicMock(events=[prior])

    def fake_dl(repo_id, filename, revision, repo_type):
        p = tmp_path / filename.replace("/", "_")
        p.write_text(_eval_yaml() if filename == "eval.yaml" else _good_yaml())
        return str(p)
    monkeypatch.setattr(post_merge_badge, "_download_at_revision", fake_dl)

    posted = []
    monkeypatch.setattr(post_merge_badge, "_post_comment",
                        lambda r, p, b: posted.append(b))

    rc = post_merge_badge.run(
        repo="Org/Foo", pr=42, sha="abc1234", api=api, gh_run_url="x",
    )
    assert rc == 0
    assert posted == []  # nothing re-posted
```

- [ ] **Step 2: Run test to verify it passes**

Run: `cd speech-spoof-bench && pytest tests/ci/test_post_merge_badge_idempotent.py -v`
Expected: PASS — `_already_posted` returns True, the loop skips.

- [ ] **Step 3: Commit**

```bash
cd speech-spoof-bench
git add tests/ci/test_post_merge_badge_idempotent.py
git commit -m "test(ci): post-merge-badge is idempotent via sentinel"
```

---

### Task 8: Edge cases — no token, no changes, missing primary metric

**Files:**
- Create: `speech-spoof-bench/tests/ci/test_post_merge_badge_edges.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/ci/test_post_merge_badge_edges.py
"""Edge cases: no token / no changes / primary metric missing."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from speech_spoof_bench.ci import post_merge_badge

from .test_post_merge_badge_happy import _good_yaml, _eval_yaml


def test_no_submission_changes_returns_zero_no_comment(monkeypatch):
    api = MagicMock()
    api.list_repo_files.return_value = ["submissions/README.md"]
    posted = []
    monkeypatch.setattr(post_merge_badge, "_post_comment",
                        lambda r, p, b: posted.append(b))
    rc = post_merge_badge.run(
        repo="Org/Foo", pr=1, sha="abc1234", api=api, gh_run_url="x",
    )
    assert rc == 0
    assert posted == []


def test_missing_hf_bot_token_prints_to_stdout(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("HF_BOT_TOKEN", raising=False)
    api = MagicMock()
    api.list_repo_files.side_effect = [
        ["submissions/README.md"],
        ["submissions/aasist.yaml", "submissions/README.md"],
    ]
    api.get_discussion_details.return_value = MagicMock(events=[])

    def fake_dl(repo_id, filename, revision, repo_type):
        p = tmp_path / filename.replace("/", "_")
        p.write_text(_eval_yaml() if filename == "eval.yaml" else _good_yaml())
        return str(p)
    monkeypatch.setattr(post_merge_badge, "_download_at_revision", fake_dl)

    rc = post_merge_badge.run(
        repo="Org/Foo", pr=1, sha="abc1234", api=api, gh_run_url="x",
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "speech-spoof-bench" in out and "submission merged" in out
    # HfApi.comment_discussion must NOT be called.
    api.comment_discussion.assert_not_called()


def test_primary_metric_missing_exits_nonzero(monkeypatch, tmp_path):
    """If the submission lacks the dataset's primary metric, return non-zero."""
    api = MagicMock()
    api.list_repo_files.side_effect = [
        ["submissions/README.md"],
        ["submissions/aasist.yaml", "submissions/README.md"],
    ]
    api.get_discussion_details.return_value = MagicMock(events=[])

    def fake_dl(repo_id, filename, revision, repo_type):
        p = tmp_path / filename.replace("/", "_")
        if filename == "eval.yaml":
            # eval.yaml declares min_tdcf, but the submission only has eer_percent.
            p.write_text(
                "name: Foo\ntasks:\n  - split: test\n    metrics: [min_tdcf]\n"
            )
        else:
            p.write_text(_good_yaml())
        return str(p)
    monkeypatch.setattr(post_merge_badge, "_download_at_revision", fake_dl)

    posted = []
    monkeypatch.setattr(post_merge_badge, "_post_comment",
                        lambda r, p, b: posted.append(b))

    rc = post_merge_badge.run(
        repo="Org/Foo", pr=1, sha="abc1234", api=api, gh_run_url="x",
    )
    assert rc == 1
    assert posted == []
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `cd speech-spoof-bench && pytest tests/ci/test_post_merge_badge_edges.py -v`
Expected: PASS (3 tests).

- [ ] **Step 3: Commit**

```bash
cd speech-spoof-bench
git add tests/ci/test_post_merge_badge_edges.py
git commit -m "test(ci): post-merge-badge edge cases (no token / no changes / metric absent)"
```

---

### Task 9: Wire CLI `ci post-merge-badge`

**Files:**
- Modify: `speech-spoof-bench/src/speech_spoof_bench/cli.py`
- Create: `speech-spoof-bench/tests/test_cli_ci_post_merge_badge.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli_ci_post_merge_badge.py
"""CLI smoke: `ci post-merge-badge` dispatches to post_merge_badge.run."""
from __future__ import annotations

from unittest.mock import MagicMock

from speech_spoof_bench import cli


def test_cli_dispatches_with_args(monkeypatch):
    seen = {}
    def fake_run(**kwargs):
        seen.update(kwargs)
        return 0
    from speech_spoof_bench.ci import post_merge_badge
    monkeypatch.setattr(post_merge_badge, "run", fake_run)

    rc = cli.main([
        "ci", "post-merge-badge",
        "--repo", "Org/Foo",
        "--pr", "42",
        "--sha", "deadbeef",
    ])
    assert rc == 0
    assert seen == {"repo": "Org/Foo", "pr": 42, "sha": "deadbeef"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd speech-spoof-bench && pytest tests/test_cli_ci_post_merge_badge.py -v`
Expected: FAIL — subparser does not exist.

- [ ] **Step 3: Add CLI wiring**

In `src/speech_spoof_bench/cli.py`, add a new dispatcher function near `_cmd_ci_verify_pr`:

```python
def _cmd_ci_post_merge_badge(args: argparse.Namespace) -> int:
    from .ci import post_merge_badge
    return post_merge_badge.run(repo=args.repo, pr=int(args.pr), sha=args.sha)
```

In `build_parser()`, inside the `ci_sub` block (right after the `nr` subparser block, before `return p`):

```python
    pmb = ci_sub.add_parser(
        "post-merge-badge",
        help="post badge snippet to merged HF discussion (Phase 9)",
    )
    pmb.add_argument("--repo", required=True, help="dataset id (org/name)")
    pmb.add_argument("--pr", required=True, help="HF PR (discussion) number")
    pmb.add_argument("--sha", required=True, help="merge commit sha on the dataset main branch")
    pmb.set_defaults(func=_cmd_ci_post_merge_badge)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd speech-spoof-bench && pytest tests/test_cli_ci_post_merge_badge.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd speech-spoof-bench
git add src/speech_spoof_bench/cli.py tests/test_cli_ci_post_merge_badge.py
git commit -m "feat(cli): wire \`ci post-merge-badge\`"
```

---

## Slice 3 — GitHub Actions workflow + webhook bridge

### Task 10: Add `post-merge-badge.yml` workflow

**Files:**
- Create: `speech-spoof-bench/.github/workflows/post-merge-badge.yml`

- [ ] **Step 1: Write the workflow**

```yaml
# .github/workflows/post-merge-badge.yml
name: post-merge-badge

on:
  workflow_dispatch:
    inputs:
      repo:
        description: "HF dataset id (org/name)"
        required: true
        type: string
      pr:
        description: "HF PR / discussion number"
        required: true
        type: string
      sha:
        description: "Merge commit sha on the dataset main branch"
        required: true
        type: string

jobs:
  post:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install
        run: pip install -e .
      - name: Post badge snippet
        env:
          HF_BOT_TOKEN: ${{ secrets.HF_BOT_TOKEN }}
          GH_RUN_URL: ${{ github.server_url }}/${{ github.repository }}/actions/runs/${{ github.run_id }}
        run: |
          speech-spoof-bench ci post-merge-badge \
            --repo "${{ inputs.repo }}" \
            --pr   "${{ inputs.pr }}" \
            --sha  "${{ inputs.sha }}"
```

- [ ] **Step 2: Verify YAML is well-formed**

Run: `cd speech-spoof-bench && python -c "import yaml; yaml.safe_load(open('.github/workflows/post-merge-badge.yml'))"`
Expected: no output, no error.

- [ ] **Step 3: Commit**

```bash
cd speech-spoof-bench
git add .github/workflows/post-merge-badge.yml
git commit -m "ci(workflow): post-merge-badge.yml (workflow_dispatch)"
```

---

### Task 11: Webhook — dispatch new workflow when a `refs/heads/main` event carries PR metadata

**Files:**
- Modify: `arena/webhook.py`
- Modify: `arena/tests/test_webhook.py`

Important context: HF webhook payloads carry the merged PR/discussion under `payload["discussion"]["num"]` with `isPullRequest: True` (see existing `_pr_payload` in `arena/tests/test_webhook.py`). For a *merge*, the `refs/heads/main` event arrives with that same `discussion` block populated. That's the marker we use.

- [ ] **Step 1: Write the failing test**

In `arena/tests/test_webhook.py`, append:

```python
def _merge_payload(repo="Org/Foo", pr=42, sha="cafef00d"):
    return {
        "event": {"action": "update", "scope": "repo.content"},
        "repo":  {"type": "dataset", "name": repo},
        "updatedRefs": [{"ref": "refs/heads/main", "newSha": sha, "oldSha": "00"}],
        "discussion": {"num": pr, "isPullRequest": True},
    }


def test_main_event_with_pr_metadata_dispatches_post_merge(monkeypatch):
    monkeypatch.setenv("HF_WEBHOOK_SECRET", "s3cret")
    monkeypatch.setenv("GH_PAT", "ghp_xxx")
    import webhook as wh
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    monkeypatch.setattr(wh, "_subscribed_repos", lambda: {"Org/Foo"})
    monkeypatch.setattr(wh, "_refresh_and_commit", lambda repo: None)

    posted = []
    def fake_post(url, headers, json, timeout):
        posted.append({"url": url, "json": json})
        class R: status_code = 204
        return R()
    monkeypatch.setattr(wh.requests, "post", fake_post)

    wh._dup_cache.clear()
    app = FastAPI(); app.include_router(wh.router)
    client = TestClient(app)

    resp = client.post("/webhook", json=_merge_payload(),
                       headers={"X-Webhook-Secret": "s3cret"})
    assert resp.status_code == 200
    # Two dispatches: cache-refresh runs in BackgroundTasks, badge dispatch runs inline.
    urls = [p["url"] for p in posted]
    assert any(u.endswith("post-merge-badge.yml/dispatches") for u in urls)
    badge_call = next(p for p in posted if p["url"].endswith("post-merge-badge.yml/dispatches"))
    assert badge_call["json"] == {
        "ref": "main",
        "inputs": {"repo": "Org/Foo", "pr": "42", "sha": "cafef00d"},
    }


def test_main_event_without_pr_metadata_only_refreshes(client):
    """A direct-to-main commit (no discussion block) must not dispatch a badge."""
    c, _, _ = client
    resp = c.post("/webhook", json=_main_payload(),
                  headers={"X-Webhook-Secret": "s3cret"})
    assert resp.status_code == 200
    # The existing client fixture mocks load_state + save_and_commit but not
    # requests.post; if a dispatch were attempted it would fail. So 200 + no
    # exception is sufficient.
```

- [ ] **Step 2: Run tests to verify the merge test fails**

Run: `cd arena && pytest tests/test_webhook.py -v -k "post_merge or without_pr_metadata_only"`
Expected: `test_main_event_with_pr_metadata_dispatches_post_merge` FAILS — webhook only refreshes, no badge dispatch.

- [ ] **Step 3: Update webhook**

In `arena/webhook.py`, add a helper near `_dispatch_verify_workflow`:

```python
def _dispatch_post_merge_workflow(repo: str, pr: int, sha: str) -> None:
    token = os.environ.get("GH_PAT")
    target = os.environ.get("GH_VERIFY_WORKFLOW_REPO", "lab260ru/speech_spoof_bench")
    if not token:
        logger.warning("GH_PAT not set; skipping post-merge-badge dispatch")
        return
    url = f"{_GH_API}/repos/{target}/actions/workflows/post-merge-badge.yml/dispatches"
    resp = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
        },
        json={"ref": "main", "inputs": {"repo": repo, "pr": str(pr), "sha": sha}},
        timeout=10,
    )
    if resp.status_code >= 300:
        body = resp.text[:500]
        logger.warning(
            "GH dispatch (post-merge-badge) failed: target=%s pr=%s status=%s body=%s",
            target, pr, resp.status_code, body,
        )
```

Then in the `refs/heads/main` branch of `hf_webhook` (inside `for r in refs:`), modify the existing block:

```python
        if ref == "refs/heads/main":
            if _is_duplicate(repo, sha):
                return {"status": "ignored", "reason": "duplicate"}
            background.add_task(_refresh_and_commit, repo)
            discussion = payload.get("discussion") or {}
            if discussion.get("isPullRequest") and discussion.get("num") is not None:
                _dispatch_post_merge_workflow(repo, int(discussion["num"]), sha)
                return {"status": "scheduled", "kind": "refresh+post-merge"}
            return {"status": "scheduled", "kind": "refresh"}
```

Note the failure mode: post-merge dispatch failures are *logged-and-swallowed*, not raised. A flaky GH API call must not block the cache refresh, and the user can re-run the workflow manually if it doesn't fire.

- [ ] **Step 4: Run all webhook tests**

Run: `cd arena && pytest tests/test_webhook.py -v`
Expected: PASS — all existing tests still pass, the two new ones pass.

- [ ] **Step 5: Commit**

```bash
cd arena
git add webhook.py tests/test_webhook.py
git commit -m "feat(webhook): dispatch post-merge-badge.yml on PR-merge events"
```

---

## Slice 4 — Documentation

### Task 12: Add PLAN.md §3.6.2 and update ROADMAP Phase 9 entry

The roadmap Phase 9 entry references "§3.6.2" of PLAN.md which currently doesn't exist. Add it now so the reference resolves.

**Files:**
- Modify: `speech-spoof-bench/docs/roadmap/PLAN.md`
- Modify: `speech-spoof-bench/docs/roadmap/ROADMAP.md`

- [ ] **Step 1: Insert §3.6.2 into PLAN.md**

Insert this section between §3.6 (Generic tier system) and §3.7 (Ranking logic):

```markdown
### §3.6.2 Backlink badges (model-page side)

Phase 9 — after a submission PR is merged on a dataset repo, CI posts a
follow-up comment on the HF discussion with a paste-ready `result.yaml`,
a static `shields.io` badge line, and a `huggingface-cli upload` one-liner.
The submitter pastes the YAML into their model repo at
`.eval_results/<dataset-org>/<dataset-name>/result.yaml` and the badge line
into their README. The HF model page then renders a shields badge that
links back to the Arena.

`result.yaml` shape (`schema_version: 1`, validated by
`src/speech_spoof_bench/schema/result.schema.json`):

- `system`: `name`, `slug`, `paper.arxiv_id` — projected from the submission.
- `dataset`: `id`, `revision`, `split` — copied verbatim.
- `scores`: all metric values + `n_trials` + `n_skipped`.
- `arena.url`: Arena Space URL.
- `arena.system_url`: `<arena.url>?system=<slug>` — opaque to the Arena
  today, deep-linked in Phase 10.
- `artifact.scores_url`: copied verbatim from the merged submission.

No `submitter`, no `reproduction`, no `bench_version` — those are
submission-side concerns. The badge value comes from the dataset's *primary
metric* (first id in `eval.yaml`'s `metrics:` list at the submission's
pinned revision). Color band by EER percentile:
`<2% brightgreen / <5% green / <10% yellow / else lightgrey`.

The comment is made idempotent by an HTML sentinel:
`<!-- ssb:badge --> sha=<merge-sha> path=<submissions/*.yaml>`. CI scans
existing discussion comments for this marker before posting; a re-run of
the workflow on the same merge sha is a no-op.
```

- [ ] **Step 2: Update ROADMAP Phase 9**

In `docs/roadmap/ROADMAP.md`, the Phase 9 block currently reads:

```markdown
- [ ] `src/speech_spoof_bench/badge.py` — generates `result.yaml` per §3.6.2.
- [ ] `result.schema.json` — JSON Schema validator.
- [ ] `ci verify-pr` post-merge step — emits a second comment with the snippet and one-liner upload command.
- [ ] Manually verify once: paste into random-baseline model repo → badge renders on `huggingface.co/<you>/random-baseline-asas`.
```

Replace with:

```markdown
- [ ] `src/speech_spoof_bench/badge.py` — `build_result_yaml` + `build_paste_comment` per §3.6.2.
- [ ] `result.schema.json` — JSON Schema validator.
- [ ] `ci post-merge-badge` CLI + `post-merge-badge.yml` workflow — posts a follow-up comment on the merged HF discussion.
- [ ] `arena/webhook.py` — dispatches `post-merge-badge.yml` when a `refs/heads/main` event carries PR-merge metadata.
- [ ] Manually verify once: paste into random-baseline model repo → badge renders on `huggingface.co/<you>/random-baseline-asas`.
```

- [ ] **Step 3: Commit**

```bash
cd speech-spoof-bench
git add docs/roadmap/PLAN.md docs/roadmap/ROADMAP.md
git commit -m "docs(roadmap): add §3.6.2 (backlink badges); refresh Phase 9 entry"
```

---

## Manual end-to-end (Phase 9 Done-when)

These steps verify the system on real infra. They are not part of CI.

- [ ] **M1.** Confirm `HF_BOT_TOKEN` is already configured as a GitHub Actions secret on `lab260ru/speech_spoof_bench` (it is — Phase 8). No new secrets needed.

- [ ] **M2.** Open a trivial re-submission PR for `random-baseline-asas` on the `SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA` dataset repo. (A single-character edit to `notes:` is enough.) Wait for `verify-pr` to comment ✅.

- [ ] **M3.** Merge the PR via the HF UI. Within ~60s, a second bot comment must appear on the merged discussion carrying the YAML, badge line, and upload one-liner.

- [ ] **M4.** Copy the `result.yaml` block. Save it locally as `result.yaml`. Run the upload one-liner exactly as printed, replacing `<your-model-repo>` with `SpeechAntiSpoofingBenchmarks/random-baseline-asas`:

```bash
huggingface-cli upload SpeechAntiSpoofingBenchmarks/random-baseline-asas result.yaml \
  .eval_results/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA/result.yaml
```

- [ ] **M5.** Edit the model repo `README.md`: paste the badge Markdown line at the top of the body (under the front-matter `---`).

- [ ] **M6.** Visit `https://huggingface.co/SpeechAntiSpoofingBenchmarks/random-baseline-asas`. Verify the EER badge renders and clicks through to the Arena.

- [ ] **M7.** Re-run `post-merge-badge.yml` from the GH Actions UI with the same `{repo, pr, sha}` inputs. Verify no duplicate comment appears on the discussion (idempotency).

- [ ] **M8.** Check off Phase 9 in `docs/roadmap/ROADMAP.md` and update the Phase-8 example note in the "Critical-path summary" table if needed.

---

## Self-review notes

- **Spec coverage:** all five components from the spec (`badge.py`, `result.schema.json`, `post_merge_badge.py`, CLI, workflow, webhook change) are covered. PLAN.md §3.6.2 docs follow-up is Task 12. Manual DoD covered by M1–M8.
- **No placeholders:** every code block in Steps 1/3 is fully written. No "TBD" / "similar to Task N" / "handle edge cases".
- **Type consistency:** `build_paste_comment` signature in Task 4 matches its invocation in Task 5 (`post_merge_badge.run`). `_post_comment(repo, pr, body)` shape is identical to the existing `verify_pr` helper. `_changed_submissions` signature matches the existing `verify_pr` pattern.
- **Idempotency marker** is consistent: `<!-- ssb:badge --> sha=<sha> path=<path>` everywhere it appears (Task 4, Task 5, Task 7, §3.6.2).
- **Cross-repo discipline:** every commit explicitly `cd`s into the right sub-repo (`speech-spoof-bench/` or `arena/`) — these are independent `.git`s.
