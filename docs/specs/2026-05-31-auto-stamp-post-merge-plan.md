# Auto-Stamp Reproduction in Post-Merge — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** After a submission PR is merged, the `post-merge-badge` job auto-fills the `reproduction:` block on the dataset repo's `main` branch (trusting the pre-merge `verify-pr` result), so the model appears on the Arena hands-off.

**Architecture:** Fold a `_maybe_stamp` step into the existing per-path loop of `post_merge_badge.run()` (Approach A). It re-reads the merged YAML at `sha`, skips if `reproduction.match` is already set, gates on finding our own passing `verify-pr` discussion comment for that path, then writes the complete 4-key `reproduction` block via a `ruamel.yaml` round-trip (minimal diff) and commits it to `main` with a no-`(#N)` message. Three independent loop guards prevent a webhook→commit→webhook loop.

**Tech Stack:** Python 3.10+, `huggingface_hub` (`HfApi`), `ruamel.yaml` (new `[ci]` extra), `pytest` + `unittest.mock`.

**Source spec:** `docs/specs/2026-05-31-res2tcnguard-arena-submission-design.md` → "Addendum 2".

---

## File Structure

- **Modify** `pyproject.toml` — add `[project.optional-dependencies] ci = ["ruamel.yaml>=0.18"]`; add `ruamel.yaml` to `dev`.
- **Modify** `src/speech_spoof_bench/ci/post_merge_badge.py` — add `_today_iso`, `_already_stamped`, `_verify_pr_passed`, `_bot_identity`, `_fill_reproduction_block`, `_commit_stamp`, `_maybe_stamp`; wire `_maybe_stamp` into `run()`'s loop; add `today` kwarg to `run()`.
- **Create** `tests/ci/test_post_merge_badge_stamp.py` — unit + integration tests for the stamp path.
- **Modify** `.github/workflows/post-merge-badge.yml` and `.github/workflows/verify-hf-pr.yml` — `pip install -e .` → `pip install -e ".[ci]"`.
- **Modify** `docs/architecture/cicd.md` and `docs/architecture/submission-lifecycle.md` — document the new behavior.

Reference facts (verified against the codebase):
- `reproduction` schema (`src/speech_spoof_bench/schema/submission.schema.json`) is `oneOf`: empty `{}` (maxProperties 0) **or** a fully-populated block requiring exactly `reproduced_by` (minLength 1), `reproduced_at` (format date), `reproduced_bench_version` (minLength 1), `match` (enum `["scoring","inference"]`), `additionalProperties: false`.
- `verify_pr.format_markdown` posts a body containing `**speech-spoof-bench ci verify-pr**`, the overall line `✅ all checks passed` when every verdict passes, and a table row `` | `submissions/<slug>.yaml` | … `` per submission.
- Package version: `speech_spoof_bench.__version__` (currently `"0.3.0"`); the canonical `bench_version` string format is `speech-spoof-bench==X.Y.Z` (see `benchmark.py:76`).
- `_post_comment` in `post_merge_badge.py` is the existing pattern for building a token-bearing `HfApi(token=os.environ["HF_BOT_TOKEN"])`.

---

## Task 1: Add the `ci` extra (ruamel.yaml) to packaging

**Files:**
- Modify: `pyproject.toml:23-24` (the `[project.optional-dependencies]` block)

- [ ] **Step 1: Add the `ci` extra and put ruamel in `dev` too**

Replace the optional-dependencies block:

```toml
[project.optional-dependencies]
dev = ["pytest>=7.0", "pytest-cov>=4.0", "ruamel.yaml>=0.18"]
ci = ["ruamel.yaml>=0.18"]
```

- [ ] **Step 2: Install the extra into the working env**

Run: `pip install -e ".[dev]"`
Expected: completes; `ruamel.yaml` is installed.

- [ ] **Step 3: Verify the import works**

Run: `python -c "from ruamel.yaml import YAML; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "build: add ci extra with ruamel.yaml for post-merge auto-stamp"
```

---

## Task 2: `_fill_reproduction_block` — ruamel round-trip that fills the block

**Files:**
- Modify: `src/speech_spoof_bench/ci/post_merge_badge.py`
- Test: `tests/ci/test_post_merge_badge_stamp.py`

- [ ] **Step 1: Write the failing test**

Create `tests/ci/test_post_merge_badge_stamp.py`:

```python
"""Auto-stamp: fill the reproduction block on a merged submission."""
from __future__ import annotations

from pathlib import Path

from speech_spoof_bench import submission
from speech_spoof_bench.ci import post_merge_badge


SUBMISSION_EMPTY_REPRO = """schema_version: 4
system:
  name: AASIST
  slug: aasist
  description: x
  code: https://x
  checkpoint: https://x
  paper:
    arxiv_id: "2110.01200"
    url: https://x
    bibtex: "@x{1, }"
dataset:
  id: Org/ASVspoof2019_LA
  revision: abc1234
  split: test
scores:
  eer_percent: 1.23
  n_trials: 1
  n_skipped: 0
artifact:
  scores_url: "https://huggingface.co/u/r/resolve/abc1234/.eval_results/Org/ASVspoof2019_LA/scores.txt"
  scores_sha256: "{sha}"
  bench_version: "speech-spoof-bench==0.1.0"
# maintainer fills this at merge
reproduction: {{}}
submitter:
  hf_username: u
  contact: u@example.com
submitted_at: 2026-05-23
""".format(sha="0" * 64)


def test_fill_reproduction_block_sets_four_keys_and_is_schema_valid():
    out = post_merge_badge._fill_reproduction_block(
        SUBMISSION_EMPTY_REPRO,
        reproduced_by="ssb-ci-bot",
        reproduced_at="2026-05-31",
        bench_version="speech-spoof-bench==0.3.0",
    )
    # Other keys are preserved (minimal diff): the comment line survives.
    assert "# maintainer fills this at merge" in out
    assert "slug: aasist" in out
    # Parses + validates against the schema (oneOf populated branch).
    data = submission.parse_submission(out)
    repro = data["reproduction"]
    assert repro == {
        "reproduced_by": "ssb-ci-bot",
        "reproduced_at": "2026-05-31",
        "reproduced_bench_version": "speech-spoof-bench==0.3.0",
        "match": "scoring",
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/ci/test_post_merge_badge_stamp.py::test_fill_reproduction_block_sets_four_keys_and_is_schema_valid -v`
Expected: FAIL with `AttributeError: module 'speech_spoof_bench.ci.post_merge_badge' has no attribute '_fill_reproduction_block'`.

- [ ] **Step 3: Implement `_fill_reproduction_block`**

In `post_merge_badge.py`, add near the top after the existing imports (do **not** import ruamel at module top — keep it lazy so the package imports without the `ci` extra):

```python
import io


def _fill_reproduction_block(
    yaml_text: str, *, reproduced_by: str, reproduced_at: str, bench_version: str
) -> str:
    """Round-trip the submission YAML, replacing `reproduction` with a complete
    block. Uses ruamel to preserve key order and comments (minimal diff)."""
    from ruamel.yaml import YAML

    y = YAML()
    y.preserve_quotes = True
    data = y.load(yaml_text)
    data["reproduction"] = {
        "reproduced_by": reproduced_by,
        "reproduced_at": reproduced_at,
        "reproduced_bench_version": bench_version,
        "match": "scoring",
    }
    buf = io.StringIO()
    y.dump(data, buf)
    return buf.getvalue()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/ci/test_post_merge_badge_stamp.py::test_fill_reproduction_block_sets_four_keys_and_is_schema_valid -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/speech_spoof_bench/ci/post_merge_badge.py tests/ci/test_post_merge_badge_stamp.py
git commit -m "feat(ci): fill reproduction block via ruamel round-trip"
```

---

## Task 3: `_already_stamped` — loop guard #3

**Files:**
- Modify: `src/speech_spoof_bench/ci/post_merge_badge.py`
- Test: `tests/ci/test_post_merge_badge_stamp.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/ci/test_post_merge_badge_stamp.py`:

```python
def test_already_stamped_true_when_match_set():
    stamped = SUBMISSION_EMPTY_REPRO.replace(
        "reproduction: {}",
        "reproduction:\n"
        "  reproduced_by: ssb-ci-bot\n"
        "  reproduced_at: '2026-05-30'\n"
        "  reproduced_bench_version: 'speech-spoof-bench==0.3.0'\n"
        "  match: scoring",
    )
    assert post_merge_badge._already_stamped(stamped) is True


def test_already_stamped_false_when_empty():
    assert post_merge_badge._already_stamped(SUBMISSION_EMPTY_REPRO) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/ci/test_post_merge_badge_stamp.py -k already_stamped -v`
Expected: FAIL with `AttributeError: … has no attribute '_already_stamped'`.

- [ ] **Step 3: Implement `_already_stamped`**

In `post_merge_badge.py` (`yaml` is already imported at module top):

```python
def _already_stamped(yaml_text: str) -> bool:
    """True if the submission already carries a filled reproduction block."""
    repro = (yaml.safe_load(yaml_text) or {}).get("reproduction") or {}
    return bool(repro.get("match"))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/ci/test_post_merge_badge_stamp.py -k already_stamped -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/speech_spoof_bench/ci/post_merge_badge.py tests/ci/test_post_merge_badge_stamp.py
git commit -m "feat(ci): add _already_stamped reproduction guard"
```

---

## Task 4: `_verify_pr_passed` — the verify-pr gate (Component A1)

**Files:**
- Modify: `src/speech_spoof_bench/ci/post_merge_badge.py`
- Test: `tests/ci/test_post_merge_badge_stamp.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/ci/test_post_merge_badge_stamp.py`:

```python
from unittest.mock import MagicMock


def _ev(body):
    e = MagicMock()
    e.content = body
    return e


def _details(*bodies):
    return MagicMock(events=[_ev(b) for b in bodies])


def _verify_comment(path, passed=True):
    overall = "✅ all checks passed" if passed else "❌ failures present"
    return (
        f"**speech-spoof-bench ci verify-pr** — {overall}\n\n"
        f"| Submission | Schema |\n|---|---|\n| `{path}` | ✅ |\n"
    )


def test_verify_pr_passed_true_on_passing_comment():
    api = MagicMock()
    api.get_discussion_details.return_value = _details(
        _verify_comment("submissions/aasist.yaml", passed=True)
    )
    assert post_merge_badge._verify_pr_passed(
        api, "Org/Foo", 42, "submissions/aasist.yaml"
    ) is True


def test_verify_pr_passed_false_on_failing_comment():
    api = MagicMock()
    api.get_discussion_details.return_value = _details(
        _verify_comment("submissions/aasist.yaml", passed=False)
    )
    assert post_merge_badge._verify_pr_passed(
        api, "Org/Foo", 42, "submissions/aasist.yaml"
    ) is False


def test_verify_pr_passed_false_when_no_our_comment():
    api = MagicMock()
    api.get_discussion_details.return_value = _details("a human comment ✅ all checks passed")
    assert post_merge_badge._verify_pr_passed(
        api, "Org/Foo", 42, "submissions/aasist.yaml"
    ) is False


def test_verify_pr_passed_uses_most_recent_for_path():
    # Oldest-first: a failing re-run after a passing one wins (most recent decides).
    api = MagicMock()
    api.get_discussion_details.return_value = _details(
        _verify_comment("submissions/aasist.yaml", passed=True),
        _verify_comment("submissions/aasist.yaml", passed=False),
    )
    assert post_merge_badge._verify_pr_passed(
        api, "Org/Foo", 42, "submissions/aasist.yaml"
    ) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/ci/test_post_merge_badge_stamp.py -k verify_pr_passed -v`
Expected: FAIL with `AttributeError: … has no attribute '_verify_pr_passed'`.

- [ ] **Step 3: Implement `_verify_pr_passed`**

In `post_merge_badge.py`:

```python
_VERIFY_MARKER = "**speech-spoof-bench ci verify-pr**"
_VERIFY_PASS = "✅ all checks passed"


def _verify_pr_passed(api: HfApi, repo: str, pr: int, path: str) -> bool:
    """True iff our most-recent verify-pr comment that mentions <path> shows
    all checks passed. Reads the existing markdown (no machine sentinel)."""
    try:
        details = api.get_discussion_details(
            repo_id=repo, repo_type="dataset", discussion_num=pr,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("get_discussion_details failed for %s#%d: %s", repo, pr, exc)
        return False
    verdict = False
    for ev in getattr(details, "events", []) or []:
        body = getattr(ev, "content", "") or ""
        if _VERIFY_MARKER in body and f"`{path}`" in body:
            verdict = _VERIFY_PASS in body  # events are oldest-first → last wins
    return verdict
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/ci/test_post_merge_badge_stamp.py -k verify_pr_passed -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/speech_spoof_bench/ci/post_merge_badge.py tests/ci/test_post_merge_badge_stamp.py
git commit -m "feat(ci): gate stamp on passing verify-pr comment"
```

---

## Task 5: `_bot_identity` and `_commit_stamp` — token-bearing write helpers

**Files:**
- Modify: `src/speech_spoof_bench/ci/post_merge_badge.py`
- Test: `tests/ci/test_post_merge_badge_stamp.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/ci/test_post_merge_badge_stamp.py`:

```python
def test_bot_identity_from_whoami(monkeypatch):
    fake_api = MagicMock()
    fake_api.whoami.return_value = {"name": "ssb-bot"}
    monkeypatch.setattr(post_merge_badge, "HfApi", lambda token=None: fake_api)
    monkeypatch.setenv("HF_BOT_TOKEN", "tok")
    assert post_merge_badge._bot_identity() == "ssb-bot"


def test_bot_identity_falls_back_without_token(monkeypatch):
    monkeypatch.delenv("HF_BOT_TOKEN", raising=False)
    assert post_merge_badge._bot_identity() == "ssb-ci-bot"


def test_commit_stamp_uploads_to_main(monkeypatch):
    fake_api = MagicMock()
    monkeypatch.setattr(post_merge_badge, "HfApi", lambda token=None: fake_api)
    monkeypatch.setenv("HF_BOT_TOKEN", "tok")
    post_merge_badge._commit_stamp(
        "Org/Foo", "submissions/aasist.yaml", "content: 1\n", "ci: msg"
    )
    kwargs = fake_api.upload_file.call_args.kwargs
    assert kwargs["path_in_repo"] == "submissions/aasist.yaml"
    assert kwargs["repo_id"] == "Org/Foo"
    assert kwargs["repo_type"] == "dataset"
    assert kwargs["revision"] == "main"
    assert kwargs["commit_message"] == "ci: msg"
    assert kwargs["path_or_fileobj"] == b"content: 1\n"


def test_commit_stamp_skips_without_token(monkeypatch):
    fake_api = MagicMock()
    monkeypatch.setattr(post_merge_badge, "HfApi", lambda token=None: fake_api)
    monkeypatch.delenv("HF_BOT_TOKEN", raising=False)
    post_merge_badge._commit_stamp("Org/Foo", "p", "c", "m")
    fake_api.upload_file.assert_not_called()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/ci/test_post_merge_badge_stamp.py -k "bot_identity or commit_stamp" -v`
Expected: FAIL with `AttributeError: … has no attribute '_bot_identity'`.

- [ ] **Step 3: Implement both helpers**

In `post_merge_badge.py`:

```python
def _bot_identity() -> str:
    """The HF username behind HF_BOT_TOKEN (for `reproduced_by`); falls back to
    a constant if the token is absent or whoami fails."""
    token = os.environ.get("HF_BOT_TOKEN")
    if not token:
        return "ssb-ci-bot"
    try:
        return HfApi(token=token).whoami()["name"]
    except Exception as exc:  # noqa: BLE001
        logger.warning("whoami failed; using fallback identity: %s", exc)
        return "ssb-ci-bot"


def _commit_stamp(repo: str, path: str, content: str, message: str) -> None:
    """Commit the stamped YAML to the dataset repo's main branch. Message
    deliberately carries no `(#N)` suffix (loop guard #2)."""
    token = os.environ.get("HF_BOT_TOKEN")
    if not token:
        logger.warning("HF_BOT_TOKEN not set; skipping stamp commit for %s", path)
        return
    HfApi(token=token).upload_file(
        path_or_fileobj=content.encode("utf-8"),
        path_in_repo=path,
        repo_id=repo,
        repo_type="dataset",
        revision="main",
        commit_message=message,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/ci/test_post_merge_badge_stamp.py -k "bot_identity or commit_stamp" -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/speech_spoof_bench/ci/post_merge_badge.py tests/ci/test_post_merge_badge_stamp.py
git commit -m "feat(ci): add bot-identity and main-branch commit helpers"
```

---

## Task 6: `_maybe_stamp` and `_today_iso` — orchestrate the stamp (Component A2)

**Files:**
- Modify: `src/speech_spoof_bench/ci/post_merge_badge.py`
- Test: `tests/ci/test_post_merge_badge_stamp.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/ci/test_post_merge_badge_stamp.py`:

```python
def _patch_download(monkeypatch, text):
    def fake_dl(repo_id, filename, revision, repo_type):
        import tempfile
        p = Path(tempfile.mkstemp(suffix=".yaml")[1])
        p.write_text(text)
        return str(p)
    monkeypatch.setattr(post_merge_badge, "_download_at_revision", fake_dl)


def test_maybe_stamp_commits_when_gate_passes(monkeypatch):
    _patch_download(monkeypatch, SUBMISSION_EMPTY_REPRO)
    api = MagicMock()
    api.get_discussion_details.return_value = _details(
        _verify_comment("submissions/aasist.yaml", passed=True)
    )
    monkeypatch.setattr(post_merge_badge, "_bot_identity", lambda: "ssb-bot")
    committed = {}
    monkeypatch.setattr(
        post_merge_badge, "_commit_stamp",
        lambda repo, path, content, message: committed.update(
            repo=repo, path=path, content=content, message=message
        ),
    )
    post_merge_badge._maybe_stamp(
        api, "Org/Foo", 42, "submissions/aasist.yaml", "deadbeef", today="2026-05-31"
    )
    assert committed["path"] == "submissions/aasist.yaml"
    assert committed["message"] == "ci: auto-stamp reproduction for submissions/aasist.yaml"
    assert "(#" not in committed["message"]  # loop guard #2
    assert "reproduced_by: ssb-bot" in committed["content"]
    assert "match: scoring" in committed["content"]


def test_maybe_stamp_skips_when_gate_fails(monkeypatch):
    _patch_download(monkeypatch, SUBMISSION_EMPTY_REPRO)
    api = MagicMock()
    api.get_discussion_details.return_value = _details(
        _verify_comment("submissions/aasist.yaml", passed=False)
    )
    called = []
    monkeypatch.setattr(post_merge_badge, "_commit_stamp",
                        lambda *a, **k: called.append(1))
    post_merge_badge._maybe_stamp(
        api, "Org/Foo", 42, "submissions/aasist.yaml", "deadbeef", today="2026-05-31"
    )
    assert called == []


def test_maybe_stamp_skips_when_already_stamped(monkeypatch):
    stamped = SUBMISSION_EMPTY_REPRO.replace(
        "reproduction: {}",
        "reproduction:\n  reproduced_by: x\n  reproduced_at: '2026-05-30'\n"
        "  reproduced_bench_version: 'speech-spoof-bench==0.3.0'\n  match: scoring",
    )
    _patch_download(monkeypatch, stamped)
    api = MagicMock()
    called = []
    monkeypatch.setattr(post_merge_badge, "_commit_stamp",
                        lambda *a, **k: called.append(1))
    post_merge_badge._maybe_stamp(
        api, "Org/Foo", 42, "submissions/aasist.yaml", "deadbeef", today="2026-05-31"
    )
    assert called == []
    api.get_discussion_details.assert_not_called()  # short-circuits before the gate
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/ci/test_post_merge_badge_stamp.py -k maybe_stamp -v`
Expected: FAIL with `AttributeError: … has no attribute '_maybe_stamp'`.

- [ ] **Step 3: Implement `_today_iso` and `_maybe_stamp`**

Add to the imports at the top of `post_merge_badge.py`:

```python
import datetime

from .. import __version__ as _BENCH_VERSION
```

Then add:

```python
def _today_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).date().isoformat()


def _maybe_stamp(
    api: HfApi, repo: str, pr: int, path: str, sha: str, *, today: str
) -> None:
    """Stamp the reproduction block on <path> if it is unstamped and verify-pr
    passed. Guard order: already-stamped (cheap, no network) → verify-pr gate."""
    local = _download_at_revision(repo, path, revision=sha, repo_type="dataset")
    text = Path(local).read_text()
    if _already_stamped(text):
        logger.info("reproduction already set for %s; skip stamp", path)
        return
    if not _verify_pr_passed(api, repo, pr, path):
        logger.info("no passing verify-pr comment for %s; skip stamp", path)
        return
    new_text = _fill_reproduction_block(
        text,
        reproduced_by=_bot_identity(),
        reproduced_at=today,
        bench_version=f"speech-spoof-bench=={_BENCH_VERSION}",
    )
    _commit_stamp(
        repo, path, new_text,
        message=f"ci: auto-stamp reproduction for {path}",
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/ci/test_post_merge_badge_stamp.py -k maybe_stamp -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/speech_spoof_bench/ci/post_merge_badge.py tests/ci/test_post_merge_badge_stamp.py
git commit -m "feat(ci): orchestrate reproduction auto-stamp (_maybe_stamp)"
```

---

## Task 7: Wire `_maybe_stamp` into `run()`

**Files:**
- Modify: `src/speech_spoof_bench/ci/post_merge_badge.py:104-142` (the `run()` function)
- Test: `tests/ci/test_post_merge_badge_stamp.py`

- [ ] **Step 1: Write the failing test**

`_ev`, `_details`, and `_verify_comment` already exist from Task 4. Append the test:

```python
def test_run_stamps_after_badge(monkeypatch, tmp_path):
    from tests.ci.test_post_merge_badge_happy import _eval_yaml, make_api

    api = make_api(
        sha="deadbeefcafe1234",
        parent="parent0000",
        sha_files=["submissions/aasist.yaml", "submissions/README.md"],
        parent_files=["submissions/README.md"],
        events=[_ev(_verify_comment("submissions/aasist.yaml", passed=True))],
    )

    def fake_dl(repo_id, filename, revision, repo_type):
        p = tmp_path / filename.replace("/", "_")
        p.write_text(_eval_yaml() if filename == "eval.yaml" else SUBMISSION_EMPTY_REPRO)
        return str(p)
    monkeypatch.setattr(post_merge_badge, "_download_at_revision", fake_dl)
    monkeypatch.setattr(post_merge_badge, "_post_comment", lambda r, p, b: None)
    monkeypatch.setattr(post_merge_badge, "_bot_identity", lambda: "ssb-bot")

    committed = []
    monkeypatch.setattr(post_merge_badge, "_commit_stamp",
                        lambda repo, path, content, message: committed.append(path))

    rc = post_merge_badge.run(
        repo="Org/ASVspoof2019_LA", pr=42, sha="deadbeefcafe1234",
        api=api, gh_run_url="x", today="2026-05-31",
    )
    assert rc == 0
    assert committed == ["submissions/aasist.yaml"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/ci/test_post_merge_badge_stamp.py::test_run_stamps_after_badge -v`
Expected: FAIL — `run()` does not accept `today` / does not call `_commit_stamp`.

- [ ] **Step 3: Modify `run()`**

Change the signature and loop. The current loop body posts the badge; add the stamp call after the badge `try/except`. Edit `run()` to:

```python
def run(*, repo: str, pr: int, sha: str,
        api: HfApi | None = None,
        gh_run_url: str | None = None,
        today: str | None = None) -> int:
    api = api or HfApi()
    gh_run_url = gh_run_url or os.environ.get(
        "GH_RUN_URL", "https://github.com/lab260ru/speech_spoof_bench/actions"
    )
    today = today or _today_iso()

    paths = _changed_submissions(api, repo, sha)
    if not paths:
        logger.info("no new submissions in %s@%s; nothing to do", repo, sha)
        return 0

    errors = 0
    for path in paths:
        sentinel = _sentinel_for(sha, path)
        if not _already_posted(api, repo, pr, sentinel):
            try:
                local = _download_at_revision(repo, path, revision=sha, repo_type="dataset")
                data = submission.parse_submission(Path(local).read_text())
                dataset_id = data["dataset"]["id"]
                dataset_rev = data["dataset"]["revision"]
                primary = _primary_metric_at(api, dataset_id, dataset_rev)
                body = badge.build_paste_comment(
                    data,
                    arena_url=badge.ARENA_URL,
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
        else:
            logger.info("badge comment already present for %s; skipping", path)

        # Auto-stamp the reproduction block (independent of badge outcome).
        try:
            _maybe_stamp(api, repo, pr, path, sha, today=today)
        except Exception as exc:  # noqa: BLE001
            logger.error("auto-stamp failed for %s: %s", path, exc)
            errors += 1
    return 0 if errors == 0 else 1
```

- [ ] **Step 4: Run the test and the full CI suite**

Run: `pytest tests/ci/test_post_merge_badge_stamp.py::test_run_stamps_after_badge tests/ci/ -v`
Expected: all pass (existing badge tests still green — they have no passing verify-pr comment, so `_maybe_stamp` skips without error).

- [ ] **Step 5: Commit**

```bash
git add src/speech_spoof_bench/ci/post_merge_badge.py tests/ci/test_post_merge_badge_stamp.py
git commit -m "feat(ci): run auto-stamp in post-merge per-submission loop"
```

---

## Task 8: Switch workflows to install the `ci` extra

**Files:**
- Modify: `.github/workflows/post-merge-badge.yml:28`
- Modify: `.github/workflows/verify-hf-pr.yml:28`

- [ ] **Step 1: Update `post-merge-badge.yml`**

Replace the install step's run line:

```yaml
      - name: Install
        run: pip install -e ".[ci]"
```

- [ ] **Step 2: Update `verify-hf-pr.yml`**

Replace the install step's run line:

```yaml
      - name: Install
        run: pip install -e ".[ci]"
```

(Both workflows install the same extra so the `ci` dependency set stays consistent; verify-pr doesn't need ruamel today but pinning both avoids drift.)

- [ ] **Step 3: Verify the extra resolves**

Run: `pip install -e ".[ci]"`
Expected: completes without error.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/post-merge-badge.yml .github/workflows/verify-hf-pr.yml
git commit -m "ci: install [ci] extra in verify and post-merge workflows"
```

---

## Task 9: Document the new behavior

**Files:**
- Modify: `docs/architecture/cicd.md` (the `post-merge-badge.yml` section ~line 91-106, the secrets table ~line 117, and "When CI changes require coordinated edits" ~line 138)
- Modify: `docs/architecture/submission-lifecycle.md` (the `reproduction` description ~line 74-76)

- [ ] **Step 1: Update the post-merge-badge section in `cicd.md`**

After the existing badge-idempotency paragraph, add:

```markdown
**Auto-stamp (option 2).** The same command now *also* fills the `reproduction:` block on
`main` when the merged PR was verified. Per added submission it: re-reads the YAML at the
merge sha; skips if `reproduction.match` is already set (guard #3); requires a passing
`verify-pr` comment for that path in the discussion (`**speech-spoof-bench ci verify-pr**`
+ `✅ all checks passed`); then writes the complete block (`reproduced_by` = bot whoami,
`reproduced_at` = today UTC, `reproduced_bench_version` = installed package version,
`match: scoring`) via a `ruamel.yaml` round-trip and `HfApi.upload_file(... revision="main")`.
The commit message carries **no `(#N)` suffix**, so the re-fired webhook schedules only a
refresh (guard #2); and the stamp *modifies* rather than *adds* the file, so the next
`_changed_submissions` diff is empty (guard #1). Any one guard alone terminates the loop.
```

- [ ] **Step 2: Update the `HF_BOT_TOKEN` row in the secrets table**

Change the `HF_BOT_TOKEN` Purpose cell to:

```markdown
| `HF_BOT_TOKEN` | GitHub Actions secret | `ci/verify_pr.py`, `ci/post_merge_badge.py`, `ci/nightly.py` | HF token to comment on dataset discussions **and (post-merge) commit the auto-stamped `reproduction` block to `main` — requires write access to the dataset repos.** |
```

- [ ] **Step 3: Add a coordinated-edit bullet**

Under "When CI changes require coordinated edits", add:

```markdown
- Change the verify-pr comment header or pass line → update `_VERIFY_MARKER` /
  `_VERIFY_PASS` in `post_merge_badge.py` (the auto-stamp gate parses them).
```

- [ ] **Step 4: Update `submission-lifecycle.md`**

Change the `reproduction` `oneOf` bullet (~line 74-76) to note CI authorship:

```markdown
- `reproduction` is `oneOf`: an **empty `{}`** (as opened by the submitter) **or** a
  fully populated block (`reproduced_by`, `reproduced_at`, `reproduced_bench_version`,
  `match`). It can't be partial. It is filled either by a maintainer at merge **or
  automatically by the post-merge job** (auto-stamp) when the PR passed `verify-pr` — see
  [architecture/cicd.md](../architecture/cicd.md).
```

- [ ] **Step 5: Commit**

```bash
git add docs/architecture/cicd.md docs/architecture/submission-lifecycle.md
git commit -m "docs: describe post-merge reproduction auto-stamp"
```

---

## Task 10: Full-suite green + final verification

**Files:** none (verification only)

- [ ] **Step 1: Run the whole test suite**

Run: `pytest -q`
Expected: all tests pass (no regressions in `tests/ci/` or elsewhere).

- [ ] **Step 2: Confirm the package still imports without the `ci` extra installed**

Run: `python -c "import speech_spoof_bench.ci.post_merge_badge as m; print(hasattr(m, '_maybe_stamp'))"`
Expected: prints `True` (ruamel is imported lazily inside `_fill_reproduction_block`, so module import never requires it).

- [ ] **Step 3: Final commit (if anything outstanding)**

```bash
git status
# commit any stragglers
```

---

## Self-Review Notes (addressed)

- **Spec coverage:** A1 gate → Task 4; A2 stamp/guards → Tasks 3,5,6,7; A3 ruamel/ci extra → Tasks 1,8; trust model (no re-run) → Task 6 (`_maybe_stamp` never calls `reproduce`); doc sync → Task 9; token write-scope precondition → documented in Task 9 (operational check: confirm `HF_BOT_TOKEN` has write on dataset repos before first real merge).
- **Loop guards:** #1 modified-not-added (existing `_changed_submissions`, unchanged), #2 no-`(#N)` message (Task 5/6, asserted in Task 6 test), #3 already-stamped (Task 3, asserted in Task 6 test).
- **Type consistency:** `_maybe_stamp`, `_verify_pr_passed`, `_fill_reproduction_block`, `_already_stamped`, `_bot_identity`, `_commit_stamp`, `_today_iso` signatures match across definition and call sites; `run()` gains `today: str | None = None`.
- **Schema validity:** `_fill_reproduction_block` writes exactly the 4 required keys with `match: "scoring"`; Task 2 asserts `submission.parse_submission` accepts the output.

## Operational precondition (not a code task)

Before the first real merge relies on auto-stamp, confirm `HF_BOT_TOKEN` has **write** permission on the `SpeechAntiSpoofingBenchmarks/*` dataset repos (it previously only needed comment access). A read-only token will surface as a logged `auto-stamp failed … 403` and the model stays hidden until manually stamped.
