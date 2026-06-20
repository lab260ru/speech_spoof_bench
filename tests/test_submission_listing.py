"""Tests for submission.list_submission_files — scoped + retried listing.

The lister must scope to the ``submissions/`` subtree via ``list_repo_tree``
(never a full-repo ``list_repo_files``), pass ``revision`` through, filter out
README/results_template/non-yaml, return [] when the folder doesn't exist, and
retry on a transient HF 429.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError

from speech_spoof_bench import submission


def _tree(*paths):
    return [SimpleNamespace(path=p) for p in paths]


def _429():
    resp = httpx.Response(429, request=httpx.Request("GET", "https://hf.co/x"))
    return HfHubHTTPError("429", response=resp)


def test_scopes_to_submissions_subtree_and_filters():
    api = MagicMock()
    api.list_repo_tree.return_value = _tree(
        "submissions/aasist.yaml",
        "submissions/README.md",
        "submissions/results_template.yaml",
        "submissions/notes.txt",
    )
    out = submission.list_submission_files("Org/Foo", api=api)
    assert out == ["submissions/aasist.yaml"]
    api.list_repo_files.assert_not_called()
    _, kwargs = api.list_repo_tree.call_args
    assert kwargs["path_in_repo"] == "submissions"
    assert kwargs["repo_type"] == "dataset"


def test_passes_revision_through():
    api = MagicMock()
    api.list_repo_tree.return_value = _tree("submissions/a.yaml")
    submission.list_submission_files("Org/Foo", revision="refs/pr/7", api=api)
    _, kwargs = api.list_repo_tree.call_args
    assert kwargs["revision"] == "refs/pr/7"


def test_missing_submissions_dir_returns_empty():
    api = MagicMock()
    api.list_repo_tree.side_effect = EntryNotFoundError("no submissions/")
    assert submission.list_submission_files("Org/New", api=api) == []


def test_uses_retry_on_429_wrapper(monkeypatch):
    """list_submission_files routes its HF listing through retry_on_429 so a
    transient 429 self-heals rather than crashing the caller (e.g. nightly)."""
    api = MagicMock()
    calls = {"n": 0}

    def flaky(repo_id, path_in_repo, recursive, revision, repo_type):
        calls["n"] += 1
        if calls["n"] < 2:
            raise _429()
        return _tree("submissions/a.yaml")

    api.list_repo_tree.side_effect = flaky

    seen = {}

    def fast_retry(fn, *a, **k):
        seen["used"] = True
        from speech_spoof_bench.ci._hf_retry import retry_on_429
        return retry_on_429(fn, *a, _base=0.0, _sleep=lambda s: None, **k)

    monkeypatch.setattr(submission, "retry_on_429", fast_retry)

    out = submission.list_submission_files("Org/Foo", api=api)
    assert out == ["submissions/a.yaml"]
    assert calls["n"] == 2 and seen.get("used")
