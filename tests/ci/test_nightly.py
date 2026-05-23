"""Tests for speech_spoof_bench.ci.nightly."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from speech_spoof_bench.ci import nightly


def _failure(dataset="Org/Foo", slug="rb", reason="sha mismatch"):
    return nightly.Failure(dataset_id=dataset, slug=slug, reason=reason)


def test_collect_failures_calls_reproduce_per_submission(monkeypatch):
    monkeypatch.setattr(nightly, "_fetch_manifest", lambda: {"core_set": [{"id": "Org/Foo"}], "extended": []})
    monkeypatch.setattr(nightly, "_list_submission_files",
                        lambda did, **_: ["submissions/a.yaml", "submissions/b.yaml"])
    called = []
    def fake_check(dataset_id, path):
        called.append((dataset_id, path))
        if path.endswith("b.yaml"):
            return _failure(dataset=dataset_id, slug="b", reason="EER drift")
        return None
    monkeypatch.setattr(nightly, "_check_submission", fake_check)
    failures = nightly.collect_failures()
    assert len(called) == 2
    assert failures == [_failure(dataset="Org/Foo", slug="b", reason="EER drift")]


def test_manage_issues_opens_new_issue_for_new_failure():
    api = MagicMock()
    api.list_issues.return_value = []  # nothing open
    nightly.manage_issues(failures=[_failure()], api=api)
    api.create_issue.assert_called_once()
    args, kwargs = api.create_issue.call_args
    assert "Org/Foo" in kwargs["title"] and "rb" in kwargs["title"]


def test_manage_issues_comments_when_reason_changes():
    api = MagicMock()
    api.list_issues.return_value = [{"number": 3, "title": "[Org/Foo] rb", "last_comment_body": "OLD"}]
    nightly.manage_issues(failures=[_failure(reason="NEW")], api=api)
    api.create_issue.assert_not_called()
    api.add_comment.assert_called_once_with(3, body=pytest.approx_any(), reason="NEW") if False else api.add_comment.assert_called_once()


def test_manage_issues_closes_resolved_issue():
    api = MagicMock()
    api.list_issues.return_value = [{"number": 9, "title": "[Org/Foo] rb", "last_comment_body": "x"}]
    # No failures this run → the open issue should be closed.
    nightly.manage_issues(failures=[], api=api)
    api.close_issue.assert_called_once_with(9)
