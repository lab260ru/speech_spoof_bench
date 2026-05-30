"""Tests for speech_spoof_bench.ci.nightly."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from speech_spoof_bench.ci import nightly


def _failure(dataset="Org/Foo", slug="rb", reason="sha mismatch"):
    return nightly.Failure(dataset_id=dataset, slug=slug, reason=reason)


def _sub(slug="rb", sha="sha1", rev="rev1"):
    return {
        "system": {"slug": slug},
        "artifact": {"scores_sha256": sha},
        "dataset": {"revision": rev},
    }


def _wire(monkeypatch, subs, check_result):
    """subs: {path: submission_dict}; check_result: callable(did, data)->Failure|None."""
    monkeypatch.setattr(nightly, "_fetch_manifest",
                        lambda: {"core_set": [{"id": "Org/Foo"}], "extended": []})
    monkeypatch.setattr(nightly, "_list_submission_files",
                        lambda did, **_: list(subs.keys()))
    monkeypatch.setattr(nightly, "fetch_submission", lambda did, p: subs[p])
    monkeypatch.setattr(nightly, "_check_submission_data", check_result)


def test_collect_failures_verifies_and_records_green(monkeypatch, tmp_path):
    from speech_spoof_bench.ci import green_store
    store_path = tmp_path / "green.json"
    monkeypatch.setattr(green_store, "DEFAULT_STORE_PATH", store_path)
    subs = {"submissions/a.yaml": _sub("a"), "submissions/b.yaml": _sub("b")}
    called = []
    def check(did, data):
        called.append(data["system"]["slug"])
        return nightly.Failure(did, "b", "EER drift") if data["system"]["slug"] == "b" else None
    _wire(monkeypatch, subs, check)

    failures = nightly.collect_failures()
    assert set(called) == {"a", "b"}
    assert failures == [nightly.Failure("Org/Foo", "b", "EER drift")]
    saved = green_store.load(store_path)
    assert green_store.is_green(saved, "Org/Foo", "a", "sha1", "rev1") is True
    assert green_store.is_green(saved, "Org/Foo", "b", "sha1", "rev1") is False


def test_collect_failures_skips_green_submission(monkeypatch, tmp_path):
    from speech_spoof_bench.ci import green_store
    store_path = tmp_path / "green.json"
    monkeypatch.setattr(green_store, "DEFAULT_STORE_PATH", store_path)
    seed = {}
    green_store.record_green(seed, "Org/Foo", "a", "sha1", "rev1")
    green_store.save(seed, store_path)

    subs = {"submissions/a.yaml": _sub("a")}
    called = []
    _wire(monkeypatch, subs, lambda did, data: called.append(data) or None)

    failures = nightly.collect_failures()
    assert failures == []
    assert called == []  # skipped — reproduce never ran


def test_collect_failures_full_ignores_store(monkeypatch, tmp_path):
    from speech_spoof_bench.ci import green_store
    store_path = tmp_path / "green.json"
    monkeypatch.setattr(green_store, "DEFAULT_STORE_PATH", store_path)
    seed = {}
    green_store.record_green(seed, "Org/Foo", "a", "sha1", "rev1")
    green_store.save(seed, store_path)

    subs = {"submissions/a.yaml": _sub("a")}
    called = []
    _wire(monkeypatch, subs, lambda did, data: called.append(data) or None)

    nightly.collect_failures(full=True)
    assert len(called) == 1  # --full re-verifies despite the green entry


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
