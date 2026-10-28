"""post-merge-badge re-runs: skip an up-to-date comment, edit a stale one
(e.g. a badge URL fixed after the comment was posted), post when absent."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from speech_spoof_bench.ci import post_merge_badge as pmb

SHA = "deadbeef"
PATH = "submissions/m1.yaml"


def _body(url: str, run: str) -> str:
    return (f"badge {url}\n\n<!-- ssb:badge --> sha={SHA} path={PATH}\n\n"
            f"---\n_🤖 [view CI run]({run})_\n")


@pytest.fixture
def harness(monkeypatch):
    calls = {"post": [], "edit": []}
    monkeypatch.setattr(pmb, "_changed_submissions", lambda api, repo, sha: [PATH])
    monkeypatch.setattr(pmb, "_build_body", lambda api, repo, path, sha, run: _body("NEW", run))
    monkeypatch.setattr(pmb, "_post_comment", lambda repo, pr, body: calls["post"].append(body))
    monkeypatch.setattr(pmb, "_edit_comment",
                        lambda repo, pr, cid, body: calls["edit"].append((cid, body)))

    def with_events(*contents):
        events = [SimpleNamespace(type="comment", id=f"c{i}", content=c)
                  for i, c in enumerate(contents)]
        api = SimpleNamespace(get_discussion_details=lambda **kw: SimpleNamespace(events=events))
        return api
    return calls, with_events


def test_posts_when_absent(harness):
    calls, with_events = harness
    assert pmb.run(repo="Org/Foo", pr=1, sha=SHA, api=with_events(), gh_run_url="r2") == 0
    assert len(calls["post"]) == 1 and not calls["edit"]


def test_skips_when_only_run_link_differs(harness):
    calls, with_events = harness
    api = with_events(_body("NEW", "r1"))
    assert pmb.run(repo="Org/Foo", pr=1, sha=SHA, api=api, gh_run_url="r2") == 0
    assert not calls["post"] and not calls["edit"]


def test_edits_stale_comment_in_place(harness):
    calls, with_events = harness
    api = with_events("unrelated", _body("OLD", "r1"))
    assert pmb.run(repo="Org/Foo", pr=1, sha=SHA, api=api, gh_run_url="r2") == 0
    assert not calls["post"]
    assert calls["edit"] == [("c1", _body("NEW", "r2"))]
