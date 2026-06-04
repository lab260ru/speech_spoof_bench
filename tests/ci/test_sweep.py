"""Tests for speech_spoof_bench.ci.sweep."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from speech_spoof_bench.ci import sweep


def _disc(num, *, is_pr=True, status="open"):
    return SimpleNamespace(num=num, is_pull_request=is_pr, status=status)


def _comment(text):
    return SimpleNamespace(type="comment", content=text)


def _verdict_comment():
    return _comment("**speech-spoof-bench ci verify-pr** — ✅ all checks passed")


def _api(discussions_by_repo, events_by_pr):
    """Build a MagicMock HfApi.

    discussions_by_repo: {repo: [SimpleNamespace(...)]}
    events_by_pr: {(repo, num): [events]}
    """
    api = MagicMock()
    api.get_repo_discussions.side_effect = lambda repo_id, repo_type: list(
        discussions_by_repo.get(repo_id, [])
    )
    api.get_discussion_details.side_effect = lambda repo_id, repo_type, discussion_num: SimpleNamespace(
        events=events_by_pr.get((repo_id, discussion_num), [])
    )
    return api


def test_dispatches_only_verdictless_open_prs():
    api = _api(
        {"Org/A": [_disc(1), _disc(2)]},
        {
            ("Org/A", 1): [_verdict_comment()],            # has verdict → skip
            ("Org/A", 2): [_comment("please review")],     # no verdict → candidate
        },
    )
    sent = []
    rc = sweep.run(datasets=["Org/A"], max_dispatch=10, api=api,
                   dispatch=lambda r, p: sent.append((r, p)))
    assert rc == 0
    assert sent == [("Org/A", 2)]


def test_respects_max_dispatch_one_at_a_time():
    api = _api(
        {"Org/A": [_disc(3), _disc(4), _disc(5)]},
        {  # all three lack a verdict
            ("Org/A", 3): [],
            ("Org/A", 4): [],
            ("Org/A", 5): [],
        },
    )
    sent = []
    sweep.run(datasets=["Org/A"], max_dispatch=1, api=api,
              dispatch=lambda r, p: sent.append((r, p)))
    # Only the first (sorted) candidate is dispatched this run.
    assert sent == [("Org/A", 3)]


def test_skips_closed_and_non_pr_discussions():
    api = _api(
        {"Org/A": [_disc(6, status="closed"), _disc(7, is_pr=False), _disc(8)]},
        {("Org/A", 8): []},
    )
    sent = []
    sweep.run(datasets=["Org/A"], max_dispatch=10, api=api,
              dispatch=lambda r, p: sent.append((r, p)))
    assert sent == [("Org/A", 8)]


def test_dry_run_dispatches_nothing():
    api = _api({"Org/A": [_disc(9)]}, {("Org/A", 9): []})
    sent = []
    sweep.run(datasets=["Org/A"], max_dispatch=5, api=api, dry_run=True,
              dispatch=lambda r, p: sent.append((r, p)))
    assert sent == []


def test_one_bad_repo_does_not_abort_sweep():
    def boom_for_A(repo_id, repo_type):
        if repo_id == "Org/A":
            raise RuntimeError("HF down")
        return [_disc(10)]
    api = MagicMock()
    api.get_repo_discussions.side_effect = boom_for_A
    api.get_discussion_details.side_effect = lambda repo_id, repo_type, discussion_num: SimpleNamespace(events=[])
    sent = []
    rc = sweep.run(datasets=["Org/A", "Org/B"], max_dispatch=10, api=api,
                   dispatch=lambda r, p: sent.append((r, p)))
    assert rc == 0
    assert sent == [("Org/B", 10)]
