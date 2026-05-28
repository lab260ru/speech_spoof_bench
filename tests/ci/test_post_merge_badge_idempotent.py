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
