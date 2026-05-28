"""Edge cases: no token / no changes / primary metric missing."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from speech_spoof_bench.ci import post_merge_badge

from .test_post_merge_badge_happy import _good_yaml, _eval_yaml, make_api


def test_no_submission_changes_returns_zero_no_comment(monkeypatch):
    # Merge added nothing under submissions/ — sha and parent both have only README.
    api = make_api(
        sha="abc1234",
        parent="parent0000",
        sha_files=["submissions/README.md"],
        parent_files=["submissions/README.md"],
    )
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
    api = make_api(
        sha="abc1234",
        parent="parent0000",
        sha_files=["submissions/aasist.yaml", "submissions/README.md"],
        parent_files=["submissions/README.md"],
    )

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
    api = make_api(
        sha="abc1234",
        parent="parent0000",
        sha_files=["submissions/aasist.yaml", "submissions/README.md"],
        parent_files=["submissions/README.md"],
    )

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
