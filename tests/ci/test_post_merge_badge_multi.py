"""Two new submission YAMLs in one merge sha → two comments."""
from __future__ import annotations

from unittest.mock import MagicMock

from speech_spoof_bench.ci import post_merge_badge

from .test_post_merge_badge_happy import _good_yaml, _eval_yaml, make_api


def test_two_new_submissions_two_comments(monkeypatch, tmp_path):
    api = make_api(
        sha="abc1234",
        parent="parent0000",
        sha_files=["submissions/aasist.yaml", "submissions/rawnet.yaml", "submissions/README.md"],
        parent_files=["submissions/README.md"],
    )

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
