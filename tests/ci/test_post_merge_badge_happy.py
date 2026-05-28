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


def _commit(cid):
    c = MagicMock()
    c.commit_id = cid
    return c


def make_api(*, sha, parent, sha_files, parent_files, events=None):
    """Build a mock HfApi modelling the sha-vs-parent diff path.

    list_repo_files responds by `revision` kwarg; list_repo_commits returns
    [<sha>, <parent>] newest-first so `_parent_sha` resolves correctly.
    """
    api = MagicMock()

    def files(repo_id, revision=None, repo_type=None):
        if revision == sha:
            return list(sha_files)
        if revision == parent:
            return list(parent_files)
        return list(parent_files)  # default = current main == parent here

    api.list_repo_files.side_effect = files
    api.list_repo_commits.return_value = [_commit(sha), _commit(parent)]
    api.get_discussion_details.return_value = MagicMock(events=events or [])
    return api


def test_one_new_submission_posts_one_comment(monkeypatch, tmp_path):
    api = make_api(
        sha="deadbeefcafe1234",
        parent="parent0000",
        sha_files=["submissions/aasist.yaml", "submissions/README.md"],
        parent_files=["submissions/README.md"],
    )

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
