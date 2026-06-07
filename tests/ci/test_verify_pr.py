"""Tests for speech_spoof_bench.ci.verify_pr."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from speech_spoof_bench.ci import verify_pr


def _tree(*paths):
    """Mimic HfApi.list_repo_tree output: objects carrying a ``.path``."""
    return [SimpleNamespace(path=p) for p in paths]


def _good_yaml(slug="good-system"):
    sha64 = "0" * 64
    return f"""schema_version: 4
system: {{name: G, slug: {slug}, description: x, code: https://x, checkpoint: https://x, paper: {{arxiv_id: "1", url: https://x, bibtex: "@x{{1, }}"}}}}
dataset: {{id: Org/Foo, revision: abc, split: test}}
scores: {{eer_percent: 0.5, n_trials: 1, n_skipped: 0}}
artifact: {{scores_url: "https://huggingface.co/u/r/resolve/abc1234/.eval_results/Org/Foo/scores.txt", scores_sha256: "{sha64}", bench_version: "speech-spoof-bench==0.1.0"}}
reproduction: {{reproduced_by: M, reproduced_at: 2026-05-23, reproduced_bench_version: "speech-spoof-bench==0.1.0", match: scoring}}
submitter: {{hf_username: u, contact: u@example.com}}
submitted_at: 2026-05-23
"""


def test_verdict_table_columns():
    rows = [
        verify_pr.Verdict("submissions/a.yaml", schema_ok=True, sha_ok=True, eer_ok=True, notes=""),
        verify_pr.Verdict("submissions/b.yaml", schema_ok=False, sha_ok=None, eer_ok=None, notes="schema: missing field"),
    ]
    md = verify_pr.format_markdown(rows, gh_run_url="https://github.com/o/r/actions/runs/1")
    assert "| Submission | Schema | sha256 | EER match | Notes |" in md
    assert "submissions/a.yaml" in md and "submissions/b.yaml" in md
    assert "speech-spoof-bench ci verify-pr" in md
    assert "https://github.com/o/r/actions/runs/1" in md


def test_run_with_one_good_and_one_bad(monkeypatch, tmp_path):
    """End-to-end: fetched PR contains two YAMLs, one passes, one fails."""
    api = MagicMock()
    api.list_repo_tree.side_effect = [
        # main:
        _tree("submissions/a.yaml", "submissions/README.md", "submissions/results_template.yaml"),
        # branch:
        _tree("submissions/a.yaml", "submissions/b.yaml", "submissions/README.md"),
    ]
    # b.yaml is added on the branch; a.yaml is identical to main (skipped).
    def fake_dl(repo_id, filename, revision, repo_type):
        p = tmp_path / filename.replace("/", "_")
        if filename == "submissions/b.yaml":
            p.write_text(_good_yaml("bad-system"))
        else:
            p.write_text(_good_yaml("good-system"))
        return str(p)
    monkeypatch.setattr(verify_pr, "_download_at_revision", fake_dl)
    # Bypass reproduce; assert schema-OK + force EER mismatch on b.yaml.
    monkeypatch.setattr(verify_pr, "_run_scoring_repro", lambda data: (False, "EER mismatch: claimed 0.5 got 0.6"))

    posted = {}
    def fake_post(repo, pr, body):
        posted["repo"] = repo; posted["pr"] = pr; posted["body"] = body
    monkeypatch.setattr(verify_pr, "_post_comment", fake_post)

    rc = verify_pr.run(repo="Org/Foo", pr=42, branch="refs/pr/42",
                       api=api, gh_run_url="https://gh/run")
    assert rc == 1
    assert "submissions/b.yaml" in posted["body"]
    assert "EER mismatch" in posted["body"]
    # The repo listing must be scoped to submissions/, never a full repo tree.
    for _, kwargs in api.list_repo_tree.call_args_list:
        assert kwargs.get("path_in_repo") == "submissions"
    api.list_repo_files.assert_not_called()


def test_run_with_only_passing_submission_exits_zero(monkeypatch, tmp_path):
    api = MagicMock()
    api.list_repo_tree.side_effect = [
        _tree("submissions/README.md"),
        _tree("submissions/a.yaml", "submissions/README.md"),
    ]
    def fake_dl(repo_id, filename, revision, repo_type):
        p = tmp_path / filename.replace("/", "_")
        p.write_text(_good_yaml())
        return str(p)
    monkeypatch.setattr(verify_pr, "_download_at_revision", fake_dl)
    monkeypatch.setattr(verify_pr, "_run_scoring_repro", lambda data: (True, "ok"))
    monkeypatch.setattr(verify_pr, "_post_comment", lambda repo, pr, body: None)
    assert verify_pr.run(repo="Org/Foo", pr=1, branch="refs/pr/1", api=api, gh_run_url="x") == 0


def test_run_with_zero_changed_submissions_exits_zero(monkeypatch):
    api = MagicMock()
    api.list_repo_tree.return_value = _tree("submissions/README.md")
    posted = []
    monkeypatch.setattr(verify_pr, "_post_comment", lambda r, p, b: posted.append(b))
    rc = verify_pr.run(repo="Org/Foo", pr=2, branch="refs/pr/2", api=api, gh_run_url="x")
    assert rc == 0
    assert posted and "no submission changes" in posted[0]


def test_changed_submissions_detects_modified_file(monkeypatch, tmp_path):
    """A submission present on BOTH main and branch but with different content
    must be detected (not only added files)."""
    api = MagicMock()
    api.list_repo_tree.return_value = _tree("submissions/sys.yaml")  # on main AND branch
    main_f = tmp_path / "main.yaml"; main_f.write_text("eer: 24.85\n")
    branch_f = tmp_path / "branch.yaml"; branch_f.write_text("eer: 0.48\n")

    def fake_dl(repo, filename, revision, repo_type):
        return str(branch_f if revision != "main" else main_f)

    monkeypatch.setattr(verify_pr, "_download_at_revision", fake_dl)
    changed = verify_pr._changed_submissions(api, "Org/Foo", "refs/pr/2")
    assert changed == ["submissions/sys.yaml"]


def test_changed_submissions_ignores_identical_file(monkeypatch, tmp_path):
    """Same content on main and branch → not a change."""
    api = MagicMock()
    api.list_repo_tree.return_value = _tree("submissions/sys.yaml")
    same = tmp_path / "same.yaml"; same.write_text("eer: 0.48\n")
    monkeypatch.setattr(verify_pr, "_download_at_revision", lambda *a, **k: str(same))
    assert verify_pr._changed_submissions(api, "Org/Foo", "refs/pr/2") == []
