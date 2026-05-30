"""Regression: _changed_submissions must diff the merge sha against its PARENT,
not against current main.

The earlier implementation did `candidates - main_files`, which is always empty
post-merge because the merged file is already on main. The fix diffs against the
parent commit instead.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from speech_spoof_bench.ci import post_merge_badge


def _commit(cid):
    c = MagicMock()
    c.commit_id = cid
    return c


def test_diffs_against_parent_not_main(monkeypatch):
    """File present at both sha and main, absent at parent → detected as added."""
    api = MagicMock()
    sha, parent = "mergesha01", "parentsha0"

    def files(repo_id, revision=None, repo_type=None):
        if revision == parent:
            return ["submissions/README.md"]
        # sha and current-main (revision=None) both already have the new file
        return ["submissions/new.yaml", "submissions/README.md"]

    monkeypatch.setattr(post_merge_badge.hf_fetch, "list_repo_files", files)
    api.list_repo_commits.return_value = [_commit(sha), _commit(parent)]

    assert post_merge_badge._changed_submissions(api, "Org/Foo", sha) == [
        "submissions/new.yaml"
    ]


def test_no_addition_when_file_already_at_parent(monkeypatch, tmp_path):
    """File present at parent too → not 'added' by this merge."""
    api = MagicMock()
    sha, parent = "mergesha01", "parentsha0"

    def files(repo_id, revision=None, repo_type=None):
        return ["submissions/existing.yaml", "submissions/README.md"]

    monkeypatch.setattr(post_merge_badge.hf_fetch, "list_repo_files", files)
    api.list_repo_commits.return_value = [_commit(sha), _commit(parent)]

    def fake_download(repo_id, filename, revision, repo_type):
        p = tmp_path / f"{revision}_{filename.replace('/', '_')}"
        p.write_text("same\n")
        return str(p)

    monkeypatch.setattr(post_merge_badge, "_download_at_revision", fake_download)

    assert post_merge_badge._changed_submissions(api, "Org/Foo", sha) == []


def test_first_commit_no_parent_treats_all_as_added(monkeypatch):
    """When sha is the repo's first commit, every candidate counts as added."""
    api = MagicMock()
    sha = "firstcommit"

    monkeypatch.setattr(
        post_merge_badge.hf_fetch,
        "list_repo_files",
        lambda repo_id, revision=None, repo_type=None: [
            "submissions/a.yaml", "submissions/README.md"
        ],
    )
    api.list_repo_commits.return_value = [_commit(sha)]  # no parent

    assert post_merge_badge._changed_submissions(api, "Org/Foo", sha) == [
        "submissions/a.yaml"
    ]


def test_content_edit_against_parent_counts_as_changed(monkeypatch, tmp_path):
    api = MagicMock()
    sha, parent = "mergesha01", "parentsha0"

    monkeypatch.setattr(
        post_merge_badge.hf_fetch,
        "list_repo_files",
        lambda repo_id, revision=None, repo_type=None: [
            "submissions/a.yaml", "submissions/README.md"
        ],
    )
    api.list_repo_commits.return_value = [_commit(sha), _commit(parent)]

    def fake_download(repo_id, filename, revision, repo_type):
        p = tmp_path / f"{revision}_{filename.replace('/', '_')}"
        p.write_text("merge\n" if revision == sha else "parent\n")
        return str(p)

    monkeypatch.setattr(post_merge_badge, "_download_at_revision", fake_download)

    assert post_merge_badge._changed_submissions(api, "Org/Foo", sha) == [
        "submissions/a.yaml"
    ]
