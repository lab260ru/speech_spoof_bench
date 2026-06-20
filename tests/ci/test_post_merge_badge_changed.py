"""Regression: _changed_submissions must diff the merge sha against its PARENT,
not against current main.

The earlier implementation did `candidates - main_files`, which is always empty
post-merge because the merged file is already on main. The fix diffs against the
parent commit instead.

The listing is scoped to the ``submissions/`` subtree via ``list_repo_tree``
(not a full-repo ``list_repo_files``), so these tests mock the scoped call.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from huggingface_hub.errors import EntryNotFoundError

from speech_spoof_bench.ci import post_merge_badge


def _commit(cid):
    c = MagicMock()
    c.commit_id = cid
    return c


def _tree(*paths):
    return [SimpleNamespace(path=p) for p in paths]


def test_diffs_against_parent_not_main():
    """File present at both sha and main, absent at parent → detected as added."""
    api = MagicMock()
    sha, parent = "mergesha01", "parentsha0"

    def tree(repo_id, path_in_repo=None, recursive=None, revision=None, repo_type=None):
        if revision == parent:
            return _tree("submissions/README.md")
        # sha already has the new file
        return _tree("submissions/new.yaml", "submissions/README.md")

    api.list_repo_tree.side_effect = tree
    api.list_repo_commits.return_value = [_commit(sha), _commit(parent)]

    assert post_merge_badge._changed_submissions(api, "Org/Foo", sha) == [
        "submissions/new.yaml"
    ]
    api.list_repo_files.assert_not_called()
    for _, kwargs in api.list_repo_tree.call_args_list:
        assert kwargs.get("path_in_repo") == "submissions"


def test_no_addition_when_file_already_at_parent():
    """File present at parent too → not 'added' by this merge."""
    api = MagicMock()
    sha, parent = "mergesha01", "parentsha0"

    api.list_repo_tree.side_effect = lambda *a, **k: _tree(
        "submissions/existing.yaml", "submissions/README.md"
    )
    api.list_repo_commits.return_value = [_commit(sha), _commit(parent)]

    assert post_merge_badge._changed_submissions(api, "Org/Foo", sha) == []


def test_first_commit_no_parent_treats_all_as_added():
    """When sha is the repo's first commit, every candidate counts as added."""
    api = MagicMock()
    sha = "firstcommit"

    api.list_repo_tree.side_effect = lambda *a, **k: _tree(
        "submissions/a.yaml", "submissions/README.md"
    )
    api.list_repo_commits.return_value = [_commit(sha)]  # no parent

    assert post_merge_badge._changed_submissions(api, "Org/Foo", sha) == [
        "submissions/a.yaml"
    ]


def test_missing_submissions_dir_at_parent_treats_sha_files_as_added():
    """A merge that introduces the submissions/ folder: listing at the parent
    raises EntryNotFoundError (no folder yet) → every sha submission is added."""
    api = MagicMock()
    sha, parent = "mergesha01", "parentsha0"

    def tree(repo_id, path_in_repo=None, recursive=None, revision=None, repo_type=None):
        if revision == parent:
            raise EntryNotFoundError("submissions/ does not exist at parent")
        return _tree("submissions/first.yaml")

    api.list_repo_tree.side_effect = tree
    api.list_repo_commits.return_value = [_commit(sha), _commit(parent)]

    assert post_merge_badge._changed_submissions(api, "Org/Foo", sha) == [
        "submissions/first.yaml"
    ]
