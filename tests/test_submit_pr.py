"""open_submission_pr calls HfApi.create_commit with create_pr=True and parent_commit."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from huggingface_hub import CommitOperationAdd

from speech_spoof_bench.submit import open_submission_pr


def test_open_submission_pr_passes_expected_kwargs():
    api = MagicMock()
    commit_info = MagicMock()
    commit_info.pr_url = "https://huggingface.co/datasets/Org/A/discussions/7"
    api.create_commit.return_value = commit_info

    pr_url = open_submission_pr(
        api=api,
        dataset_id="Org/A",
        parent_commit="abcdef1",
        slug="aasist-test",
        yaml_text="schema_version: 4\n",
    )

    assert pr_url == "https://huggingface.co/datasets/Org/A/discussions/7"
    kwargs = api.create_commit.call_args.kwargs
    assert kwargs["repo_id"] == "Org/A"
    assert kwargs["repo_type"] == "dataset"
    assert kwargs["create_pr"] is True
    assert kwargs["parent_commit"] == "abcdef1"

    ops = kwargs["operations"]
    assert len(ops) == 1
    op = ops[0]
    assert isinstance(op, CommitOperationAdd)
    assert op.path_in_repo == "submissions/aasist-test.yaml"


def test_open_submission_pr_propagates_unknown_pr_url():
    """When the HF response lacks `pr_url`, raise so the caller notices."""
    api = MagicMock()
    commit_info = MagicMock(spec=[])  # no pr_url attr
    api.create_commit.return_value = commit_info

    with pytest.raises(RuntimeError, match="PR url"):
        open_submission_pr(
            api=api,
            dataset_id="Org/A",
            parent_commit="abcdef1",
            slug="x",
            yaml_text="x",
        )
