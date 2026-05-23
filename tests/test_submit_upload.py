"""upload_scores pushes scores.txt to <model-repo>/.eval_results/<canonical_id>/."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from speech_spoof_bench.submit import upload_scores


def _fake_api_returning_oid(oid: str):
    api = MagicMock()
    info = MagicMock()
    info.oid = oid
    api.upload_file.return_value = info
    return api


def test_upload_scores_calls_upload_file_with_correct_args(tmp_path: Path):
    scores = tmp_path / "scores.txt"
    scores.write_text("x 1.0\n")
    api = _fake_api_returning_oid("abcdef1234567890abcdef1234567890abcdef12")

    url, oid = upload_scores(
        api=api,
        model_repo="Org/random-baseline",
        dataset_canonical_id="Org/ASVspoof2019_LA",
        local_path=scores,
    )

    api.upload_file.assert_called_once()
    kwargs = api.upload_file.call_args.kwargs
    assert kwargs["path_or_fileobj"] == str(scores)
    assert kwargs["path_in_repo"] == ".eval_results/Org/ASVspoof2019_LA/scores.txt"
    assert kwargs["repo_id"] == "Org/random-baseline"
    assert kwargs["repo_type"] == "model"
    assert "commit_message" in kwargs


def test_upload_scores_returns_pinned_url(tmp_path: Path):
    scores = tmp_path / "scores.txt"
    scores.write_text("x 1.0\n")
    oid = "abcdef1234567890abcdef1234567890abcdef12"
    api = _fake_api_returning_oid(oid)

    url, returned_oid = upload_scores(
        api=api,
        model_repo="Org/random-baseline",
        dataset_canonical_id="Org/ASVspoof2019_LA",
        local_path=scores,
    )

    assert returned_oid == oid
    assert url == (
        f"https://huggingface.co/Org/random-baseline/resolve/{oid}/"
        ".eval_results/Org/ASVspoof2019_LA/scores.txt"
    )


def test_upload_scores_missing_local_file(tmp_path: Path):
    api = _fake_api_returning_oid("a" * 40)
    with pytest.raises(FileNotFoundError):
        upload_scores(
            api=api,
            model_repo="Org/x",
            dataset_canonical_id="Org/A",
            local_path=tmp_path / "nope.txt",
        )
