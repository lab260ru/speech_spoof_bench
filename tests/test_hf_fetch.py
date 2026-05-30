"""Tests for the HF resolve-URL parser and download wrapper."""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import patch

import pytest
import requests

from speech_spoof_bench import hf_fetch


VALID_URL = (
    "https://huggingface.co/SpeechAntiSpoofingBenchmarks/random-baseline-asas/"
    "resolve/f63c30bade6e2d059b2e805dea7a807f2f57e99a/"
    ".eval_results/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA/scores.txt"
)


def test_parse_valid_url():
    repo, sha, path = hf_fetch.parse_hf_resolve_url(VALID_URL)
    assert repo == "SpeechAntiSpoofingBenchmarks/random-baseline-asas"
    assert sha == "f63c30bade6e2d059b2e805dea7a807f2f57e99a"
    assert path == ".eval_results/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA/scores.txt"


def test_parse_rejects_main_ref():
    bad = VALID_URL.replace("/resolve/f63c30bade6e2d059b2e805dea7a807f2f57e99a/", "/resolve/main/")
    with pytest.raises(ValueError, match="commit-pinned"):
        hf_fetch.parse_hf_resolve_url(bad)


def test_parse_rejects_non_hf():
    with pytest.raises(ValueError):
        hf_fetch.parse_hf_resolve_url("https://example.com/file.txt")


def test_download_passes_hf_token(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "tok_xyz")
    fake_file = tmp_path / "scores.txt"
    fake_file.write_text("LA_E_0001 0.5\n")

    captured = {}
    def fake_download(**kwargs):
        captured.update(kwargs)
        return str(fake_file)

    with patch.object(hf_fetch, "hf_hub_download", side_effect=fake_download):
        local, sha = hf_fetch.download(VALID_URL)

    assert captured["repo_id"] == "SpeechAntiSpoofingBenchmarks/random-baseline-asas"
    assert captured["revision"] == "f63c30bade6e2d059b2e805dea7a807f2f57e99a"
    assert captured["token"] == "tok_xyz"
    assert captured["repo_type"] == "model"
    assert local == fake_file
    assert sha == hashlib.sha256(fake_file.read_bytes()).hexdigest()


def test_download_without_token(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    fake_file = tmp_path / "scores.txt"
    fake_file.write_text("x")
    captured = {}
    def fake_download(**kwargs):
        captured.update(kwargs)
        return str(fake_file)
    with patch.object(hf_fetch, "hf_hub_download", side_effect=fake_download):
        hf_fetch.download(VALID_URL)
    assert captured["token"] is None


def test_hub_download_retries_transient_failure_and_sets_etag_timeout(tmp_path):
    fake_file = tmp_path / "file.txt"
    fake_file.write_text("ok")
    calls = []

    def flaky_download(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise requests.Timeout("slow")
        return str(fake_file)

    with patch.object(hf_fetch, "hf_hub_download", side_effect=flaky_download):
        local = hf_fetch.hub_download(
            repo_id="Org/Ds",
            filename="eval.yaml",
            repo_type="dataset",
            attempts=2,
            timeout=3,
            sleep=0,
        )

    assert local == fake_file
    assert len(calls) == 2
    assert calls[0]["etag_timeout"] == 3
    assert calls[1]["etag_timeout"] == 3


def test_repo_sha_uses_bounded_request(monkeypatch):
    captured = {}

    class Response:
        headers = {}
        links = {}

        def raise_for_status(self):
            return None

        def json(self):
            return {"sha": "abc123"}

    def fake_get(url, *, headers, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setenv("HF_TOKEN", "tok_xyz")
    monkeypatch.setattr(hf_fetch.requests, "get", fake_get)

    assert hf_fetch.repo_sha("Org/Ds", repo_type="dataset", timeout=7) == "abc123"
    assert captured["url"] == "https://huggingface.co/api/datasets/Org/Ds"
    assert captured["headers"]["authorization"] == "Bearer tok_xyz"
    assert captured["timeout"] == 7


def test_list_repo_files_uses_bounded_tree_api_and_paginates(monkeypatch):
    seen = []

    class Response:
        def __init__(self, payload, next_url=None):
            self._payload = payload
            self.links = {"next": {"url": next_url}} if next_url else {}

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    pages = [
        Response(
            [
                {"type": "directory", "path": "submissions"},
                {"type": "file", "path": "submissions/a.yaml"},
            ],
            next_url="https://huggingface.co/api/next-page",
        ),
        Response([{"type": "file", "path": "README.md"}]),
    ]

    def fake_get(url, *, headers, timeout):
        seen.append((url, timeout))
        return pages.pop(0)

    monkeypatch.setattr(hf_fetch.requests, "get", fake_get)

    files = hf_fetch.list_repo_files(
        "Org/Ds",
        repo_type="dataset",
        revision="refs/pr/2",
        timeout=5,
    )

    assert files == ["submissions/a.yaml", "README.md"]
    assert seen == [
        ("https://huggingface.co/api/datasets/Org/Ds/tree/refs%2Fpr%2F2?recursive=true", 5),
        ("https://huggingface.co/api/next-page", 5),
    ]
