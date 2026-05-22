"""Tests for the HF resolve-URL parser and download wrapper."""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import patch

import pytest

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
