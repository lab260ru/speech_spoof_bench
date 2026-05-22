"""Submission schema must accept reproduction: {} so submit-time YAMLs parse."""

from __future__ import annotations

import datetime as _dt

import pytest
import yaml

from speech_spoof_bench import submission


_BASE = {
    "schema_version": 4,
    "system": {
        "name": "x",
        "slug": "x",
        "description": "x",
        "code": "https://example.com",
        "checkpoint": "https://example.com",
        "paper": {
            "arxiv_id": "1234.5678",
            "url": "https://arxiv.org/abs/1234.5678",
            "bibtex": "@x{}",
        },
    },
    "dataset": {"id": "Org/A", "revision": "abcdef1", "split": "test"},
    "scores": {"eer_percent": 1.0, "n_trials": 1, "n_skipped": 0},
    "artifact": {
        "scores_url": (
            "https://huggingface.co/Org/x/resolve/abcdef1/.eval_results/"
            "Org/A/scores.txt"
        ),
        "scores_sha256": "0" * 64,
        "bench_version": "speech-spoof-bench==0.1.0",
    },
    "submitter": {"hf_username": "x", "contact": "x@example.com"},
    "submitted_at": _dt.date(2026, 5, 22).isoformat(),
}


def test_empty_reproduction_parses():
    data = dict(_BASE)
    data["reproduction"] = {}
    text = yaml.safe_dump(data)
    out = submission.parse_submission(text)
    assert out["reproduction"] == {}


def test_missing_reproduction_key_parses():
    data = dict(_BASE)  # no 'reproduction' key
    text = yaml.safe_dump(data)
    out = submission.parse_submission(text)
    assert out.get("reproduction", {}) in ({}, None)


def test_filled_reproduction_still_parses():
    data = dict(_BASE)
    data["reproduction"] = {
        "reproduced_by": "Org",
        "reproduced_at": "2026-05-22",
        "reproduced_bench_version": "speech-spoof-bench==0.1.0",
        "match": "scoring",
    }
    text = yaml.safe_dump(data)
    out = submission.parse_submission(text)
    assert out["reproduction"]["match"] == "scoring"


def test_partial_reproduction_rejected():
    data = dict(_BASE)
    data["reproduction"] = {"reproduced_by": "Org"}  # missing other keys
    text = yaml.safe_dump(data)
    with pytest.raises(submission.SubmissionValidationError):
        submission.parse_submission(text)
