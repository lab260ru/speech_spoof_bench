"""build_submission_payload merges result.yaml + meta into a v4 submission."""

from __future__ import annotations

import yaml

from speech_spoof_bench import submission
from speech_spoof_bench.submit import build_submission_payload


_RESULT = {
    "schema_version": 4,
    "system": {
        "name": "unknown", "slug": None, "description": None,
        "code": None, "checkpoint": None, "paper": None,
    },
    "dataset": {"id": "Org/A", "revision": "abcdef1", "split": "test"},
    "scores": {"eer_percent": 1.234, "n_trials": 100, "n_skipped": 0},
    "artifact": {
        "scores_url": None,
        "scores_sha256": "f" * 64,
        "bench_version": "speech-spoof-bench==0.1.0",
    },
    "reproduction": {},
    "submitter": {},
    "submitted_at": None,
    "notes": None,
}


_META = {
    "system": {
        "name": "AASIST",
        "slug": "aasist-test",
        "description": "AASIST desc",
        "code": "https://github.com/clovaai/aasist",
        "checkpoint": "https://huggingface.co/owner/repo",
        "paper": {
            "arxiv_id": "2110.01200",
            "url": "https://arxiv.org/abs/2110.01200",
            "bibtex": "@inproceedings{jung2022aasist}",
        },
    },
    "notes": "from meta",
}


def _build():
    return build_submission_payload(
        result_yaml=_RESULT,
        meta=_META,
        scores_url=(
            "https://huggingface.co/owner/repo/resolve/"
            "1234567890abcdef1234567890abcdef12345678/.eval_results/Org/A/scores.txt"
        ),
        scores_sha256="f" * 64,
        hf_username="kborodin",
        contact="k@example.com",
        submitted_at="2026-05-22",
    )


def test_payload_parses_against_submission_schema():
    payload = _build()
    # Round-trip through YAML for realism.
    submission.parse_submission(yaml.safe_dump(payload))


def test_payload_system_block_mirrors_meta():
    payload = _build()
    assert payload["system"]["slug"] == "aasist-test"
    assert payload["system"]["name"] == "AASIST"
    assert payload["system"]["paper"]["arxiv_id"] == "2110.01200"


def test_payload_dataset_block_from_result():
    payload = _build()
    assert payload["dataset"] == {
        "id": "Org/A", "revision": "abcdef1", "split": "test",
    }


def test_payload_artifact_block():
    payload = _build()
    assert payload["artifact"]["scores_url"].endswith("/scores.txt")
    assert payload["artifact"]["scores_sha256"] == "f" * 64
    assert payload["artifact"]["bench_version"] == "speech-spoof-bench==0.1.0"


def test_payload_reproduction_empty():
    payload = _build()
    assert payload["reproduction"] == {}


def test_payload_submitter_from_flags():
    payload = _build()
    assert payload["submitter"] == {"hf_username": "kborodin", "contact": "k@example.com"}


def test_payload_submitted_at_from_arg():
    payload = _build()
    assert payload["submitted_at"] == "2026-05-22"


def test_payload_notes_from_meta():
    payload = _build()
    assert payload["notes"] == "from meta"


def test_payload_omits_notes_when_meta_lacks_it():
    meta = {"system": _META["system"]}  # no notes
    payload = build_submission_payload(
        result_yaml=_RESULT,
        meta=meta,
        scores_url="https://huggingface.co/o/r/resolve/abcdef1/x",
        scores_sha256="f" * 64,
        hf_username="u", contact="c",
        submitted_at="2026-05-22",
    )
    assert "notes" not in payload or payload["notes"] is None
