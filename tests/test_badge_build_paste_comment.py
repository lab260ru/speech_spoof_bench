"""build_paste_comment renders the full Markdown body."""
from __future__ import annotations

import pytest

from speech_spoof_bench import badge

_ARENA_URL = "https://huggingface.co/spaces/Org/Arena"

_SUBMISSION = {
    "schema_version": 4,
    "system": {
        "name": "AASIST",
        "slug": "aasist",
        "description": "x",
        "code": "https://x",
        "checkpoint": "https://x",
        "paper": {"arxiv_id": "2110.01200", "url": "https://x", "bibtex": "@x{1, }"},
    },
    "dataset": {"id": "Org/ASVspoof2019_LA", "revision": "abc1234", "split": "test"},
    "scores": {"eer_percent": 1.23, "n_trials": 71237, "n_skipped": 0},
    "artifact": {
        "scores_url": "https://huggingface.co/u/r/resolve/abc1234/.eval_results/Org/ASVspoof2019_LA/scores.txt",
        "scores_sha256": "0" * 64,
        "bench_version": "speech-spoof-bench==0.1.0",
    },
    "submitter": {"hf_username": "u", "contact": "u@example.com"},
    "submitted_at": "2026-05-23",
}


def _build():
    return badge.build_paste_comment(
        _SUBMISSION,
        arena_url=_ARENA_URL,
        dataset_canonical_id="Org/ASVspoof2019_LA",
        primary_metric="eer_percent",
        submission_path="submissions/aasist.yaml",
        merge_sha="deadbeefcafe1234",
        gh_run_url="https://github.com/lab260ru/speech_spoof_bench/actions/runs/9",
    )


def test_includes_sentinel_with_sha_and_path():
    body = _build()
    assert "<!-- ssb:badge --> sha=deadbeefcafe1234 path=submissions/aasist.yaml" in body


def test_includes_result_yaml_block():
    body = _build()
    assert "schema_version: 1" in body
    assert "slug: aasist" in body


def test_includes_upload_one_liner():
    body = _build()
    assert "huggingface-cli upload" in body
    assert ".eval_results/Org/ASVspoof2019_LA/result.yaml" in body


def test_includes_shields_url_with_correct_encoding():
    body = _build()
    # Dataset underscores doubled, % encoded as %25, value baked in.
    assert "https://img.shields.io/badge/EER%25%20on%20ASVspoof2019__LA-1.23%25-brightgreen" in body


def test_badge_links_back_to_arena_with_slug():
    body = _build()
    assert f"]({_ARENA_URL}?system=aasist)" in body


def test_includes_ci_run_footer():
    body = _build()
    assert "https://github.com/lab260ru/speech_spoof_bench/actions/runs/9" in body


def test_raises_when_primary_metric_absent_from_scores():
    sub = {**_SUBMISSION, "scores": {"n_trials": 1, "n_skipped": 0}}
    with pytest.raises(badge.BadgeError):
        badge.build_paste_comment(
            sub, arena_url=_ARENA_URL,
            dataset_canonical_id="Org/Foo", primary_metric="eer_percent",
            submission_path="submissions/x.yaml", merge_sha="abc1234",
            gh_run_url="https://gh/run",
        )


def test_uses_color_for_high_eer():
    sub = {**_SUBMISSION, "scores": {"eer_percent": 20.0, "n_trials": 1, "n_skipped": 0}}
    body = badge.build_paste_comment(
        sub, arena_url=_ARENA_URL,
        dataset_canonical_id="Org/ASVspoof2019_LA", primary_metric="eer_percent",
        submission_path="submissions/x.yaml", merge_sha="abc1234",
        gh_run_url="https://gh/run",
    )
    assert "-lightgrey)" in body


def test_long_float_value_rounded_in_badge():
    """A high-precision metric value is rounded to 2 decimals for display."""
    sub = {**_SUBMISSION,
           "scores": {"eer_percent": 49.870836165873556, "n_trials": 1, "n_skipped": 0}}
    body = badge.build_paste_comment(
        sub, arena_url=_ARENA_URL,
        dataset_canonical_id="Org/ASVspoof2019_LA", primary_metric="eer_percent",
        submission_path="submissions/x.yaml", merge_sha="abc1234",
        gh_run_url="https://gh/run",
    )
    # badge shows the rounded 49.87, not the raw float
    assert "49.87%25" in body
    assert "49.870836%25" not in body
    # but result.yaml keeps full precision
    assert "49.870836165873556" in body


@pytest.mark.parametrize("value,expected", [
    (49.870836165873556, "49.87"),
    (1.23, "1.23"),
    (1.20, "1.2"),
    (1.0, "1"),
    (0.0, "0"),
    (3.14159, "3.14"),
    (99.99, "99.99"),
])
def test_fmt_value(value, expected):
    assert badge._fmt_value(value) == expected


def test_comment_includes_tier_and_rank_endpoint_badges():
    body = _build()
    host = "speechantispoofingbenchmarks-speechantispoofingarena.hf.space"
    assert f"https://img.shields.io/endpoint?url=https://{host}/badge/aasist/tier.json" in body
    assert f"https://img.shields.io/endpoint?url=https://{host}/badge/aasist/rank.json" in body
    assert body.count("?system=aasist)") >= 3  # eer + tier + rank click targets


def test_endpoint_badge_md_builder():
    md = badge._endpoint_badge_md("arena tier", "aasist", "tier")
    assert md.startswith("[![arena tier](https://img.shields.io/endpoint?url=")
    assert "/badge/aasist/tier.json" in md
    assert md.endswith(f"({badge.ARENA_URL}?system=aasist)")
