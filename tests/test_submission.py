import json
import textwrap

import pytest
import yaml

from speech_spoof_bench.submission import (
    SubmissionValidationError,
    load_submission_schema,
    parse_submission,
)


VALID_YAML = textwrap.dedent("""
schema_version: 4
system:
  name: random-baseline
  slug: random-baseline
  description: stub
  code: https://github.com/example/x
  checkpoint: https://huggingface.co/example/x
  paper:
    arxiv_id: "1911.01601"
    url: https://arxiv.org/abs/1911.01601
    bibtex: "@misc{x,title={x}}"
dataset:
  id: SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA
  revision: 151aa4c6
  split: test
scores:
  eer_percent: 49.87
  n_trials: 71237
  n_skipped: 0
artifact:
  scores_url: https://huggingface.co/example/x/resolve/abcdef1/path/scores.txt
  scores_sha256: 71ac000c0712a4551873dba87183e746cb9730cd5ab17aaa87892009bde55587
  bench_version: speech-spoof-bench==0.1.0
reproduction:
  reproduced_by: SpeechAntiSpoofingBenchmarks
  reproduced_at: 2026-05-21
  reproduced_bench_version: speech-spoof-bench==0.1.0
  match: scoring
submitter:
  hf_username: example
  contact: e@example.com
submitted_at: 2026-05-21
""")


def test_load_schema_accepts_v4_and_v5():
    # submission schema 4->5 uses an enum (not a const) so already-merged v4
    # submission docs stay valid on re-ingest / verify-pr; a hard const:5 would
    # mass-reject them.
    schema = load_submission_schema()
    assert schema["properties"]["schema_version"]["enum"] == [4, 5]


def test_parse_submission_returns_dict_for_valid_yaml():
    sub = parse_submission(VALID_YAML)
    assert sub["system"]["slug"] == "random-baseline"
    assert sub["scores"]["eer_percent"] == 49.87


def test_parse_submission_accepts_missing_reproduction():
    # reproduction is optional at submit time; maintainer fills it at merge time
    data = yaml.safe_load(VALID_YAML)
    del data["reproduction"]
    result = parse_submission(yaml.safe_dump(data))
    assert result.get("reproduction", {}) in ({}, None)


def test_parse_submission_accepts_empty_reproduction():
    data = yaml.safe_load(VALID_YAML)
    data["reproduction"] = {}
    result = parse_submission(yaml.safe_dump(data))
    assert result["reproduction"] == {}


def test_parse_submission_rejects_unpinned_scores_url():
    bad = yaml.safe_load(VALID_YAML)
    bad["artifact"]["scores_url"] = "https://huggingface.co/example/x/resolve/main/path/scores.txt"
    with pytest.raises(SubmissionValidationError):
        parse_submission(yaml.safe_dump(bad))


# --- submission schema 4 -> 5 (additive; v4 stays valid) -------------------

def test_parse_submission_accepts_schema_version_5():
    data = yaml.safe_load(VALID_YAML)
    data["schema_version"] = 5
    out = parse_submission(yaml.safe_dump(data))
    assert out["schema_version"] == 5


def test_parse_submission_still_accepts_schema_version_4():
    # back-compat: the many already-merged v4 docs must keep validating.
    data = yaml.safe_load(VALID_YAML)
    assert data["schema_version"] == 4
    out = parse_submission(yaml.safe_dump(data))
    assert out["schema_version"] == 4


def test_parse_submission_rejects_schema_version_6():
    data = yaml.safe_load(VALID_YAML)
    data["schema_version"] = 6
    with pytest.raises(SubmissionValidationError):
        parse_submission(yaml.safe_dump(data))


def test_parse_submission_accepts_trained_on():
    data = yaml.safe_load(VALID_YAML)
    data["schema_version"] = 5
    data["system"]["trained_on"] = [
        "SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA",
        "SpeechAntiSpoofingBenchmarks/InTheWild",
    ]
    out = parse_submission(yaml.safe_dump(data))
    assert out["system"]["trained_on"][0].endswith("ASVspoof2019_LA")


def test_parse_submission_rejects_bad_trained_on_id():
    data = yaml.safe_load(VALID_YAML)
    data["system"]["trained_on"] = ["no-slash"]
    with pytest.raises(SubmissionValidationError):
        parse_submission(yaml.safe_dump(data))


def test_parse_submission_rejects_whitespace_in_trained_on_id():
    # trained_on references canonical manifest dataset ids, which forbid
    # whitespace; a value with leading/internal space can never match one.
    data = yaml.safe_load(VALID_YAML)
    data["system"]["trained_on"] = [" SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA"]
    with pytest.raises(SubmissionValidationError):
        parse_submission(yaml.safe_dump(data))


def test_parse_submission_accepts_calibration_block():
    data = yaml.safe_load(VALID_YAML)
    data["schema_version"] = 5
    data["calibration"] = {
        "source_dataset": "SpeechAntiSpoofingBenchmarks/DeepVoice",
        "threshold": -1.234,
    }
    out = parse_submission(yaml.safe_dump(data))
    assert out["calibration"]["source_dataset"].endswith("DeepVoice")
    assert out["calibration"]["threshold"] == -1.234


def test_parse_submission_rejects_calibration_missing_threshold():
    data = yaml.safe_load(VALID_YAML)
    data["calibration"] = {"source_dataset": "SpeechAntiSpoofingBenchmarks/DeepVoice"}
    with pytest.raises(SubmissionValidationError):
        parse_submission(yaml.safe_dump(data))


def test_parse_submission_rejects_whitespace_in_calibration_source():
    data = yaml.safe_load(VALID_YAML)
    data["calibration"] = {"source_dataset": "Org Name/x", "threshold": 0.5}
    with pytest.raises(SubmissionValidationError):
        parse_submission(yaml.safe_dump(data))


def test_parse_submission_rejects_calibration_extra_field():
    data = yaml.safe_load(VALID_YAML)
    data["calibration"] = {
        "source_dataset": "SpeechAntiSpoofingBenchmarks/DeepVoice",
        "threshold": 0.5,
        "bogus": 1,
    }
    with pytest.raises(SubmissionValidationError):
        parse_submission(yaml.safe_dump(data))


def test_match_accepts_organizer_and_closed_eval():
    for m in ("organizer", "closed-eval"):
        data = yaml.safe_load(VALID_YAML)
        data["reproduction"]["match"] = m
        out = parse_submission(yaml.safe_dump(data))
        assert out["reproduction"]["match"] == m


def test_match_rejects_unknown():
    data = yaml.safe_load(VALID_YAML)
    data["reproduction"]["match"] = "bogus"
    with pytest.raises(SubmissionValidationError):
        parse_submission(yaml.safe_dump(data))
