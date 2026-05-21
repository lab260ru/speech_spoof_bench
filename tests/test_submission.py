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


def test_load_schema_has_v4_const():
    schema = load_submission_schema()
    assert schema["properties"]["schema_version"]["const"] == 4


def test_parse_submission_returns_dict_for_valid_yaml():
    sub = parse_submission(VALID_YAML)
    assert sub["system"]["slug"] == "random-baseline"
    assert sub["scores"]["eer_percent"] == 49.87


def test_parse_submission_rejects_missing_reproduction():
    bad = yaml.safe_load(VALID_YAML)
    del bad["reproduction"]
    with pytest.raises(SubmissionValidationError):
        parse_submission(yaml.safe_dump(bad))


def test_parse_submission_rejects_unpinned_scores_url():
    bad = yaml.safe_load(VALID_YAML)
    bad["artifact"]["scores_url"] = "https://huggingface.co/example/x/resolve/main/path/scores.txt"
    with pytest.raises(SubmissionValidationError):
        parse_submission(yaml.safe_dump(bad))
