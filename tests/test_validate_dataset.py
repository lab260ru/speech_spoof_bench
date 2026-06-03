"""Tests for validate.py dataset-side checks (D1–D7)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import yaml
from datasets import Audio, ClassLabel, Features, Value

from speech_spoof_bench import validate


def test_happy_path(synth_local_dataset):
    report = validate.validate_dataset(str(synth_local_dataset), skip_submissions=True)
    assert report.ok, report.format()
    assert all(check.passed for check in report.dataset_checks)


def test_d6_missing_arena_ready_tag(synth_local_dataset):
    readme = (synth_local_dataset / "README.md").read_text()
    bad = readme.replace("- arena-ready\n", "")
    (synth_local_dataset / "README.md").write_text(bad)
    report = validate.validate_dataset(str(synth_local_dataset), skip_submissions=True)
    assert not report.ok
    assert any("arena-ready" in c.message for c in report.dataset_checks if not c.passed)


def test_d6_missing_arxiv(synth_local_dataset):
    readme = (synth_local_dataset / "README.md").read_text()
    bad = readme.replace("arxiv:\n  - 1911.01601\n", "")
    (synth_local_dataset / "README.md").write_text(bad)
    report = validate.validate_dataset(str(synth_local_dataset), skip_submissions=True)
    assert not report.ok
    assert any(c.id == "D6" and not c.passed for c in report.dataset_checks)


def test_d7_unregistered_metric(synth_local_dataset):
    ev = yaml.safe_load((synth_local_dataset / "eval.yaml").read_text())
    ev["tasks"][0]["metrics"] = ["does_not_exist"]
    (synth_local_dataset / "eval.yaml").write_text(yaml.safe_dump(ev))
    report = validate.validate_dataset(str(synth_local_dataset), skip_submissions=True)
    assert not report.ok
    assert any(c.id == "D7" and not c.passed for c in report.dataset_checks)


def test_hf_dataset_checks_use_metadata_without_decoding_audio(monkeypatch):
    source = SimpleNamespace(
        is_local=False,
        local_path=None,
        canonical_id="Org/Ds",
        metrics=["eer_percent"],
    )

    class MetadataOnlyDataset:
        features = Features({
            "path": Value("string"),
            "audio": Audio(sampling_rate=16000),
            "label": ClassLabel(names=["bonafide", "spoof"]),
            "notes": Value("string"),
        })

        def __iter__(self):
            raise AssertionError("online validation must not decode audio rows")

    scan_rows = [
        {"path": "a.flac", "notes": '{"utterance_id":"a"}'},
        {"path": "b.flac", "notes": '{"utterance_id":"b"}'},
    ]

    def fake_resolve(spec, *, streaming=True, force_remote=False, columns=None):
        assert spec == "Org/Ds"
        assert force_remote is True
        if columns is None:
            raise AssertionError("HF metadata validation must not call load_dataset")
        assert columns == ["notes", "path"]
        return source, iter(scan_rows)

    monkeypatch.setattr(
        validate,
        "_resolve_hf_metadata",
        lambda spec: (source, MetadataOnlyDataset()),
        raising=False,
    )
    monkeypatch.setattr(
        validate,
        "hf_hub_download",
        lambda **kw: (_ for _ in ()).throw(FileNotFoundError("no labels.parquet")),
    )
    monkeypatch.setattr(validate, "resolve", fake_resolve)
    monkeypatch.setattr(validate, "_load_readme", lambda **kw: """---
license: other
language: [en]
pretty_name: Test
task_categories: [audio-classification]
size_categories: [n<1K]
configs: []
tags: [arena-ready]
arxiv: ["1234.5678"]
---
body
""")

    checks, _ = validate._check_dataset_side("Org/Ds", force_remote=True)
    by_id = {c.id: c for c in checks}
    assert by_id["D1"].passed
    assert by_id["D2"].passed
    assert by_id["D3"].passed
    assert by_id["D4"].passed
    assert by_id["D5"].passed


def test_hf_metadata_resolver_does_not_call_datasets_builder(monkeypatch, tmp_path):
    eval_yaml = tmp_path / "eval.yaml"
    eval_yaml.write_text("""
name: Test
tasks:
  - id: antispoofing_eval
    split: test
    metrics: [eer_percent]
""")

    monkeypatch.setattr(
        validate,
        "hf_hub_download",
        lambda **kw: eval_yaml,
    )
    monkeypatch.setattr(
        validate,
        "load_dataset_builder",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("online validation must not glob remote parquet")
        ),
        raising=False,
    )

    source, ds = validate._resolve_hf_metadata("Org/Ds")
    assert source.canonical_id == "Org/Ds"
    assert set(ds.features.keys()) == {"path", "audio", "label", "notes"}
    assert ds.features["audio"].sampling_rate == 16000
    assert ds.features["label"].names == ["bonafide", "spoof"]


def test_hf_dataset_checks_use_labels_file_without_streaming_shards(
    monkeypatch, tmp_path
):
    import pyarrow as pa
    import pyarrow.parquet as pq

    labels_path = tmp_path / "labels.parquet"
    pq.write_table(
        pa.table({
            "utterance_id": pa.array(["a", "b"], pa.string()),
            "label": pa.array([0, 1], pa.int8()),
        }),
        labels_path,
    )

    source = SimpleNamespace(
        is_local=False,
        local_path=None,
        canonical_id="Org/Ds",
        metrics=["eer_percent"],
    )

    class MetadataOnlyDataset:
        features = Features({
            "path": Value("string"),
            "audio": Audio(sampling_rate=16000),
            "label": ClassLabel(names=["bonafide", "spoof"]),
            "notes": Value("string"),
        })

    def fake_resolve(spec, *, streaming=True, force_remote=False, columns=None):
        raise AssertionError("online validation must not stream parquet shards")

    def fake_download(*, repo_id, filename, repo_type):
        assert repo_id == "Org/Ds"
        assert repo_type == "dataset"
        assert filename == "data/labels.parquet"
        return labels_path

    monkeypatch.setattr(
        validate,
        "_resolve_hf_metadata",
        lambda spec: (source, MetadataOnlyDataset()),
        raising=False,
    )
    monkeypatch.setattr(validate, "resolve", fake_resolve)
    monkeypatch.setattr(validate, "hf_hub_download", fake_download)
    monkeypatch.setattr(validate, "_load_readme", lambda **kw: """---
license: other
language: [en]
pretty_name: Test
task_categories: [audio-classification]
size_categories: [n<1K]
configs: []
tags: [arena-ready]
arxiv: ["1234.5678"]
---
body
""")

    checks, _ = validate._check_dataset_side("Org/Ds", force_remote=True)
    by_id = {c.id: c for c in checks}
    assert by_id["D4"].passed
    assert "labels.parquet" in by_id["D4"].message
    assert by_id["D5"].passed
    assert "utterance_id" in by_id["D5"].message


import hashlib
from unittest.mock import patch


def test_submissions_happy_path(synth_local_dataset, tmp_path):
    fake_scores = tmp_path / "scores.txt"
    fake_scores.write_text("x\n")
    sha = hashlib.sha256(fake_scores.read_bytes()).hexdigest()

    sub_path = synth_local_dataset / "submissions" / "fixture.yaml"
    sub_yaml = yaml.safe_load(sub_path.read_text())
    sub_yaml["artifact"]["scores_sha256"] = sha
    sub_path.write_text(yaml.safe_dump(sub_yaml))

    with patch("speech_spoof_bench.validate.hf_fetch.download",
               return_value=(fake_scores, sha)):
        report = validate.validate_dataset(str(synth_local_dataset))
    assert report.ok, report.format()
    assert len(report.submission_reports) == 1
    assert report.submission_reports[0].ok


def test_submission_unreachable_url(synth_local_dataset):
    with patch("speech_spoof_bench.validate.hf_fetch.download",
               side_effect=OSError("HTTP 404")):
        report = validate.validate_dataset(str(synth_local_dataset))
    assert not report.ok
    sr = report.submission_reports[0]
    assert any(c.id == "S3" and not c.passed for c in sr.checks)
    assert any(c.id == "S4" and "depends" in c.message for c in sr.checks)


def test_submission_sha_mismatch(synth_local_dataset, tmp_path):
    fake_scores = tmp_path / "scores.txt"
    fake_scores.write_text("x\n")
    # The fixture claims scores_sha256 = "0" * 64; return a different hash
    # so S4 detects the mismatch.
    wrong_sha = "a" * 64
    with patch("speech_spoof_bench.validate.hf_fetch.download",
               return_value=(fake_scores, wrong_sha)):
        report = validate.validate_dataset(str(synth_local_dataset))
    assert not report.ok
    sr = report.submission_reports[0]
    assert any(c.id == "S4" and not c.passed for c in sr.checks)


def test_skip_submissions_flag(synth_local_dataset):
    report = validate.validate_dataset(
        str(synth_local_dataset), skip_submissions=True
    )
    assert report.submission_reports == []


def test_list_submission_paths_local(tmp_path):
    sub = tmp_path / "submissions"
    sub.mkdir()
    (sub / "valid_one.yaml").write_text("placeholder")
    (sub / "valid_two.yaml").write_text("placeholder")
    (sub / "README.md").write_text("readme")
    (sub / "results_template.yaml").write_text("template")
    (sub / "not_a_yaml.txt").write_text("nope")

    result = validate._list_submission_paths(spec_path=tmp_path, repo_id=None)

    assert len(result) == 2
    display_paths = sorted(d for d, _ in result)
    assert display_paths == [
        "submissions/valid_one.yaml",
        "submissions/valid_two.yaml",
    ]
    for _, local in result:
        assert Path(local).is_file()


def test_list_submission_paths_no_submissions_dir(tmp_path):
    result = validate._list_submission_paths(spec_path=tmp_path, repo_id=None)
    assert result == []
