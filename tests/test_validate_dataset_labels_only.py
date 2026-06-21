"""Tests for `validate-dataset --labels-only` (a labels.parquet repo, no audio)."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from speech_spoof_bench import validate


def _ids(report) -> set[str]:
    return {c.id for c in report.dataset_checks}


def _check(report, cid):
    for c in report.dataset_checks:
        if c.id == cid:
            return c
    return None


def _make_labels_only(
    tmp_path,
    *,
    labels=None,
    metrics=("srr_complement",),
    arena_ready=True,
    with_labels_file=True,
):
    root = tmp_path / "LabelsOnly_TEST"
    (root / "data").mkdir(parents=True)

    if labels is None:
        labels = [("u0", 1), ("u1", 1), ("u2", 1), ("u3", 1)]
    if with_labels_file:
        uids = [u for u, _ in labels]
        vals = [v for _, v in labels]
        pq.write_table(
            pa.table({
                "utterance_id": pa.array(uids, pa.string()),
                "label": pa.array(vals, pa.int8()),
            }),
            root / "data" / "labels.parquet",
        )

    eval_yaml = {
        "name": "Labels Only TEST",
        "description": "Labels-only fixture.",
        "evaluation_framework": "inspect-ai",
        "tasks": [{
            "id": "antispoofing_eval",
            "config": "default",
            "split": "test",
            "field_spec": {"input": "audio", "target": "label"},
            "solvers": [{"name": "speech_spoof_bench_solver"}],
            "scorers": [{"name": "speech_spoof_scorer"}],
            "metrics": list(metrics),
        }],
    }
    (root / "eval.yaml").write_text(yaml.safe_dump(eval_yaml))

    tags = ["anti-spoofing", "speech"]
    if arena_ready:
        tags.append("arena-ready")
    fm = {
        "license": "mit",
        "language": ["en"],
        "pretty_name": "Labels Only TEST",
        "task_categories": ["audio-classification"],
        "size_categories": ["1K<n<10K"],
        "configs": [{"config_name": "default",
                     "data_files": [{"split": "test", "path": "data/labels.parquet"}]}],
        "arxiv": ["2603.02364"],
        "tags": tags,
    }
    readme = "---\n" + yaml.safe_dump(fm) + "---\n\n# Labels Only TEST\n"
    (root / "README.md").write_text(readme)
    return root


def test_labels_only_happy_path(tmp_path):
    root = _make_labels_only(tmp_path)
    report = validate.validate_dataset(str(root), labels_only=True, skip_submissions=True)
    assert report.ok, report.format()
    # Audio checks must NOT run in labels-only mode.
    assert _ids(report).isdisjoint({"D1", "D2", "D3", "D4", "D5"})
    # Labels + README + metric checks must pass.
    for cid in ("L1", "L2", "L3", "D6", "D7"):
        c = _check(report, cid)
        assert c is not None and c.passed, f"{cid}: {c}"


def test_labels_only_missing_labels_file(tmp_path):
    root = _make_labels_only(tmp_path, with_labels_file=False)
    report = validate.validate_dataset(str(root), labels_only=True, skip_submissions=True)
    assert not report.ok
    assert _check(report, "L1") is not None and not _check(report, "L1").passed


def test_labels_only_bad_label_value(tmp_path):
    root = _make_labels_only(tmp_path, labels=[("u0", 1), ("u1", 2)])
    report = validate.validate_dataset(str(root), labels_only=True, skip_submissions=True)
    assert not report.ok
    assert not _check(report, "L2").passed


def test_labels_only_duplicate_utterance_id(tmp_path):
    root = _make_labels_only(tmp_path, labels=[("dup", 1), ("dup", 1), ("u2", 1)])
    report = validate.validate_dataset(str(root), labels_only=True, skip_submissions=True)
    assert not report.ok
    assert not _check(report, "L3").passed


def test_labels_only_n_trials_mismatch(tmp_path):
    root = _make_labels_only(tmp_path)  # 4 rows
    bad = validate.validate_dataset(str(root), labels_only=True, skip_submissions=True, n_trials=999)
    assert not _check(bad, "L4").passed
    good = validate.validate_dataset(str(root), labels_only=True, skip_submissions=True, n_trials=4)
    assert _check(good, "L4").passed


def test_labels_only_n_trials_soft_pass_when_omitted(tmp_path):
    root = _make_labels_only(tmp_path)
    report = validate.validate_dataset(str(root), labels_only=True, skip_submissions=True)
    l4 = _check(report, "L4")
    assert l4 is not None and l4.passed  # soft pass, reports the count


def test_labels_only_unregistered_metric(tmp_path):
    root = _make_labels_only(tmp_path, metrics=("does_not_exist",))
    report = validate.validate_dataset(str(root), labels_only=True, skip_submissions=True)
    assert not report.ok
    assert not _check(report, "D7").passed


def test_labels_only_empty_file_fails(tmp_path):
    # A 0-row labels.parquet must NOT earn trust (default CLI omits --n-trials).
    root = _make_labels_only(tmp_path, labels=[])
    report = validate.validate_dataset(str(root), labels_only=True, skip_submissions=True)
    assert not report.ok


def test_labels_only_null_utterance_id_fails(tmp_path):
    # A null utterance_id is unmatchable against a submitter's scores.txt.
    root = _make_labels_only(tmp_path, with_labels_file=False)
    pq.write_table(
        pa.table({
            "utterance_id": pa.array(["a", None, "b"], pa.string()),
            "label": pa.array([1, 1, 1], pa.int8()),
        }),
        root / "data" / "labels.parquet",
    )
    report = validate.validate_dataset(str(root), labels_only=True, skip_submissions=True)
    assert not report.ok


def test_labels_only_missing_arena_ready_tag(tmp_path):
    root = _make_labels_only(tmp_path, arena_ready=False)
    report = validate.validate_dataset(str(root), labels_only=True, skip_submissions=True)
    assert not report.ok
    assert not _check(report, "D6").passed
