"""Tests for validate.py dataset-side checks (D1–D7)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

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
