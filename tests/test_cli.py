"""Tests for the CLI entry point."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

import speech_spoof_bench.metrics.eer  # noqa: F401
from speech_spoof_bench.cli import main


def test_cli_help_runs(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0


def test_cli_run_against_local(synth_local_dataset: Path, tmp_path: Path):
    out = tmp_path / "results"
    rc = main(
        [
            "run",
            "--model-module",
            "speech_spoof_bench.examples.random_baseline:RandomBaseline",
            "--datasets",
            str(synth_local_dataset),
            "--output-dir",
            str(out),
            "--no-cleanup",
            "--no-skip-existing",
        ]
    )
    assert rc == 0
    result_yaml = out / "SynthDataset_TEST" / "result.yaml"
    assert result_yaml.exists()


def test_cli_validate_dataset_local(synth_local_dataset: Path):
    rc = main(["validate-dataset", str(synth_local_dataset)])
    assert rc == 0


from speech_spoof_bench import manifest as _mf


_FAKE_MANIFEST = {
    "ranking_version": "v1",
    "schema_version": 1,
    "metrics_in_use": ["eer_percent"],
    "tiers": [
        {"name": "gold", "min_coverage": 1.0},
        {"name": "silver", "min_coverage": 0.5},
        {"name": "bronze", "min_coverage": 0.0},
    ],
    "core_set": [
        {"id": "Org/A", "revision": "9b2040e8c57749dcd9a4f16ad61b4f47626b89ec"}
    ],
    "extended": [
        {"id": "Org/B", "revision": "deadbeef"}
    ],
}


def test_cli_manifest_prints_yaml(monkeypatch, capsys, tmp_path):
    """`manifest` prints the raw file contents verbatim."""
    raw = (
        "ranking_version: v1\n"
        "schema_version: 1\n"
        "metrics_in_use:\n  - eer_percent\n"
        "tiers:\n  - {name: gold, min_coverage: 1.0}\n"
        "  - {name: silver, min_coverage: 0.5}\n"
        "  - {name: bronze, min_coverage: 0.0}\n"
        "core_set:\n  - id: Org/A\n    revision: 9b2040e8c57749dcd9a4f16ad61b4f47626b89ec\n"
        "extended: []\n"
    )
    fake = tmp_path / "manifest.yaml"
    fake.write_text(raw)

    def fake_download(**kwargs):
        return str(fake)

    monkeypatch.setattr(_mf, "hf_hub_download", fake_download)
    rc = main(["manifest"])
    assert rc == 0
    out = capsys.readouterr().out
    assert out.rstrip("\n") == raw.rstrip("\n")


def test_cli_list_prints_core_then_extended(monkeypatch, capsys):
    def fake_fetch():
        return _FAKE_MANIFEST

    monkeypatch.setattr(_mf, "fetch_manifest", fake_fetch)
    rc = main(["list"])
    assert rc == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines == ["[core] Org/A", "[ext]  Org/B"]
