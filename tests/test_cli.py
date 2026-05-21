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


def test_cli_list_raises_at_phase_2(capsys):
    rc = main(["list"])
    assert rc != 0
    captured = capsys.readouterr()
    assert "phase 4" in (captured.out + captured.err).lower()
