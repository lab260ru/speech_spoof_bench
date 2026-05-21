"""Tests for Benchmark.run orchestration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

# Register eer_percent.
import speech_spoof_bench.metrics.eer  # noqa: F401
from speech_spoof_bench.benchmark import Benchmark, BenchmarkResult
from speech_spoof_bench.model import SimpleAntiSpoofingModel


class _SeededRandom(SimpleAntiSpoofingModel):
    name = "seeded-random"

    def __init__(self):
        self.load_count = 0
        self.unload_count = 0
        self.rng = None

    def load(self):
        self.load_count += 1
        self.rng = np.random.default_rng(42)

    def unload(self):
        self.unload_count += 1
        self.rng = None

    def score(self, audio, sr):
        return float(self.rng.standard_normal())


def test_run_against_local_dataset(synth_local_dataset: Path, tmp_path: Path):
    out = tmp_path / "results"
    m = _SeededRandom()
    results = Benchmark.run(
        m,
        datasets=[str(synth_local_dataset)],
        output_dir=out,
        streaming=True,
        cleanup=False,
        skip_existing=False,
    )

    assert isinstance(results, dict)
    key = "SynthDataset_TEST"
    assert key in results
    br = results[key]
    assert isinstance(br, BenchmarkResult)
    assert "eer_percent" in br.metrics

    # On 4 items the metric may be anywhere; just sanity-check the YAML.
    result_yaml = out / key / "result.yaml"
    assert result_yaml.exists()
    parsed = yaml.safe_load(result_yaml.read_text())
    assert parsed["schema_version"] == 4
    assert parsed["dataset"]["id"] == "SynthDataset_TEST"
    assert parsed["dataset"]["split"] == "test"
    assert parsed["dataset"]["revision"] is None
    assert "eer_percent" in parsed["scores"]
    assert parsed["scores"]["n_trials"] == 4
    assert parsed["scores"]["n_skipped"] == 0
    # Empty blocks reserved for later phases.
    assert parsed["reproduction"] == {}
    assert parsed["submitter"] == {}
    assert parsed["artifact"]["scores_url"] is None
    assert isinstance(parsed["artifact"]["scores_sha256"], str)
    assert len(parsed["artifact"]["scores_sha256"]) == 64


def test_load_and_unload_called_exactly_once_across_datasets(
    synth_local_dataset: Path, tmp_path: Path
):
    # Two specs pointing at the same dataset is enough — orchestrator should
    # still load once at start and unload once at end.
    out = tmp_path / "results"
    m = _SeededRandom()

    # Use skip_existing=False and a second output dir to force two runs.
    Benchmark.run(
        m,
        datasets=[str(synth_local_dataset), str(synth_local_dataset)],
        output_dir=out,
        streaming=True,
        cleanup=False,
        skip_existing=False,
    )

    assert m.load_count == 1
    assert m.unload_count == 1


def test_skip_existing_short_circuits(synth_local_dataset: Path, tmp_path: Path):
    out = tmp_path / "results"
    m = _SeededRandom()
    Benchmark.run(
        m,
        datasets=[str(synth_local_dataset)],
        output_dir=out,
        streaming=True,
        cleanup=False,
        skip_existing=False,
    )
    # Second run with skip_existing=True should not re-score.
    m2 = _SeededRandom()
    Benchmark.run(
        m2,
        datasets=[str(synth_local_dataset)],
        output_dir=out,
        streaming=True,
        cleanup=False,
        skip_existing=True,
    )
    # When everything is skipped, load/unload still run once.
    assert m2.load_count == 1
    assert m2.unload_count == 1


def test_all_keyword_not_implemented_at_phase_2(synth_local_dataset: Path, tmp_path: Path):
    m = _SeededRandom()
    with pytest.raises(NotImplementedError, match="phase 4"):
        Benchmark.run(m, datasets="all", output_dir=tmp_path)


def test_unload_runs_on_exception(synth_local_dataset: Path, tmp_path: Path):
    class M(_SeededRandom):
        def score(self, audio, sr):
            raise RuntimeError("forced failure")

    m = M()
    out = tmp_path / "results"
    with pytest.raises(Exception):
        Benchmark.run(
            m,
            datasets=[str(synth_local_dataset)],
            output_dir=out,
            streaming=True,
            cleanup=False,
            skip_existing=False,
        )
    assert m.load_count == 1
    assert m.unload_count == 1
