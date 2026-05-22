"""Shared pytest fixtures."""

from __future__ import annotations

import io
import json
import wave
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml


def _wav_bytes(audio: np.ndarray, sr: int = 16000) -> bytes:
    """Encode a float32 mono array as 16-bit PCM WAV bytes."""
    pcm = np.clip(audio * 32768.0, -32768, 32767).astype(np.int16).tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm)
    return buf.getvalue()


@pytest.fixture
def synth_local_dataset(tmp_path) -> Path:
    """Build a 4-row local dataset directory matching the v4 schema.

    Layout:
        <tmp>/
            eval.yaml
            data/test-00000-of-00001.parquet

    Returns the directory path.
    """
    root = tmp_path / "SynthDataset_TEST"
    (root / "data").mkdir(parents=True)

    rng = np.random.default_rng(0)
    rows = []
    for i in range(4):
        utt_id = f"UTT_{i:04d}"
        label = i % 2  # alternating bonafide/spoof
        audio = rng.standard_normal(16000).astype(np.float32) * 0.1
        rows.append(
            {
                "path": f"audio/{utt_id}.wav",
                "audio": {"bytes": _wav_bytes(audio), "path": None},
                "label": label,
                "notes": json.dumps({"utterance_id": utt_id, "speaker_id": "S0"}),
            }
        )

    # The HF datasets Audio feature expects {"bytes", "path"}. Storing as such
    # in parquet means load_dataset("parquet", ...) yields rows with raw dicts;
    # we cast to Audio() at load time in loader.py. For this fixture, we keep
    # the column as struct so the test exercises the same code path.
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, root / "data" / "test-00000-of-00001.parquet")

    eval_yaml = {
        "name": "Synth Dataset TEST",
        "description": "Synthetic dataset for unit tests.",
        "evaluation_framework": "inspect-ai",
        "tasks": [
            {
                "id": "antispoofing_eval",
                "config": "default",
                "split": "test",
                "field_spec": {"input": "audio", "target": "label"},
                "solvers": [{"name": "speech_spoof_bench_solver"}],
                "scorers": [{"name": "speech_spoof_scorer"}],
                "metrics": ["eer_percent"],
            }
        ],
    }
    (root / "eval.yaml").write_text(yaml.safe_dump(eval_yaml))

    readme = """---
license: cc-by-4.0
language: [en]
pretty_name: Synth Dataset TEST
task_categories: [audio-classification]
size_categories: [n<1K]
configs:
  - config_name: default
    data_files:
      - {split: test, path: "data/test-*.parquet"}
arxiv:
  - 1911.01601
tags:
  - anti-spoofing
  - arena-ready
---

# Synth Dataset TEST
Synthetic fixture used by 7a tests.
"""
    (root / "README.md").write_text(readme)

    (root / "submissions").mkdir()
    fixture_src = Path(__file__).parent / "fixtures" / "submissions" / "valid.yaml"
    sub = yaml.safe_load(fixture_src.read_text())
    (root / "submissions" / "fixture.yaml").write_text(yaml.safe_dump(sub))

    return root
