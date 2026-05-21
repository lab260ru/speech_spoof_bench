# Phase 2 — Pip skeleton with local dataset loading — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the minimum pip package that lets the random baseline run end-to-end against `ASVspoof2019_LA` (loaded from the local mount, no HF download required) and produce `scores.txt` + a partially-filled `result.yaml`.

**Architecture:** A single Python package `speech_spoof_bench` with thin, focused modules. `loader.py` dispatches between a local directory and an HF repo id. `runner.py` iterates the dataset, calls `model.score_batch`, falls back to per-item scoring when a batch raises. `benchmark.py` orchestrates load → run → metrics → write result. The model API uses `score_batch` as the primitive plus a `SimpleAntiSpoofingModel` convenience subclass.

**Tech Stack:** Python 3.10+, `datasets`, `huggingface_hub`, `numpy`, `pyyaml`, `jsonschema`, `scipy`, `pytest`. No PyTorch in core deps.

**Spec:** `docs/specs/2026-05-21-phase-2-pip-skeleton-design.md` (commit `10b41fd`, revised in HEAD).

---

## Preconditions / Assumptions

- The pip-package working copy is at `/home/kirill/speech-spoof-bench/speech-spoof-bench/`. All paths in this plan are relative to that directory unless explicitly absolute.
- The local LA dataset working copy is at `/home/kirill/mnt/drive3_8tb/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA/` with `data/test-*.parquet` shards and a valid `eval.yaml` (Phase 1 complete).
- A venv-or-equivalent Python 3.10+ environment exists. Engineer activates it before working.
- This package's git repo is **separate** from the dataset repo. All commits in this plan land in the pip-package repo only.

---

## File Structure

```
speech-spoof-bench/
├── pyproject.toml                             # Task 1
├── README.md                                  # Task 1 (stub; expanded later phases)
├── .gitignore                                 # Task 1
├── src/speech_spoof_bench/
│   ├── __init__.py                            # Task 1 (exports public API)
│   ├── model.py                               # Task 3
│   ├── metrics/
│   │   ├── __init__.py                        # Task 2 (registry + register_metric)
│   │   └── eer.py                             # Task 2 (eer_percent)
│   ├── loader.py                              # Task 4 (DatasetSource + resolve)
│   ├── runner.py                              # Task 5 (run_dataset)
│   ├── benchmark.py                           # Task 6 (Benchmark.run)
│   ├── cache.py                               # Task 7
│   ├── manifest.py                            # Task 8 (stub)
│   ├── cli.py                                 # Task 9
│   └── examples/
│       ├── __init__.py                        # Task 10
│       └── random_baseline.py                 # Task 10
└── tests/
    ├── __init__.py                            # Task 1
    ├── conftest.py                            # Task 4 (synth parquet fixture)
    ├── metrics/
    │   ├── __init__.py                        # Task 2
    │   └── test_eer.py                        # Task 2
    ├── test_loader.py                         # Task 4
    ├── test_runner.py                         # Task 5
    └── test_benchmark.py                      # Task 6
```

Each module has one job and is small enough to read in one screenful. The smoke-test against the real local dataset (Task 11) is a manual verification, not a CI test.

---

## Task 1: Package skeleton + pyproject.toml

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `README.md`
- Create: `src/speech_spoof_bench/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1.1: Write `pyproject.toml`**

Create `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "speech-spoof-bench"
version = "0.1.0"
description = "Benchmark harness for speech anti-spoofing models."
requires-python = ">=3.10"
readme = "README.md"
license = {text = "Apache-2.0"}
authors = [{name = "SpeechAntiSpoofingBenchmarks maintainers"}]
dependencies = [
    "datasets>=2.18",
    "huggingface_hub>=0.20",
    "numpy>=1.24",
    "pyyaml>=6.0",
    "jsonschema>=4.0",
    "scipy>=1.10",
]

[project.optional-dependencies]
dev = ["pytest>=7.0", "pytest-cov>=4.0"]

[project.scripts]
speech-spoof-bench = "speech_spoof_bench.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra"
```

- [ ] **Step 1.2: Write `.gitignore`**

Create `.gitignore`:

```
__pycache__/
*.py[cod]
*.egg-info/
build/
dist/
.pytest_cache/
.coverage
htmlcov/
.venv/
venv/
results/
.DS_Store
```

- [ ] **Step 1.3: Write a minimal `README.md`**

Create `README.md`:

```markdown
# speech-spoof-bench

Benchmark harness for the [SpeechAntiSpoofingBenchmarks](https://huggingface.co/SpeechAntiSpoofingBenchmarks) org. Run anti-spoofing models against published datasets (HF) or a local copy on disk.

## Install

    pip install -e .

## Quick start

    speech-spoof-bench run \
        --model-module speech_spoof_bench.examples.random_baseline:RandomBaseline \
        --datasets /path/to/local/ASVspoof2019_LA \
        --output-dir ./results

For full design and roadmap see `docs/roadmap/PLAN.md` and `docs/roadmap/ROADMAP.md`.
```

- [ ] **Step 1.4: Create the package `__init__.py`**

Create `src/speech_spoof_bench/__init__.py`:

```python
"""speech-spoof-bench — anti-spoofing benchmark harness."""

__version__ = "0.1.0"
```

- [ ] **Step 1.5: Create the tests `__init__.py`**

Create `tests/__init__.py` (empty file).

- [ ] **Step 1.6: Install in editable mode**

Run: `pip install -e ".[dev]"`
Expected: installs cleanly, including `pytest`. `speech-spoof-bench --help` will fail until Task 9, that's fine.

- [ ] **Step 1.7: Smoke-test the import**

Run: `python -c "import speech_spoof_bench; print(speech_spoof_bench.__version__)"`
Expected: prints `0.1.0`.

- [ ] **Step 1.8: Commit**

```bash
git add pyproject.toml .gitignore README.md src/speech_spoof_bench/__init__.py tests/__init__.py
git commit -m "feat: package skeleton — pyproject, gitignore, empty package"
```

---

## Task 2: Metrics registry + EER

**Files:**
- Create: `src/speech_spoof_bench/metrics/__init__.py`
- Create: `src/speech_spoof_bench/metrics/eer.py`
- Create: `tests/metrics/__init__.py`
- Create: `tests/metrics/test_eer.py`

- [ ] **Step 2.1: Write the EER test first**

Create `tests/metrics/__init__.py` (empty).

Create `tests/metrics/test_eer.py`:

```python
"""Tests for the eer_percent metric."""

import numpy as np
import pytest

from speech_spoof_bench.metrics import get_metric, list_metrics
# Importing the metric module triggers its @register_metric decorator.
import speech_spoof_bench.metrics.eer  # noqa: F401


def _make_inputs(bonafide_scores, spoof_scores):
    scores = {}
    labels = {}
    for i, s in enumerate(bonafide_scores):
        utt_id = f"bona_{i}"
        scores[utt_id] = float(s)
        labels[utt_id] = 0
    for i, s in enumerate(spoof_scores):
        utt_id = f"spoof_{i}"
        scores[utt_id] = float(s)
        labels[utt_id] = 1
    return scores, labels


def test_eer_is_registered():
    spec = get_metric("eer_percent")
    assert spec.id == "eer_percent"
    assert spec.lower_is_better is True
    assert spec.requires_audio is False
    assert "eer_percent" in {m.id for m in list_metrics()}


def test_eer_perfectly_separable_is_zero():
    # Bonafide scores all higher than every spoof score → EER == 0.
    scores, labels = _make_inputs(
        bonafide_scores=np.linspace(10.0, 20.0, 500),
        spoof_scores=np.linspace(-20.0, -10.0, 500),
    )
    result = get_metric("eer_percent").fn(scores, labels)
    assert result.value == pytest.approx(0.0, abs=1e-9)
    assert result.extras["n_trials"] == 1000


def test_eer_fully_overlapping_is_near_fifty():
    # Same distribution → EER ≈ 50%.
    rng = np.random.default_rng(0)
    scores, labels = _make_inputs(
        bonafide_scores=rng.standard_normal(5000),
        spoof_scores=rng.standard_normal(5000),
    )
    result = get_metric("eer_percent").fn(scores, labels)
    assert result.value == pytest.approx(50.0, abs=2.0)


def test_eer_known_intermediate():
    # Shifted normals: theoretical EER for shift d=2 is ~15.87%
    # (Φ(-d/2) where Φ is the standard normal CDF). Allow some Monte Carlo slack.
    rng = np.random.default_rng(1)
    scores, labels = _make_inputs(
        bonafide_scores=rng.standard_normal(10000) + 1.0,
        spoof_scores=rng.standard_normal(10000) - 1.0,
    )
    result = get_metric("eer_percent").fn(scores, labels)
    assert result.value == pytest.approx(15.87, abs=1.5)
    assert "threshold" in result.extras
```

- [ ] **Step 2.2: Run the tests to confirm they fail with ImportError**

Run: `pytest tests/metrics/test_eer.py -v`
Expected: `ImportError` or `ModuleNotFoundError` for `speech_spoof_bench.metrics`.

- [ ] **Step 2.3: Implement the registry**

Create `src/speech_spoof_bench/metrics/__init__.py`:

```python
"""Pluggable metric registry.

Adding a new metric: drop a file under this package that calls
``@register_metric(...)`` at import time. The dataset's ``eval.yaml``
references metrics by their ``id``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

ScoresMap = dict[str, float]
LabelsMap = dict[str, int]


@dataclass(frozen=True)
class MetricResult:
    value: float
    extras: dict[str, Any] = field(default_factory=dict)


MetricFn = Callable[[ScoresMap, LabelsMap], MetricResult]


@dataclass(frozen=True)
class MetricSpec:
    id: str
    display_name: str
    lower_is_better: bool
    requires_audio: bool
    fn: MetricFn


_REGISTRY: dict[str, MetricSpec] = {}


def register_metric(
    *,
    id: str,
    display_name: str,
    lower_is_better: bool,
    requires_audio: bool = False,
) -> Callable[[MetricFn], MetricFn]:
    def decorator(fn: MetricFn) -> MetricFn:
        if id in _REGISTRY:
            raise ValueError(f"metric id {id!r} already registered")
        _REGISTRY[id] = MetricSpec(
            id=id,
            display_name=display_name,
            lower_is_better=lower_is_better,
            requires_audio=requires_audio,
            fn=fn,
        )
        return fn

    return decorator


def get_metric(id: str) -> MetricSpec:
    try:
        return _REGISTRY[id]
    except KeyError:
        raise KeyError(f"metric id {id!r} is not registered") from None


def list_metrics() -> list[MetricSpec]:
    return list(_REGISTRY.values())


def is_registered(id: str) -> bool:
    return id in _REGISTRY
```

- [ ] **Step 2.4: Implement EER**

Create `src/speech_spoof_bench/metrics/eer.py`:

```python
"""Equal Error Rate metric."""

from __future__ import annotations

import numpy as np

from . import MetricResult, register_metric


@register_metric(
    id="eer_percent",
    display_name="EER (%)",
    lower_is_better=True,
    requires_audio=False,
)
def compute_eer(scores: dict[str, float], labels: dict[str, int]) -> MetricResult:
    """Compute EER as a percentage.

    Higher score = more bonafide. label 0 = bonafide, 1 = spoof.
    """
    bona = np.array(
        [scores[u] for u, y in labels.items() if y == 0 and u in scores],
        dtype=np.float64,
    )
    spoof = np.array(
        [scores[u] for u, y in labels.items() if y == 1 and u in scores],
        dtype=np.float64,
    )
    if bona.size == 0 or spoof.size == 0:
        raise ValueError("EER needs at least one bonafide and one spoof score")

    # Build the threshold sweep over all candidate values.
    thresholds = np.unique(np.concatenate([bona, spoof]))
    # FAR(t) = P(spoof score >= t) — spoof accepted as bonafide.
    # FRR(t) = P(bonafide score < t) — bonafide rejected.
    far = np.array([(spoof >= t).mean() for t in thresholds])
    frr = np.array([(bona < t).mean() for t in thresholds])

    # Find where FAR and FRR cross.
    diff = far - frr
    # Walk left-to-right; first sign change is the crossing region.
    idx = np.where(np.diff(np.sign(diff)) != 0)[0]
    if idx.size == 0:
        # No crossing — pick the threshold minimizing |FAR - FRR|.
        i = int(np.argmin(np.abs(diff)))
        eer = float((far[i] + frr[i]) / 2.0)
        threshold = float(thresholds[i])
    else:
        i = int(idx[0])
        # Linear interpolation between thresholds[i] and thresholds[i+1].
        d0, d1 = diff[i], diff[i + 1]
        if d1 == d0:
            alpha = 0.0
        else:
            alpha = d0 / (d0 - d1)
        threshold = float(thresholds[i] + alpha * (thresholds[i + 1] - thresholds[i]))
        eer_far = float(far[i] + alpha * (far[i + 1] - far[i]))
        eer_frr = float(frr[i] + alpha * (frr[i + 1] - frr[i]))
        eer = (eer_far + eer_frr) / 2.0

    return MetricResult(
        value=eer * 100.0,
        extras={
            "threshold": threshold,
            "n_trials": int(bona.size + spoof.size),
            "n_bonafide": int(bona.size),
            "n_spoof": int(spoof.size),
        },
    )
```

- [ ] **Step 2.5: Run the metric tests**

Run: `pytest tests/metrics/test_eer.py -v`
Expected: all 4 tests PASS.

- [ ] **Step 2.6: Commit**

```bash
git add src/speech_spoof_bench/metrics/ tests/metrics/
git commit -m "feat(metrics): registry + eer_percent"
```

---

## Task 3: `AntiSpoofingModel` base class

**Files:**
- Create: `src/speech_spoof_bench/model.py`
- Create: `tests/test_model.py`

- [ ] **Step 3.1: Write the model tests first**

Create `tests/test_model.py`:

```python
"""Tests for the AntiSpoofingModel ABC and the Simple helper."""

import numpy as np
import pytest

from speech_spoof_bench.model import AntiSpoofingModel, SimpleAntiSpoofingModel


def test_cannot_instantiate_abstract_base():
    with pytest.raises(TypeError):
        AntiSpoofingModel()  # type: ignore[abstract]


def test_cannot_instantiate_simple_without_score():
    with pytest.raises(TypeError):
        SimpleAntiSpoofingModel()  # type: ignore[abstract]


def test_simple_model_score_batch_falls_back_to_score():
    class M(SimpleAntiSpoofingModel):
        name = "test"
        def load(self):
            pass
        def score(self, audio, sr):
            return float(audio.sum())

    m = M()
    m.load()
    out = m.score_batch(
        [np.array([1.0, 2.0]), np.array([3.0])],
        [16000, 16000],
    )
    assert out == [3.0, 3.0]


def test_model_defaults():
    class M(SimpleAntiSpoofingModel):
        name = "test"
        def load(self):
            pass
        def score(self, audio, sr):
            return 0.0

    m = M()
    assert m.expected_sample_rate == 16000
    assert m.batch_size == 1
```

- [ ] **Step 3.2: Run the tests to confirm they fail**

Run: `pytest tests/test_model.py -v`
Expected: `ImportError` or `ModuleNotFoundError`.

- [ ] **Step 3.3: Implement the model module**

Create `src/speech_spoof_bench/model.py`:

```python
"""Base class for anti-spoofing models.

Subclass ``AntiSpoofingModel`` (full control) or ``SimpleAntiSpoofingModel``
(easy path: just implement ``score``). The runner calls ``score_batch`` and
handles per-item retry on exceptions.
"""

from __future__ import annotations

import abc
from typing import ClassVar

import numpy as np


class AntiSpoofingModel(abc.ABC):
    """Anti-spoofing scoring model.

    Subclasses set ``name`` and may override ``expected_sample_rate`` and
    ``batch_size``. They must implement ``load`` and ``score_batch``.

    Lifecycle: ``load`` is called once per ``Benchmark.run`` invocation,
    before any dataset is processed. ``unload`` is called once at the end.

    Audio passed to ``score_batch`` is always float32 mono at 16 kHz; the
    runner is responsible for resampling.
    """

    name: ClassVar[str] = "unnamed"
    expected_sample_rate: ClassVar[int] = 16000
    batch_size: ClassVar[int] = 1

    @abc.abstractmethod
    def load(self) -> None:
        """Load weights, allocate resources. Called once per evaluation."""

    @abc.abstractmethod
    def score_batch(
        self, audios: list[np.ndarray], srs: list[int]
    ) -> list[float]:
        """Score one batch. Higher = more bonafide. len(out) == len(audios).

        Must handle any batch size 1 <= k <= self.batch_size: the runner
        falls back to single-item calls when a multi-item batch raises.
        """

    def unload(self) -> None:
        """Free resources. Default: no-op. Called once at end of evaluation."""


class SimpleAntiSpoofingModel(AntiSpoofingModel):
    """Convenience subclass: implement ``score`` instead of ``score_batch``."""

    @abc.abstractmethod
    def score(self, audio: np.ndarray, sr: int) -> float:
        """Score a single utterance. Higher = more bonafide."""

    def score_batch(
        self, audios: list[np.ndarray], srs: list[int]
    ) -> list[float]:
        return [self.score(a, s) for a, s in zip(audios, srs)]
```

- [ ] **Step 3.4: Run the model tests**

Run: `pytest tests/test_model.py -v`
Expected: all 4 tests PASS.

- [ ] **Step 3.5: Commit**

```bash
git add src/speech_spoof_bench/model.py tests/test_model.py
git commit -m "feat(model): AntiSpoofingModel ABC + SimpleAntiSpoofingModel helper"
```

---

## Task 4: Loader (local + HF dispatch)

**Files:**
- Create: `src/speech_spoof_bench/loader.py`
- Create: `tests/conftest.py`
- Create: `tests/test_loader.py`

- [ ] **Step 4.1: Write a shared fixture that builds a tiny synthetic dataset on disk**

Create `tests/conftest.py`:

```python
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
    return root
```

Note: `pyarrow` is a transitive dep of `datasets`, so it's already installed.

- [ ] **Step 4.2: Write the loader tests**

Create `tests/test_loader.py`:

```python
"""Tests for the dataset loader (local + HF dispatch)."""

from __future__ import annotations

from pathlib import Path

import pytest

# Import metric so eer_percent is registered (loader validates against registry).
import speech_spoof_bench.metrics.eer  # noqa: F401
from speech_spoof_bench.loader import DatasetSource, resolve


def test_local_dispatch(synth_local_dataset: Path):
    source, ds = resolve(str(synth_local_dataset), streaming=True)

    assert source.is_local is True
    assert source.local_path == synth_local_dataset
    assert source.slug == "SynthDataset_TEST"
    assert source.canonical_id == "SynthDataset_TEST"
    assert source.display_name == "Synth Dataset TEST"
    assert source.metrics == ["eer_percent"]
    assert source.split == "test"
    assert source.revision is None

    rows = list(ds)
    assert len(rows) == 4
    assert {r["label"] for r in rows} == {0, 1}


def test_local_dispatch_non_streaming(synth_local_dataset: Path):
    source, ds = resolve(str(synth_local_dataset), streaming=False)
    assert source.is_local is True
    assert len(ds) == 4


def test_local_missing_eval_yaml(tmp_path):
    bad = tmp_path / "no_eval"
    (bad / "data").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="eval.yaml"):
        resolve(str(bad))


def test_local_unknown_metric_in_eval_yaml(tmp_path):
    bad = tmp_path / "bad_metric"
    (bad / "data").mkdir(parents=True)
    # Drop a dummy parquet file so the parquet-glob check passes.
    (bad / "data" / "test-00000-of-00001.parquet").write_bytes(b"")
    (bad / "eval.yaml").write_text(
        "name: X\n"
        "tasks:\n"
        "  - id: t\n"
        "    config: default\n"
        "    split: test\n"
        "    metrics: [made_up_metric]\n"
    )
    with pytest.raises(KeyError, match="made_up_metric"):
        resolve(str(bad))


def test_local_no_parquet_shards(tmp_path):
    bad = tmp_path / "empty"
    (bad / "data").mkdir(parents=True)
    (bad / "eval.yaml").write_text(
        "name: X\n"
        "tasks:\n"
        "  - id: t\n"
        "    config: default\n"
        "    split: test\n"
        "    metrics: [eer_percent]\n"
    )
    with pytest.raises(FileNotFoundError, match="parquet"):
        resolve(str(bad))


def test_hf_dispatch_invokes_load_dataset(monkeypatch, tmp_path):
    """HF mode calls load_dataset and hf_hub_download with the repo id."""
    from speech_spoof_bench import loader as L

    fake_eval_path = tmp_path / "eval.yaml"
    fake_eval_path.write_text(
        "name: ASVspoof 2019 LA\n"
        "tasks:\n"
        "  - id: t\n"
        "    config: default\n"
        "    split: test\n"
        "    metrics: [eer_percent]\n"
    )

    called = {}

    def fake_load_dataset(*args, **kwargs):
        called["load_dataset"] = (args, kwargs)
        return ["row"]  # standin

    def fake_hf_hub_download(*, repo_id, filename, repo_type, **_):
        called["hf_hub_download"] = (repo_id, filename, repo_type)
        return str(fake_eval_path)

    monkeypatch.setattr(L, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(L, "hf_hub_download", fake_hf_hub_download)

    source, ds = resolve("SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA", streaming=True)
    assert source.is_local is False
    assert source.canonical_id == "SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA"
    assert source.slug == "ASVspoof2019_LA"
    assert source.display_name == "ASVspoof 2019 LA"
    assert called["hf_hub_download"] == (
        "SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA",
        "eval.yaml",
        "dataset",
    )


def test_bad_spec_neither_path_nor_repo_id():
    with pytest.raises(ValueError, match="not a directory"):
        resolve("just_a_word_no_slash_no_path")
```

- [ ] **Step 4.3: Run the loader tests to confirm they fail**

Run: `pytest tests/test_loader.py -v`
Expected: `ImportError` / `ModuleNotFoundError`.

- [ ] **Step 4.4: Implement the loader**

Create `src/speech_spoof_bench/loader.py`:

```python
"""Dataset loader — local directory or HF repo id."""

from __future__ import annotations

import glob
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from datasets import Audio, Features, Value, load_dataset
from huggingface_hub import hf_hub_download

from .metrics import is_registered

_HF_ID_RE = re.compile(r"^[^/]+/[^/]+$")


@dataclass(frozen=True)
class DatasetSource:
    spec: str
    display_name: str
    slug: str
    canonical_id: str
    metrics: list[str]
    split: str
    is_local: bool
    local_path: Path | None
    revision: str | None


def _parse_eval_yaml(eval_path: Path) -> dict[str, Any]:
    if not eval_path.is_file():
        raise FileNotFoundError(f"missing eval.yaml at {eval_path}")
    data = yaml.safe_load(eval_path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"eval.yaml at {eval_path} is not a mapping")
    name = data.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"eval.yaml at {eval_path} missing non-empty 'name'")
    tasks = data.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError(f"eval.yaml at {eval_path} missing 'tasks' list")
    task = tasks[0]
    split = task.get("split", "test")
    metrics = task.get("metrics", [])
    if not isinstance(metrics, list) or not metrics:
        raise ValueError(f"eval.yaml at {eval_path} missing task[0].metrics")
    for m in metrics:
        if not is_registered(m):
            raise KeyError(f"metric id {m!r} not registered (from {eval_path})")
    return {"name": name, "split": split, "metrics": metrics}


def _resolve_local(path: Path, streaming: bool):
    meta = _parse_eval_yaml(path / "eval.yaml")
    shards = sorted(glob.glob(str(path / "data" / "test-*.parquet")))
    if not shards:
        raise FileNotFoundError(
            f"no parquet shards under {path / 'data'} matching test-*.parquet"
        )

    # Force the audio column to decode as Audio so .array / .sampling_rate work.
    features = Features(
        {
            "path": Value("string"),
            "audio": Audio(sampling_rate=None),
            "label": Value("int64"),
            "notes": Value("string"),
        }
    )
    ds = load_dataset(
        "parquet",
        data_files={"train": shards},
        split="train",
        streaming=streaming,
        features=features,
    )

    source = DatasetSource(
        spec=str(path),
        display_name=meta["name"],
        slug=path.name,
        canonical_id=path.name,
        metrics=list(meta["metrics"]),
        split=meta["split"],
        is_local=True,
        local_path=path,
        revision=None,
    )
    return source, ds


def _resolve_hf(repo_id: str, streaming: bool):
    eval_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename="eval.yaml",
            repo_type="dataset",
        )
    )
    meta = _parse_eval_yaml(eval_path)
    ds = load_dataset(repo_id, split=meta["split"], streaming=streaming)
    source = DatasetSource(
        spec=repo_id,
        display_name=meta["name"],
        slug=repo_id.split("/")[-1],
        canonical_id=repo_id,
        metrics=list(meta["metrics"]),
        split=meta["split"],
        is_local=False,
        local_path=None,
        # TODO(phase-4): resolve revision via arena-manifest.
        revision=None,
    )
    return source, ds


def resolve(spec: str, *, streaming: bool = True):
    """Resolve a dataset spec to a (DatasetSource, IterableDataset).

    Dispatch:
      1. If ``Path(spec).is_dir()`` → local mode.
      2. Else if ``spec`` looks like ``org/name`` → HF mode.
      3. Else → ValueError.
    """
    candidate_path = Path(spec)
    if candidate_path.is_dir():
        return _resolve_local(candidate_path, streaming)
    if _HF_ID_RE.match(spec):
        return _resolve_hf(spec, streaming)
    raise ValueError(
        f"dataset spec {spec!r} is not a directory and not in <org>/<name> form"
    )
```

- [ ] **Step 4.5: Run the loader tests**

Run: `pytest tests/test_loader.py -v`
Expected: all 7 tests PASS.

- [ ] **Step 4.6: Commit**

```bash
git add src/speech_spoof_bench/loader.py tests/conftest.py tests/test_loader.py
git commit -m "feat(loader): local + HF dispatch with eval.yaml validation"
```

---

## Task 5: Runner with per-item fallback

**Files:**
- Create: `src/speech_spoof_bench/runner.py`
- Create: `tests/test_runner.py`

- [ ] **Step 5.1: Write the runner tests first**

Create `tests/test_runner.py`:

```python
"""Tests for the runner (iterate + score + per-item fallback)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from speech_spoof_bench.loader import DatasetSource
from speech_spoof_bench.model import AntiSpoofingModel
from speech_spoof_bench.runner import TooManySkips, run_dataset


class _CountingModel(AntiSpoofingModel):
    """Returns sum of audio. Subclasses override _score_batch_hook."""

    name = "counting"
    batch_size = 4

    def __init__(self):
        self.load_count = 0
        self.unload_count = 0

    def load(self):
        self.load_count += 1

    def unload(self):
        self.unload_count += 1

    def score_batch(self, audios, srs):
        return [float(a.sum()) for a, s in zip(audios, srs)]


def _make_rows(n, sr=16000, bad_index=None):
    """Produce a synthetic IterableDataset-like list of rows."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        audio = rng.standard_normal(sr // 4).astype(np.float32) * 0.01
        row = {
            "path": f"a/{i}.wav",
            "audio": {"array": audio, "sampling_rate": sr},
            "label": i % 2,
            "notes": json.dumps({"utterance_id": f"UTT_{i:04d}"}),
        }
        if bad_index is not None and i == bad_index:
            row["_force_raise"] = True
        rows.append(row)
    return rows


def _source(slug="Synth"):
    return DatasetSource(
        spec=slug,
        display_name=slug,
        slug=slug,
        canonical_id=slug,
        metrics=["eer_percent"],
        split="test",
        is_local=True,
        local_path=None,
        revision=None,
    )


def test_run_dataset_writes_scores_and_labels(tmp_path):
    m = _CountingModel()
    m.load()
    res = run_dataset(m, _source(), _make_rows(8), tmp_path)
    m.unload()

    assert res.scores_path == tmp_path / "scores.txt"
    lines = res.scores_path.read_text().strip().splitlines()
    assert len(lines) == 8
    # Each line: "<utt_id> <score>"
    for line in lines:
        utt_id, score = line.split()
        assert utt_id.startswith("UTT_")
        float(score)  # parseable

    assert res.n_total == 8
    assert res.n_skipped == 0
    assert len(res.labels) == 8
    assert set(res.labels.values()) == {0, 1}


def test_per_item_skip_only_offender_in_multi_item_batch(tmp_path):
    class M(_CountingModel):
        def score_batch(self, audios, srs):
            # Raises if any item is "tagged" (length 0) OR if batch contains a tagged one.
            # We tag by appending a NaN sentinel through the audio array.
            if any(np.isnan(a).any() for a in audios):
                raise RuntimeError("batch contains bad item")
            return [float(a.sum()) for a in audios]

    rows = _make_rows(8)
    # Tag row 3 with a NaN-containing audio.
    rows[3]["audio"]["array"] = np.full(100, np.nan, dtype=np.float32)

    m = M()
    m.load()
    res = run_dataset(m, _source(), rows, tmp_path)
    m.unload()

    # Item 3 should be skipped; the other 7 scored successfully.
    assert res.n_total == 8
    assert res.n_skipped == 1
    lines = res.scores_path.read_text().strip().splitlines()
    assert len(lines) == 7
    assert "UTT_0003" not in res.scores_path.read_text()


def test_flaky_batch_recovers_via_single_item_calls(tmp_path):
    """Model fails on any batch > 1 but succeeds individually."""

    class M(_CountingModel):
        def score_batch(self, audios, srs):
            if len(audios) > 1:
                raise RuntimeError("only batch size 1 supported")
            return [float(audios[0].sum())]

    m = M()
    m.load()
    res = run_dataset(m, _source(), _make_rows(8), tmp_path)
    m.unload()

    assert res.n_total == 8
    assert res.n_skipped == 0
    lines = res.scores_path.read_text().strip().splitlines()
    assert len(lines) == 8


def test_too_many_skips_raises(tmp_path):
    class M(_CountingModel):
        def score_batch(self, audios, srs):
            raise RuntimeError("always broken")

    m = M()
    m.load()
    with pytest.raises(TooManySkips):
        run_dataset(m, _source(), _make_rows(20), tmp_path)
```

- [ ] **Step 5.2: Run the runner tests to confirm they fail**

Run: `pytest tests/test_runner.py -v`
Expected: `ImportError` for `speech_spoof_bench.runner`.

- [ ] **Step 5.3: Implement the runner**

Create `src/speech_spoof_bench/runner.py`:

```python
"""Iterate a dataset, score each utterance via the model, write scores.txt.

Lifecycle note: ``run_dataset`` does NOT call ``model.load()`` or
``model.unload()``. The orchestrator (``Benchmark.run``) does that exactly
once per evaluation, around the loop over datasets.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.signal import resample_poly

from .loader import DatasetSource
from .model import AntiSpoofingModel

_LOG = logging.getLogger(__name__)

SKIP_FRACTION_THRESHOLD = 0.05


class TooManySkips(RuntimeError):
    """Raised when >5% of items in a dataset failed to score."""


@dataclass
class RunResult:
    scores_path: Path
    labels: dict[str, int] = field(default_factory=dict)
    n_total: int = 0
    n_skipped: int = 0


def _to_float32_mono_16k(audio_array: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    a = np.asarray(audio_array, dtype=np.float32)
    if a.ndim == 2:
        # average channels (defensive — datasets are mono per §1.2)
        a = a.mean(axis=0).astype(np.float32)
    if sr != target_sr:
        # resample_poly wants integer up/down; use rationalized form via gcd.
        from math import gcd
        g = gcd(int(sr), int(target_sr))
        up, down = int(target_sr // g), int(sr // g)
        a = resample_poly(a, up, down).astype(np.float32)
    return a


def _extract(row: dict[str, Any], target_sr: int) -> tuple[str, np.ndarray, int, int]:
    notes = json.loads(row["notes"])
    utt_id = notes["utterance_id"]
    audio = row["audio"]
    array = audio["array"]
    sr = int(audio["sampling_rate"])
    label = int(row["label"])
    array = _to_float32_mono_16k(array, sr, target_sr)
    return utt_id, array, target_sr, label


def _score_with_fallback(
    model: AntiSpoofingModel,
    audios: list[np.ndarray],
    srs: list[int],
) -> list[float | None]:
    """Score a chunk. On batch-wide exception, fall back to per-item.

    Returns a list of length len(audios). Elements that could not be scored
    even individually are None.
    """
    if len(audios) == 1:
        try:
            return [float(model.score_batch(audios, srs)[0])]
        except Exception as exc:
            _LOG.warning("score_batch failed on single item: %s", exc)
            return [None]

    try:
        out = model.score_batch(audios, srs)
        return [float(x) for x in out]
    except Exception as exc:
        _LOG.debug("multi-item batch failed (%s); falling back to per-item", exc)

    results: list[float | None] = []
    for a, s in zip(audios, srs):
        try:
            results.append(float(model.score_batch([a], [s])[0]))
        except Exception as exc:
            _LOG.warning("score_batch failed on single item during fallback: %s", exc)
            results.append(None)
    return results


def run_dataset(
    model: AntiSpoofingModel,
    source: DatasetSource,
    dataset: Iterable[dict[str, Any]],
    output_dir: Path,
) -> RunResult:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scores_path = output_dir / "scores.txt"

    result = RunResult(scores_path=scores_path)
    bs = max(1, int(model.batch_size))
    target_sr = int(model.expected_sample_rate)

    buf_utt: list[str] = []
    buf_audio: list[np.ndarray] = []
    buf_sr: list[int] = []
    buf_label: list[int] = []

    def _flush(out_f) -> None:
        if not buf_utt:
            return
        scores = _score_with_fallback(model, buf_audio, buf_sr)
        for utt_id, label, score in zip(buf_utt, buf_label, scores):
            result.labels[utt_id] = label
            result.n_total += 1
            if score is None:
                result.n_skipped += 1
                continue
            out_f.write(f"{utt_id} {score:.6f}\n")
        buf_utt.clear()
        buf_audio.clear()
        buf_sr.clear()
        buf_label.clear()

    with scores_path.open("w") as out_f:
        for row in dataset:
            utt_id, array, sr, label = _extract(row, target_sr)
            buf_utt.append(utt_id)
            buf_audio.append(array)
            buf_sr.append(sr)
            buf_label.append(label)
            if len(buf_utt) >= bs:
                _flush(out_f)
        _flush(out_f)

    if result.n_total > 0 and result.n_skipped / result.n_total > SKIP_FRACTION_THRESHOLD:
        raise TooManySkips(
            f"dataset {source.slug!r}: {result.n_skipped}/{result.n_total} items "
            f"skipped (> {SKIP_FRACTION_THRESHOLD:.0%})"
        )

    return result
```

- [ ] **Step 5.4: Run the runner tests**

Run: `pytest tests/test_runner.py -v`
Expected: all 4 tests PASS.

- [ ] **Step 5.5: Commit**

```bash
git add src/speech_spoof_bench/runner.py tests/test_runner.py
git commit -m "feat(runner): iterate + score with per-item fallback on batch errors"
```

---

## Task 6: Benchmark orchestrator

**Files:**
- Create: `src/speech_spoof_bench/benchmark.py`
- Create: `tests/test_benchmark.py`

- [ ] **Step 6.1: Write the orchestrator tests**

Create `tests/test_benchmark.py`:

```python
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
```

- [ ] **Step 6.2: Run the tests to confirm they fail**

Run: `pytest tests/test_benchmark.py -v`
Expected: `ImportError` for `speech_spoof_bench.benchmark`.

- [ ] **Step 6.3: Implement `Benchmark`**

Create `src/speech_spoof_bench/benchmark.py`:

```python
"""Orchestrator: load → run → score → write result.yaml."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from . import __version__ as _BENCH_VERSION
from .cache import purge_hf_cache
from .loader import resolve
from .metrics import get_metric
from .model import AntiSpoofingModel
from .runner import run_dataset

_LOG = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    dataset_slug: str
    canonical_id: str
    metrics: dict[str, float] = field(default_factory=dict)
    metric_extras: dict[str, dict[str, Any]] = field(default_factory=dict)
    n_trials: int = 0
    n_skipped: int = 0
    scores_path: Path | None = None
    result_yaml_path: Path | None = None


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_result_yaml(
    *,
    out_path: Path,
    model: AntiSpoofingModel,
    source,
    metrics: dict[str, float],
    n_trials: int,
    n_skipped: int,
    scores_sha256: str,
) -> None:
    payload: dict[str, Any] = {
        "schema_version": 4,
        "system": {
            "name": getattr(model, "name", "unknown"),
            "slug": None,
            "description": None,
            "code": None,
            "checkpoint": None,
            "paper": None,
        },
        "dataset": {
            "id": source.canonical_id,
            "revision": source.revision,
            "split": source.split,
        },
        "scores": {
            **{k: float(v) for k, v in metrics.items()},
            "n_trials": int(n_trials),
            "n_skipped": int(n_skipped),
        },
        "artifact": {
            "scores_url": None,
            "scores_sha256": scores_sha256,
            "bench_version": f"speech-spoof-bench=={_BENCH_VERSION}",
        },
        "reproduction": {},
        "submitter": {},
        "submitted_at": None,
        "notes": None,
    }
    out_path.write_text(yaml.safe_dump(payload, sort_keys=False))


class Benchmark:
    """Static entry point. ``Benchmark.run(model, datasets=...)``."""

    @staticmethod
    def run(
        model: AntiSpoofingModel,
        datasets: list[str] | str = "all",
        output_dir: str | Path = "./results",
        *,
        streaming: bool = True,
        cleanup: bool = True,
        skip_existing: bool = True,
    ) -> dict[str, BenchmarkResult]:
        if isinstance(datasets, str) and datasets == "all":
            raise NotImplementedError(
                "datasets='all' requires the arena manifest; lands in phase 4"
            )
        if isinstance(datasets, str):
            datasets = [datasets]
        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)

        results: dict[str, BenchmarkResult] = {}

        model.load()
        try:
            for spec in datasets:
                source, ds = resolve(spec, streaming=streaming)
                out = output_root / source.slug
                result_yaml = out / "result.yaml"

                if skip_existing and result_yaml.exists():
                    parsed = yaml.safe_load(result_yaml.read_text()) or {}
                    if parsed.get("dataset", {}).get("revision") == source.revision:
                        _LOG.info("skipping %s (result.yaml present)", source.slug)
                        # Surface the existing metrics in the return value.
                        existing_scores = parsed.get("scores", {}) or {}
                        results[source.slug] = BenchmarkResult(
                            dataset_slug=source.slug,
                            canonical_id=source.canonical_id,
                            metrics={
                                k: float(v)
                                for k, v in existing_scores.items()
                                if k not in {"n_trials", "n_skipped"}
                                and isinstance(v, (int, float))
                            },
                            n_trials=int(existing_scores.get("n_trials", 0)),
                            n_skipped=int(existing_scores.get("n_skipped", 0)),
                            scores_path=out / "scores.txt",
                            result_yaml_path=result_yaml,
                        )
                        continue

                run_res = run_dataset(model, source, ds, out)

                # Compute all metrics declared by the dataset's eval.yaml.
                scores_map = _load_scores_txt(run_res.scores_path)
                metric_values: dict[str, float] = {}
                metric_extras: dict[str, dict[str, Any]] = {}
                for mid in source.metrics:
                    spec = get_metric(mid)
                    mr = spec.fn(scores_map, run_res.labels)
                    metric_values[mid] = mr.value
                    metric_extras[mid] = dict(mr.extras)

                scores_sha256 = _sha256_of_file(run_res.scores_path)
                _write_result_yaml(
                    out_path=result_yaml,
                    model=model,
                    source=source,
                    metrics=metric_values,
                    n_trials=run_res.n_total,
                    n_skipped=run_res.n_skipped,
                    scores_sha256=scores_sha256,
                )

                results[source.slug] = BenchmarkResult(
                    dataset_slug=source.slug,
                    canonical_id=source.canonical_id,
                    metrics=metric_values,
                    metric_extras=metric_extras,
                    n_trials=run_res.n_total,
                    n_skipped=run_res.n_skipped,
                    scores_path=run_res.scores_path,
                    result_yaml_path=result_yaml,
                )

                if cleanup and not source.is_local:
                    purge_hf_cache(source.canonical_id)
        finally:
            model.unload()

        return results


def _load_scores_txt(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        utt_id, score = line.split()
        out[utt_id] = float(score)
    return out
```

- [ ] **Step 6.4: Run the benchmark tests**

Run: `pytest tests/test_benchmark.py -v`
Expected: all 5 tests PASS.

- [ ] **Step 6.5: Commit**

```bash
git add src/speech_spoof_bench/benchmark.py tests/test_benchmark.py
git commit -m "feat(benchmark): orchestrator with once-per-eval load/unload"
```

---

## Task 7: Cache purger

**Files:**
- Create: `src/speech_spoof_bench/cache.py`

- [ ] **Step 7.1: Implement `purge_hf_cache`**

Create `src/speech_spoof_bench/cache.py`:

```python
"""HF cache purger. No-op for repos not present in the cache."""

from __future__ import annotations

import logging

from huggingface_hub import scan_cache_dir

_LOG = logging.getLogger(__name__)


def purge_hf_cache(repo_id: str) -> None:
    """Remove all cached revisions for ``repo_id`` (datasets only).

    Silent no-op if the repo isn't in the cache.
    """
    info = scan_cache_dir()
    revisions_to_delete: list[str] = []
    for repo in info.repos:
        if repo.repo_id == repo_id and repo.repo_type == "dataset":
            revisions_to_delete.extend(rev.commit_hash for rev in repo.revisions)
    if not revisions_to_delete:
        _LOG.debug("no cached revisions for %s; nothing to purge", repo_id)
        return
    delete_strategy = info.delete_revisions(*revisions_to_delete)
    _LOG.info(
        "purging %d revisions for %s (~%s)",
        len(revisions_to_delete),
        repo_id,
        delete_strategy.expected_freed_size_str,
    )
    delete_strategy.execute()
```

- [ ] **Step 7.2: Smoke-test the import**

Run: `python -c "from speech_spoof_bench.cache import purge_hf_cache; purge_hf_cache('definitely/not-in-cache')"`
Expected: no error, no output. (DEBUG log message is suppressed at default level.)

- [ ] **Step 7.3: Commit**

```bash
git add src/speech_spoof_bench/cache.py
git commit -m "feat(cache): purge_hf_cache helper"
```

---

## Task 8: Manifest stub

**Files:**
- Create: `src/speech_spoof_bench/manifest.py`

- [ ] **Step 8.1: Implement the stub**

Create `src/speech_spoof_bench/manifest.py`:

```python
"""arena-manifest reader. Stubbed at phase 2; implemented in phase 4."""

from __future__ import annotations


def fetch_manifest() -> dict:
    raise NotImplementedError(
        "arena-manifest support lands in phase 4 of the roadmap"
    )
```

- [ ] **Step 8.2: Commit**

```bash
git add src/speech_spoof_bench/manifest.py
git commit -m "feat(manifest): stub for phase 4"
```

---

## Task 9: CLI

**Files:**
- Create: `src/speech_spoof_bench/cli.py`
- Create: `tests/test_cli.py`

- [ ] **Step 9.1: Write CLI tests**

Create `tests/test_cli.py`:

```python
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
```

Note: `test_cli_run_against_local` depends on the `RandomBaseline` example being importable, which arrives in Task 10. Mark it `xfail` until then if executing tasks linearly; pytest will pick it up automatically after Task 10.

- [ ] **Step 9.2: Implement the CLI**

Create `src/speech_spoof_bench/cli.py`:

```python
"""Command-line interface for speech-spoof-bench."""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
from pathlib import Path
from typing import Sequence

from .benchmark import Benchmark
from .loader import resolve


def _import_model_class(spec: str):
    if ":" not in spec:
        raise SystemExit(
            f"--model-module must be <module>:<ClassName>, got {spec!r}"
        )
    mod_name, cls_name = spec.split(":", 1)
    module = importlib.import_module(mod_name)
    try:
        return getattr(module, cls_name)
    except AttributeError:
        raise SystemExit(f"class {cls_name!r} not found in module {mod_name!r}")


def _cmd_run(args: argparse.Namespace) -> int:
    cls = _import_model_class(args.model_module)
    model = cls()
    Benchmark.run(
        model,
        datasets=list(args.datasets),
        output_dir=args.output_dir,
        streaming=not args.no_streaming,
        cleanup=not args.no_cleanup,
        skip_existing=not args.no_skip_existing,
    )
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    print("listing manifest datasets is part of phase 4", file=sys.stderr)
    return 2


def _cmd_validate_dataset(args: argparse.Namespace) -> int:
    source, ds = resolve(args.spec, streaming=True)
    first = next(iter(ds))
    expected = {"path", "audio", "label", "notes"}
    actual = set(first.keys())
    if not expected.issubset(actual):
        raise SystemExit(
            f"dataset row missing required columns: {expected - actual} "
            f"(got {sorted(actual)})"
        )
    import json
    notes = json.loads(first["notes"])
    if not notes.get("utterance_id"):
        raise SystemExit("first row's notes JSON has no non-empty 'utterance_id'")
    print(f"OK: {source.canonical_id} (display: {source.display_name!r})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="speech-spoof-bench")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="run a model against one or more datasets")
    run.add_argument("--model-module", required=True,
                     help="module:ClassName, e.g. mypkg.mymod:MyModel")
    run.add_argument("--datasets", action="append", required=True,
                     help="local dir path OR org/name HF repo id; repeatable")
    run.add_argument("--output-dir", default="./results")
    run.add_argument("--no-streaming", action="store_true")
    run.add_argument("--no-cleanup", action="store_true")
    run.add_argument("--no-skip-existing", action="store_true")
    run.set_defaults(func=_cmd_run)

    lst = sub.add_parser("list", help="list datasets in the arena manifest")
    lst.set_defaults(func=_cmd_list)

    vd = sub.add_parser("validate-dataset",
                        help="quick check that a dataset loads with the v4 schema")
    vd.add_argument("spec", help="local dir path or org/name HF repo id")
    vd.set_defaults(func=_cmd_validate_dataset)

    return p


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 9.3: Smoke-test the CLI**

Run: `speech-spoof-bench --help`
Expected: prints subcommand usage.

Run: `speech-spoof-bench list`
Expected: prints "listing manifest datasets is part of phase 4" and exits non-zero.

- [ ] **Step 9.4: Commit**

```bash
git add src/speech_spoof_bench/cli.py tests/test_cli.py
git commit -m "feat(cli): run, list (stub), validate-dataset"
```

---

## Task 10: Random-baseline example

**Files:**
- Create: `src/speech_spoof_bench/examples/__init__.py`
- Create: `src/speech_spoof_bench/examples/random_baseline.py`

- [ ] **Step 10.1: Create the examples sub-package**

Create `src/speech_spoof_bench/examples/__init__.py` (empty).

- [ ] **Step 10.2: Implement RandomBaseline**

Create `src/speech_spoof_bench/examples/random_baseline.py`:

```python
"""Reference random baseline. Always returns N(0,1) for every utterance.

Useful as a smoke test: EER should land near 50% on any balanced split.
"""

from __future__ import annotations

import numpy as np

from speech_spoof_bench.model import SimpleAntiSpoofingModel


class RandomBaseline(SimpleAntiSpoofingModel):
    name = "random-baseline"

    def __init__(self, seed: int = 0):
        self._seed = seed
        self._rng: np.random.Generator | None = None

    def load(self) -> None:
        self._rng = np.random.default_rng(self._seed)

    def unload(self) -> None:
        self._rng = None

    def score(self, audio: np.ndarray, sr: int) -> float:
        assert self._rng is not None, "load() must be called before score()"
        return float(self._rng.standard_normal())
```

- [ ] **Step 10.3: Re-run the full test suite to catch the CLI test that referenced this**

Run: `pytest -v`
Expected: all tests PASS (including `test_cli_run_against_local`).

- [ ] **Step 10.4: Commit**

```bash
git add src/speech_spoof_bench/examples/
git commit -m "feat(examples): random baseline"
```

---

## Task 11: End-to-end smoke test against the real local LA dataset

This is **manual verification only** — not a CI test. The goal is the Phase 2 DoD: a working `scores.txt` + `result.yaml` produced from the real local mount, without any HF download.

- [ ] **Step 11.1: Run the random baseline against the local LA dataset**

Run:

```bash
speech-spoof-bench run \
    --model-module speech_spoof_bench.examples.random_baseline:RandomBaseline \
    --datasets /home/kirill/mnt/drive3_8tb/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA \
    --output-dir ./results \
    --no-cleanup
```

Expected output (timing depends on disk):
- Logs show "loading parquet shards from .../data" (or similar) and per-batch progress.
- No HTTP requests to `huggingface.co` (verify by sniffing or just trusting the loader's local branch — no auth is required).
- Process exits 0.

- [ ] **Step 11.2: Sanity-check the outputs**

Run:

```bash
wc -l ./results/ASVspoof2019_LA/scores.txt
head -3 ./results/ASVspoof2019_LA/scores.txt
```

Expected: ~71237 lines (the published LA eval count). Each line: `<utterance_id> <float>`.

Run:

```bash
cat ./results/ASVspoof2019_LA/result.yaml
```

Expected:
- `schema_version: 4`
- `dataset.id: ASVspoof2019_LA`
- `dataset.revision: null`
- `dataset.split: test`
- `scores.eer_percent` is a float in roughly `[40, 60]` (random baseline jitter)
- `scores.n_trials: 71237`
- `scores.n_skipped: 0`
- `artifact.scores_sha256` is a 64-char hex string
- `artifact.bench_version: speech-spoof-bench==0.1.0`
- `reproduction: {}` and `submitter: {}` left empty

- [ ] **Step 11.3: Verify resumability**

Run the same command from Step 11.1 again (without `--no-skip-existing`).
Expected: completes in <1s; logs say "skipping ASVspoof2019_LA (result.yaml present)".

- [ ] **Step 11.4: Verify HF dispatch still works (no actual download needed)**

Run:

```bash
speech-spoof-bench validate-dataset SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA
```

Expected: connects to HF, prints `OK: SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA (display: 'ASVspoof 2019 LA')`. (This downloads the first parquet shard — small price for confidence.)

- [ ] **Step 11.5: Record the smoke-test in ROADMAP.md**

Edit `docs/roadmap/ROADMAP.md` — under the Phase 2 section, check the boxes for all items now completed. Do NOT alter Phase 6 (that's its own gated milestone).

Run: `git diff docs/roadmap/ROADMAP.md`
Expected: only Phase 2 checkboxes flipped from `[ ]` to `[x]`.

- [ ] **Step 11.6: Final commit**

```bash
git add docs/roadmap/ROADMAP.md
git commit -m "docs: phase 2 complete — pip skeleton runs end-to-end on local LA"
```

---

## Done when

- All `pytest` tests pass with no skips except the ones noted.
- `speech-spoof-bench run` against the local LA path produces a valid `result.yaml` with `eer_percent` near 50%.
- Re-running with `--skip-existing` (default) is instant.
- No HTTP request to `huggingface.co` is required at any step of Task 11.1–11.3.
- `pip install -e .` works on a fresh venv with no torch dependency pulled in.
- The commit log on the pip-package repo shows one clean commit per task.
