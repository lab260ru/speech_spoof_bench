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
