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
