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
