"""1 - Spoof Rejection Rate (spoof-only-safe, lower-is-better).

Given a fixed operating threshold t* (transferred from an external calibration
dataset, e.g. DeepVoice -- see proposal section 5), SRR = mean(spoof_score < t*)
and the metric value = (1 - SRR) * 100. Convention (matches eer.py): higher
score = more bonafide, label 0 = bonafide, 1 = spoof. Only spoof rows are used,
so a single-class (spoof-only) dataset never raises.
"""

from __future__ import annotations

import numpy as np

from . import MetricResult, register_metric


@register_metric(
    id="srr_complement",
    display_name="1-SRR (%)",
    lower_is_better=True,
    requires_audio=False,
    wants_config=True,
)
def compute_srr_complement(
    scores: dict[str, float],
    labels: dict[str, int],
    config: dict | None = None,
) -> MetricResult:
    config = config or {}
    if config.get("threshold") is None:
        raise ValueError(
            "srr_complement requires config['threshold'] (t* from the "
            "calibration source dataset, e.g. DeepVoice)"
        )
    try:
        t = float(config["threshold"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"srr_complement threshold must be a number: {exc}") from exc
    if not np.isfinite(t):
        raise ValueError("srr_complement threshold must be finite (got non-finite)")
    spoof = np.array(
        [scores[u] for u, y in labels.items() if y == 1 and u in scores],
        dtype=np.float64,
    )
    if spoof.size == 0:
        raise ValueError("srr_complement needs at least one spoof score")
    if not np.all(np.isfinite(spoof)):
        raise ValueError("srr_complement got non-finite spoof scores (NaN/inf)")
    srr = float(np.mean(spoof < t))  # fraction of spoof correctly rejected
    value = (1.0 - srr) * 100.0
    return MetricResult(
        value=value,
        extras={"threshold": t, "n_spoof": int(spoof.size), "srr": srr},
    )
