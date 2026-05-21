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
