"""Badge-layer string builders (Phase 9).

Pure functions: input parsed dicts, output strings. No I/O.
"""

from __future__ import annotations


class BadgeError(Exception):
    """Raised on input that cannot produce a valid result.yaml or comment."""


def _color_for_eer(eer_percent: float) -> str:
    if eer_percent < 2.0:
        return "brightgreen"
    if eer_percent < 5.0:
        return "green"
    if eer_percent < 10.0:
        return "yellow"
    return "lightgrey"
