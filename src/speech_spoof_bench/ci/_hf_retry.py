"""Retry HF API calls on 429 Too Many Requests.

HF throttles at the account/IP/edge level. When several CI runs hit the API
near-simultaneously (e.g. multiple PRs merged at once) individual calls can
return 429 even after the workflows are serialized. This wraps a single HF
call with bounded exponential backoff (honoring Retry-After) so transient
throttling self-heals instead of failing the job.

The backoff uses **equal jitter** (half fixed + half random) so that several
runs retrying at once don't synchronize their wake-ups (thundering herd) while
still keeping a non-zero floor under sustained throttle. ``Retry-After`` is
honored exactly (no jitter) when the server provides it.
"""

from __future__ import annotations

import logging
import random
import time

logger = logging.getLogger(__name__)


def _status_of(exc: Exception) -> int | None:
    resp = getattr(exc, "response", None)
    return getattr(resp, "status_code", None)


def _retry_after_of(exc: Exception) -> float | None:
    resp = getattr(exc, "response", None)
    headers = getattr(resp, "headers", None) or {}
    raw = headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def retry_on_429(fn, /, *args, _attempts=8, _base=2.0, _cap=120.0,
                 _sleep=time.sleep, _rand=random.random, **kwargs):
    """Call ``fn(*args, **kwargs)``, retrying only on HF 429.

    - Re-raises immediately on any non-429 error (including HTTP errors with a
      different status, which are not throttling).
    - Honors the ``Retry-After`` response header **exactly** when present.
    - Otherwise sleeps with **equal jitter**: ``ceil = min(_cap, _base * 2**n)``
      then ``ceil/2 + rand*ceil/2`` before retry ``n`` (0-indexed) — a random
      delay in ``[ceil/2, ceil]`` that desynchronizes concurrent retriers while
      keeping a floor.
    - After ``_attempts`` total tries, re-raises the last 429. Defaults give a
      total backoff budget of a few minutes (vs ~1 min before), which survives
      sustained account-level throttling.
    - ``_sleep`` and ``_rand`` are injectable so tests run instantly and
      deterministically.
    """
    for attempt in range(_attempts):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 — narrowed by status check below
            if _status_of(exc) != 429:
                raise
            if attempt == _attempts - 1:
                logger.warning("429 from HF after %d attempts; giving up", _attempts)
                raise
            delay = _retry_after_of(exc)
            if delay is None:
                ceil = min(_cap, _base * (2 ** attempt))
                delay = ceil / 2 + _rand() * (ceil / 2)
            logger.info("HF 429 (attempt %d/%d); sleeping %.1fs",
                        attempt + 1, _attempts, delay)
            _sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover
