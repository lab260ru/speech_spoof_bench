"""Tests for speech_spoof_bench.ci._hf_retry.retry_on_429."""
from __future__ import annotations

import httpx
import pytest
from huggingface_hub.errors import HfHubHTTPError

from speech_spoof_bench.ci._hf_retry import retry_on_429


def _err(status, *, retry_after=None):
    headers = {"Retry-After": str(retry_after)} if retry_after is not None else {}
    resp = httpx.Response(status, headers=headers,
                          request=httpx.Request("GET", "https://hf.co/x"))
    return HfHubHTTPError(f"{status}", response=resp)


def test_retries_429_then_succeeds():
    slept = []
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise _err(429)
        return "ok"

    out = retry_on_429(flaky, _base=0.01, _sleep=slept.append)
    assert out == "ok"
    assert calls["n"] == 3
    assert len(slept) == 2  # slept before each of the 2 retries


def test_non_429_reraises_immediately():
    def boom():
        raise _err(500)

    with pytest.raises(HfHubHTTPError):
        retry_on_429(boom, _attempts=6, _sleep=lambda s: None)


def test_exhausts_attempts_then_raises():
    calls = {"n": 0}

    def always():
        calls["n"] += 1
        raise _err(429)

    with pytest.raises(HfHubHTTPError):
        retry_on_429(always, _attempts=4, _base=0.01, _sleep=lambda s: None)
    assert calls["n"] == 4


def test_backoff_uses_equal_jitter():
    """No Retry-After → delay is ceil/2 + rand*ceil/2 with ceil=min(cap, base*2**n)."""
    slept = []
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 4:
            raise _err(429)
        return "ok"

    # _rand fixed at 0.5 → delay == ceil * 0.75; base=2 → ceil = 2,4,8 for n=0,1,2
    out = retry_on_429(flaky, _base=2.0, _cap=120.0, _rand=lambda: 0.5,
                       _sleep=slept.append)
    assert out == "ok"
    assert slept == [2 * 0.75, 4 * 0.75, 8 * 0.75]


def test_backoff_respects_cap():
    slept = []

    def always():
        raise _err(429)

    with pytest.raises(HfHubHTTPError):
        # ceil caps at _cap=10 once base*2**n exceeds it; _rand=1.0 → delay==ceil
        retry_on_429(always, _attempts=8, _base=2.0, _cap=10.0,
                     _rand=lambda: 1.0, _sleep=slept.append)
    assert max(slept) <= 10.0
    assert slept[-1] == 10.0  # later attempts pinned to the cap


def test_default_attempts_is_eight():
    calls = {"n": 0}

    def always():
        calls["n"] += 1
        raise _err(429)

    with pytest.raises(HfHubHTTPError):
        retry_on_429(always, _base=0.0, _sleep=lambda s: None)
    assert calls["n"] == 8


def test_honors_retry_after_header():
    slept = []
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 2:
            raise _err(429, retry_after=7)
        return "done"

    retry_on_429(flaky, _base=0.01, _cap=60.0, _sleep=slept.append)
    assert slept == [7.0]


def test_passes_through_args_and_kwargs():
    def add(a, b, *, c):
        return a + b + c

    assert retry_on_429(add, 1, 2, c=3) == 6
