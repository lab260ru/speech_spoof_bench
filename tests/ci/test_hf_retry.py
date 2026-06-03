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
