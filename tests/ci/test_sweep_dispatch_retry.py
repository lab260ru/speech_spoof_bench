"""Tests for the GitHub workflow_dispatch retry in sweep._dispatch_verify_workflow.

A burst of merges can hit GitHub's secondary rate limit (429) or a transient 5xx
on the dispatch POST. Without retry the sweep drops that PR for a whole cycle (15
min). With bounded retry the dispatch self-heals within the run. A bare 403
(auth/permission) is NOT retried — retrying a permission error just wastes time.
"""
from __future__ import annotations

import io
from urllib.error import HTTPError, URLError

import pytest

from speech_spoof_bench.ci import sweep


@pytest.fixture(autouse=True)
def _fast_and_tokened(monkeypatch):
    monkeypatch.setenv("GH_TOKEN", "tok")
    monkeypatch.setattr(sweep, "_sleep", lambda s: None)


def _http_error(code, *, retry_after=None):
    hdrs = {}
    if retry_after is not None:
        hdrs["Retry-After"] = str(retry_after)
    return HTTPError("https://api.github.com/x", code, "err", hdrs, io.BytesIO(b"body"))


def _urlopen_seq(monkeypatch, *outcomes):
    """Each outcome is either an exception to raise or a sentinel to return."""
    calls = {"n": 0}

    def fake(req, timeout=None):
        i = calls["n"]
        calls["n"] += 1
        outcome = outcomes[min(i, len(outcomes) - 1)]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(sweep.urllib.request, "urlopen", fake)
    return calls


def test_retries_on_503_then_succeeds(monkeypatch):
    calls = _urlopen_seq(monkeypatch, _http_error(503), object())
    assert sweep._dispatch_verify_workflow("Org/A", 1) is True
    assert calls["n"] == 2


def test_retries_on_429_then_succeeds(monkeypatch):
    calls = _urlopen_seq(monkeypatch, _http_error(429, retry_after=1), object())
    assert sweep._dispatch_verify_workflow("Org/A", 1) is True
    assert calls["n"] == 2


def test_gives_up_after_attempts_on_persistent_5xx(monkeypatch):
    calls = _urlopen_seq(monkeypatch, _http_error(502))
    assert sweep._dispatch_verify_workflow("Org/A", 1) is False
    assert calls["n"] == sweep._DISPATCH_ATTEMPTS


def test_does_not_retry_bare_403_auth_error(monkeypatch):
    calls = _urlopen_seq(monkeypatch, _http_error(403))
    assert sweep._dispatch_verify_workflow("Org/A", 1) is False
    assert calls["n"] == 1  # auth failure → no point retrying


def test_retries_403_with_retry_after_secondary_limit(monkeypatch):
    calls = _urlopen_seq(monkeypatch, _http_error(403, retry_after=1), object())
    assert sweep._dispatch_verify_workflow("Org/A", 1) is True
    assert calls["n"] == 2


def test_retries_on_transient_network_error(monkeypatch):
    calls = _urlopen_seq(monkeypatch, URLError("connection reset"), object())
    assert sweep._dispatch_verify_workflow("Org/A", 1) is True
    assert calls["n"] == 2


def test_no_token_returns_false_without_calling(monkeypatch):
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.delenv("GH_PAT", raising=False)
    calls = _urlopen_seq(monkeypatch, object())
    assert sweep._dispatch_verify_workflow("Org/A", 1) is False
    assert calls["n"] == 0


def test_backoff_delay_never_negative():
    # A malformed negative Retry-After must never become a negative sleep
    # (time.sleep(<0) raises ValueError → would crash the whole sweep).
    assert sweep._backoff_delay(0, -30.0) >= 0.0
    assert sweep._backoff_delay(3, -1.0) >= 0.0
    assert sweep._backoff_delay(0, None) >= 0.0


def test_negative_retry_after_does_not_crash_dispatch(monkeypatch):
    """End-to-end: a 429 carrying a negative Retry-After must honor the
    'never raises, returns bool' contract — not blow up time.sleep."""
    def checking_sleep(s):
        assert s >= 0.0, f"sleep got a negative delay: {s}"
    monkeypatch.setattr(sweep, "_sleep", checking_sleep)
    calls = _urlopen_seq(monkeypatch, _http_error(429, retry_after=-30))
    assert sweep._dispatch_verify_workflow("Org/A", 1) is False
    assert calls["n"] == sweep._DISPATCH_ATTEMPTS
