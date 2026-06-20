"""`speech-spoof-bench ci sweep` — self-healing backstop for dropped verify-pr runs.

The HF→GitHub webhook dispatches `verify-pr` into a `concurrency: hf-ci` group
that GitHub caps at **1 running + 1 pending**. A burst of submission PRs (one
model submitted across N datasets at once) loses the extra dispatches with no
self-heal — they sit forever until someone manually re-fires them.

This sweep, run on a schedule, finds open PRs on the dataset repos that have **no
verify-pr verdict yet** and re-dispatches `verify-pr` for them. It dispatches at
most ``max_dispatch`` per run (default **1**) so it never re-creates the burst
eviction; successive scheduled runs drain any backlog one at a time. A PR that is
genuinely not a submission change just gets a one-time "no submission changes"
verdict from verify-pr and is then skipped on future sweeps.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
import urllib.error
import urllib.request

from huggingface_hub import HfApi

from ._hf_retry import retry_on_429

logger = logging.getLogger(__name__)

# Any verify-pr comment (a real verdict or "no submission changes") carries this.
_VERDICT_MARKER = "speech-spoof-bench ci verify-pr"
_GH_API = "https://api.github.com"
_DEFAULT_TARGET = "lab260ru/speech_spoof_bench"

# GitHub workflow_dispatch retry policy. The dispatch POST can hit GitHub's
# secondary rate limit (429, sometimes 403+Retry-After) or a transient 5xx
# during a merge burst; bounded retry lets it self-heal within the run instead
# of dropping the PR for a whole 15-min sweep cycle.
_DISPATCH_ATTEMPTS = 5
_DISPATCH_BASE = 2.0
_DISPATCH_CAP = 60.0
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


def _sleep(seconds: float) -> None:  # indirection so tests run instantly
    time.sleep(seconds)


def _retry_after(exc: urllib.error.HTTPError) -> float | None:
    headers = getattr(exc, "headers", None)
    if not headers:
        return None
    raw = headers.get("Retry-After")
    try:
        return float(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def _http_error_retryable(exc: urllib.error.HTTPError) -> bool:
    code = getattr(exc, "code", None)
    if code in _RETRYABLE_STATUS:
        return True
    # GitHub's secondary rate limit can surface as a 403 carrying Retry-After.
    # A *bare* 403 is an auth/permission failure — retrying it just wastes time.
    return code == 403 and _retry_after(exc) is not None


def _backoff_delay(attempt: int, retry_after: float | None) -> float:
    if retry_after is not None:
        # Clamp: a malformed/negative Retry-After must never become a negative
        # sleep (time.sleep(<0) raises ValueError and would crash the sweep).
        return max(0.0, min(_DISPATCH_CAP, retry_after))
    ceil = min(_DISPATCH_CAP, _DISPATCH_BASE * (2 ** attempt))
    return ceil / 2 + random.random() * (ceil / 2)  # equal jitter


def _err_body(exc: urllib.error.HTTPError) -> bytes:
    try:
        return exc.read()[:200]
    except Exception:  # noqa: BLE001 — body is best-effort logging only
        return b""


def _open_pr_nums(api: HfApi, repo: str) -> list[int]:
    discs = retry_on_429(
        lambda: list(api.get_repo_discussions(repo_id=repo, repo_type="dataset"))
    )
    return [
        d.num for d in discs
        if getattr(d, "is_pull_request", False) and getattr(d, "status", "") == "open"
    ]


def _has_verdict(api: HfApi, repo: str, pr_num: int) -> bool:
    d = retry_on_429(
        api.get_discussion_details, repo_id=repo, repo_type="dataset",
        discussion_num=pr_num,
    )
    for e in getattr(d, "events", []) or []:
        if getattr(e, "type", None) == "comment" and _VERDICT_MARKER in (getattr(e, "content", "") or ""):
            return True
    return False


def _dispatch_verify_workflow(repo: str, pr_num: int) -> bool:
    """Fire the verify-hf-pr GitHub workflow for one PR (workflow_dispatch).

    Mirrors the webhook's dispatch. Never raises. Returns ``True`` only if the
    dispatch request was accepted; ``False`` if no token is configured or the
    request failed (the PR is then left for a later sweep).

    Auth token is read from ``GH_TOKEN`` or ``GH_PAT``. In the GitHub Actions
    workflow this is the built-in ``GITHUB_TOKEN`` (with ``actions: write``),
    which *can* trigger a ``workflow_dispatch`` run in the same repo.
    """
    token = os.environ.get("GH_TOKEN") or os.environ.get("GH_PAT")
    if not token:
        logger.warning("no GH_TOKEN/GH_PAT; cannot dispatch verify-pr for %s#%d", repo, pr_num)
        return False
    target = os.environ.get("GH_VERIFY_WORKFLOW_REPO", _DEFAULT_TARGET)
    url = f"{_GH_API}/repos/{target}/actions/workflows/verify-hf-pr.yml/dispatches"
    data = json.dumps({
        "ref": "main",
        "inputs": {"repo": repo, "pr": str(pr_num), "branch": f"refs/pr/{pr_num}"},
    }).encode()
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "ssb-sweep",
        },
    )
    for attempt in range(_DISPATCH_ATTEMPTS):
        last = attempt == _DISPATCH_ATTEMPTS - 1
        try:
            urllib.request.urlopen(req, timeout=10)
            return True
        except urllib.error.HTTPError as exc:  # noqa: PERF203 — HTTPError is a URLError subclass; catch first
            if not _http_error_retryable(exc) or last:
                logger.warning("verify-pr dispatch failed for %s#%d: %s %s",
                               repo, pr_num, exc.code, _err_body(exc))
                return False
            delay = _backoff_delay(attempt, _retry_after(exc))
            logger.info("verify-pr dispatch %s#%d got %s (attempt %d/%d); retrying in %.1fs",
                        repo, pr_num, exc.code, attempt + 1, _DISPATCH_ATTEMPTS, delay)
            _sleep(delay)
        except urllib.error.URLError as exc:
            # Transient transport failure (DNS, reset, timeout) — retry.
            if last:
                logger.warning("verify-pr dispatch error for %s#%d: %s", repo, pr_num, exc)
                return False
            _sleep(_backoff_delay(attempt, None))
        except Exception as exc:  # noqa: BLE001
            logger.warning("verify-pr dispatch error for %s#%d: %s", repo, pr_num, exc)
            return False
    return False  # pragma: no cover — loop exits via return above


def run(
    datasets: list[str] | None = None,
    max_dispatch: int = 1,
    *,
    api: HfApi | None = None,
    dispatch=None,
    dry_run: bool = False,
) -> int:
    """Find verdict-less open PRs and re-dispatch verify-pr for up to
    ``max_dispatch`` of them. ``datasets=None`` sweeps every manifest dataset.
    ``api``/``dispatch`` are injectable for tests. Always returns 0.
    """
    api = api or HfApi()
    if dispatch is None:
        dispatch = _dispatch_verify_workflow
    if datasets is None:
        from ..manifest import all_dataset_ids, fetch_manifest
        datasets = all_dataset_ids(fetch_manifest())

    candidates: list[tuple[str, int]] = []
    for repo in datasets:
        try:
            prs = _open_pr_nums(api, repo)
        except Exception as exc:  # noqa: BLE001 — one bad repo must not abort the sweep
            logger.warning("sweep: listing PRs failed for %s: %s", repo, exc)
            continue
        for pr_num in prs:
            try:
                if not _has_verdict(api, repo, pr_num):
                    candidates.append((repo, pr_num))
            except Exception as exc:  # noqa: BLE001
                logger.warning("sweep: verdict check failed for %s#%d: %s", repo, pr_num, exc)

    candidates.sort()
    to_do = candidates if dry_run else candidates[:max_dispatch]
    dispatched = 0
    for repo, pr_num in to_do:
        if dry_run:
            logger.info("sweep: would dispatch verify-pr for %s#%d", repo, pr_num)
            dispatched += 1
        elif dispatch(repo, pr_num):
            logger.info("sweep: dispatched verify-pr for %s#%d", repo, pr_num)
            dispatched += 1
        else:
            logger.warning("sweep: dispatch skipped/failed for %s#%d", repo, pr_num)

    logger.info("sweep: %d verdict-less PR(s); %d %s, %d remaining%s",
                len(candidates), dispatched,
                "would dispatch" if dry_run else "dispatched",
                len(candidates) - dispatched, " (dry-run)" if dry_run else "")
    # Surface a missing-token misconfig as a non-zero exit so the CI run is red.
    if not dry_run and to_do and dispatched == 0:
        logger.error("sweep: had %d candidate(s) but dispatched none "
                     "(missing GH token?)", len(to_do))
        return 1
    return 0
