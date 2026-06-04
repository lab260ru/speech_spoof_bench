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
import urllib.error
import urllib.request

from huggingface_hub import HfApi

from ._hf_retry import retry_on_429

logger = logging.getLogger(__name__)

# Any verify-pr comment (a real verdict or "no submission changes") carries this.
_VERDICT_MARKER = "speech-spoof-bench ci verify-pr"
_GH_API = "https://api.github.com"
_DEFAULT_TARGET = "lab260ru/speech_spoof_bench"


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


def _dispatch_verify_workflow(repo: str, pr_num: int) -> None:
    """Fire the verify-hf-pr GitHub workflow for one PR (workflow_dispatch).

    Mirrors the webhook's dispatch. Never raises: a failed dispatch just means
    the PR is picked up by the next scheduled sweep.
    """
    token = os.environ.get("GH_PAT")
    if not token:
        logger.warning("GH_PAT not set; cannot dispatch verify-pr for %s#%d", repo, pr_num)
        return
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
    try:
        urllib.request.urlopen(req, timeout=10)
    except urllib.error.HTTPError as exc:  # noqa: PERF203
        logger.warning("verify-pr dispatch failed for %s#%d: %s %s",
                       repo, pr_num, exc.code, exc.read()[:200])
    except Exception as exc:  # noqa: BLE001
        logger.warning("verify-pr dispatch error for %s#%d: %s", repo, pr_num, exc)


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
    for repo, pr_num in to_do:
        if dry_run:
            logger.info("sweep: would dispatch verify-pr for %s#%d", repo, pr_num)
        else:
            dispatch(repo, pr_num)
            logger.info("sweep: dispatched verify-pr for %s#%d", repo, pr_num)

    n_done = 0 if dry_run else len(to_do)
    logger.info("sweep: %d verdict-less PR(s); %d dispatched, %d remaining%s",
                len(candidates), n_done, len(candidates) - n_done,
                " (dry-run)" if dry_run else "")
    return 0
