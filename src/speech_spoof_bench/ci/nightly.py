"""`speech-spoof-bench ci nightly-revalidate`.

Walks every merged submission across the manifest, runs the same checks as
`ci verify-pr` (schema + sha + EER), and opens/comments/closes GitHub issues
labelled `stale-submission`.
"""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass

from huggingface_hub import HfApi

from .. import reproduce, submission
from ..manifest import fetch_manifest
from ..submission import list_submission_files, fetch_submission
from . import green_store

logger = logging.getLogger(__name__)
LABEL = "stale-submission"


@dataclass(frozen=True)
class Failure:
    dataset_id: str
    slug: str
    reason: str


def _fetch_manifest():
    return fetch_manifest()


def _list_submission_files(dataset_id: str, **kw) -> list[str]:
    return list_submission_files(dataset_id, **kw)


def _check_submission_data(dataset_id: str, data: dict) -> Failure | None:
    """Run reproduce --scoring on an already-fetched submission dict."""
    import os
    import tempfile
    import yaml as _yaml
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        _yaml.safe_dump(data, fh)
        local = fh.name
    try:
        rc = reproduce.run_scoring(local, tolerance=1e-6)
    finally:
        try:
            os.unlink(local)
        except OSError:
            pass
    if rc != 0:
        return Failure(dataset_id, data["system"]["slug"],
                       "reproduce --scoring failed (see job log)")
    return None


def _check_submission(dataset_id: str, path: str) -> Failure | None:
    # Retained as a path-based convenience wrapper (collect_failures inlines this flow).
    try:
        data = fetch_submission(dataset_id, path)
    except submission.SubmissionValidationError as e:
        return Failure(dataset_id, path.rsplit("/", 1)[-1].removesuffix(".yaml"),
                       f"schema: {e}")
    return _check_submission_data(dataset_id, data)


def collect_failures(*, full: bool = False) -> list[Failure]:
    failures: list[Failure] = []
    store = green_store.load(green_store.DEFAULT_STORE_PATH)
    n_skipped = n_verified = 0
    m = _fetch_manifest()
    for entry in m.get("core_set", []) + m.get("extended", []):
        did = entry["id"]
        try:
            paths = _list_submission_files(did)
        except Exception as exc:  # noqa: BLE001
            failures.append(Failure(did, "<list>", f"list failed: {exc}"))
            continue
        for p in paths:
            try:
                data = fetch_submission(did, p)
            except submission.SubmissionValidationError as e:
                failures.append(Failure(
                    did, p.rsplit("/", 1)[-1].removesuffix(".yaml"), f"schema: {e}"))
                continue
            slug = data["system"]["slug"]
            sha = data["artifact"]["scores_sha256"]
            rev = data["dataset"]["revision"]
            if not full and green_store.is_green(store, did, slug, sha, rev):
                n_skipped += 1
                logger.info("nightly skip (unchanged/green): %s/%s", did, slug)
                continue
            n_verified += 1
            f = _check_submission_data(did, data)
            if f is not None:
                failures.append(f)
            else:
                green_store.record_green(store, did, slug, sha, rev)
    green_store.save(store, green_store.DEFAULT_STORE_PATH)
    logger.info("nightly: %d verified, %d skipped", n_verified, n_skipped)
    return failures


class _GhApi:
    """Thin wrapper around `gh` CLI for issues. Tests substitute a MagicMock."""

    def list_issues(self) -> list[dict]:
        out = subprocess.check_output([
            "gh", "issue", "list", "--label", LABEL, "--state", "open",
            "--json", "number,title,comments",
        ]).decode()
        import json
        rows = json.loads(out)
        for r in rows:
            r["last_comment_body"] = (r.get("comments") or [{}])[-1].get("body", "")
        return rows

    def create_issue(self, *, title: str, body: str) -> None:
        subprocess.run(["gh", "issue", "create", "--label", LABEL,
                        "--title", title, "--body", body], check=True)

    def add_comment(self, number: int, *, body: str, reason: str = "") -> None:
        subprocess.run(["gh", "issue", "comment", str(number), "--body", body], check=True)

    def close_issue(self, number: int) -> None:
        subprocess.run(["gh", "issue", "close", str(number)], check=True)


def _title_for(failure: Failure) -> str:
    return f"[{failure.dataset_id}] {failure.slug}"


def manage_issues(*, failures: list[Failure], api=None) -> None:
    api = api or _GhApi()
    open_issues = {i["title"]: i for i in api.list_issues()}
    failure_titles = {_title_for(f): f for f in failures}

    for title, failure in failure_titles.items():
        if title not in open_issues:
            api.create_issue(
                title=title,
                body=f"Nightly revalidation failed for `{failure.dataset_id}` / `{failure.slug}`.\n\n"
                     f"Reason: {failure.reason}",
            )
        else:
            existing = open_issues[title]
            new_body = f"Still failing: {failure.reason}"
            if existing.get("last_comment_body") != new_body:
                api.add_comment(existing["number"], body=new_body, reason=failure.reason)

    for title, issue in open_issues.items():
        if title not in failure_titles:
            api.close_issue(issue["number"])


def run(*, open_issues: bool, full: bool = False) -> int:
    failures = collect_failures(full=full)
    for f in failures:
        logger.warning("nightly failure: %s/%s — %s", f.dataset_id, f.slug, f.reason)
    if open_issues:
        try:
            manage_issues(failures=failures)
        except Exception as exc:  # noqa: BLE001
            logger.warning("issue management failed: %s", exc)
    return 0 if not failures else 1
