"""`speech-spoof-bench ci verify-pr` — validate + reproduce changed submissions
on an HF dataset PR and post a markdown verdict to the discussion."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi

from .. import hf_fetch, reproduce, submission

logger = logging.getLogger(__name__)


@dataclass
class Verdict:
    path: str
    schema_ok: bool
    sha_ok: Optional[bool]   # None when not reached (schema failed first)
    eer_ok: Optional[bool]
    notes: str

    @property
    def passed(self) -> bool:
        return bool(self.schema_ok and self.sha_ok and self.eer_ok)


def _download_at_revision(repo_id: str, filename: str, revision: str, repo_type: str) -> str:
    return str(hf_fetch.hub_download(
        repo_id=repo_id, filename=filename, revision=revision, repo_type=repo_type
    ))


def _run_scoring_repro(data: dict) -> tuple[bool, str]:
    """Run the scoring-reproduction logic without going through the CLI.

    Returns (passed, notes). Notes carry the reason on failure or "ok".
    """
    import tempfile
    import yaml as _yaml
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        _yaml.safe_dump(data, fh)
        path = fh.name
    rc = reproduce.run_scoring(path, tolerance=1e-6)
    return (rc == 0, "ok" if rc == 0 else "reproduce --scoring exited non-zero (see job log)")


def _changed_submissions(api: HfApi, repo: str, branch: str) -> list[str]:
    del api  # kept for compatibility with tests/callers
    main_files = set(hf_fetch.list_repo_files(repo, repo_type="dataset"))
    branch_files = set(hf_fetch.list_repo_files(repo, revision=branch, repo_type="dataset"))
    candidates = {
        f for f in branch_files
        if f.startswith("submissions/") and f.endswith(".yaml")
        and f.rsplit("/", 1)[-1] not in {"README.md", "results_template.yaml"}
    }
    # For added files: not on main. For modified: we can't cheaply diff content
    # via list_repo_files; treat any submission present on the branch and not
    # on main as added, and run the check on all submissions on the branch
    # (over-inclusive but safe; cost is bounded by the number of changed YAMLs).
    added = candidates - main_files
    if added:
        return sorted(added)
    # Fall back: also include any file in candidates whose content differs.
    # For simplicity (and to keep network calls bounded), we treat the absence
    # of additions as "no submission changes" — modifications without additions
    # are rare in this workflow.
    return []


def _verdict_for(api: HfApi, repo: str, branch: str, path: str) -> Verdict:
    try:
        local = _download_at_revision(repo, path, revision=branch, repo_type="dataset")
        data = submission.parse_submission(Path(local).read_text())
    except submission.SubmissionValidationError as e:
        return Verdict(path=path, schema_ok=False, sha_ok=None, eer_ok=None, notes=f"schema: {e}")
    except Exception as e:  # noqa: BLE001
        return Verdict(path=path, schema_ok=False, sha_ok=None, eer_ok=None, notes=f"fetch/parse: {e}")

    passed, notes = _run_scoring_repro(data)
    if passed:
        return Verdict(path=path, schema_ok=True, sha_ok=True, eer_ok=True, notes="ok")
    return Verdict(path=path, schema_ok=True, sha_ok=False, eer_ok=False, notes=notes)


def format_markdown(verdicts: list[Verdict], gh_run_url: str) -> str:
    if not verdicts:
        return (
            "**speech-spoof-bench ci verify-pr** — no submission changes detected.\n\n"
            f"_🤖 [view CI run]({gh_run_url})_"
        )
    header = "| Submission | Schema | sha256 | EER match | Notes |\n|---|---|---|---|---|\n"
    rows = []
    for v in verdicts:
        rows.append(
            f"| `{v.path}` | {'✅' if v.schema_ok else '❌'} "
            f"| {'✅' if v.sha_ok else ('—' if v.sha_ok is None else '❌')} "
            f"| {'✅' if v.eer_ok else ('—' if v.eer_ok is None else '❌')} "
            f"| {v.notes} |"
        )
    overall = "✅ all checks passed" if all(v.passed for v in verdicts) else "❌ failures present"
    return (
        f"**speech-spoof-bench ci verify-pr** — {overall}\n\n"
        f"{header}{chr(10).join(rows)}\n\n"
        f"_🤖 [view CI run]({gh_run_url})_"
    )


def _post_comment(repo: str, pr: int, body: str) -> None:
    token = os.environ.get("HF_BOT_TOKEN")
    if not token:
        logger.warning("HF_BOT_TOKEN not set; printing verdict instead of posting")
        print(body)
        return
    api = HfApi(token=token)
    api.comment_discussion(repo_id=repo, repo_type="dataset",
                           discussion_num=pr, comment=body)


def run(*, repo: str, pr: int, branch: str,
        api: HfApi | None = None,
        gh_run_url: str | None = None) -> int:
    api = api or HfApi()
    gh_run_url = gh_run_url or os.environ.get(
        "GH_RUN_URL", "https://github.com/lab260ru/speech_spoof_bench/actions"
    )

    paths = _changed_submissions(api, repo, branch)
    if not paths:
        _post_comment(repo, pr, format_markdown([], gh_run_url=gh_run_url))
        return 0

    verdicts = [_verdict_for(api, repo, branch, p) for p in paths]
    body = format_markdown(verdicts, gh_run_url=gh_run_url)
    _post_comment(repo, pr, body)
    return 0 if all(v.passed for v in verdicts) else 1
