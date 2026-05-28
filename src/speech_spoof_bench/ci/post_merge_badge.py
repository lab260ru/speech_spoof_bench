"""`speech-spoof-bench ci post-merge-badge` — post the paste-ready badge
snippet to the merged HF discussion."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import yaml
from huggingface_hub import HfApi, hf_hub_download

from .. import badge, submission

logger = logging.getLogger(__name__)

ARENA_URL = "https://huggingface.co/spaces/SpeechAntiSpoofingBenchmarks/SpeechAntiSpoofingArena"


def _download_at_revision(repo_id: str, filename: str, revision: str, repo_type: str) -> str:
    return hf_hub_download(repo_id=repo_id, filename=filename,
                           revision=revision, repo_type=repo_type)


def _changed_submissions(api: HfApi, repo: str, sha: str) -> list[str]:
    main_files = set(api.list_repo_files(repo_id=repo, repo_type="dataset"))
    sha_files = set(api.list_repo_files(repo_id=repo, revision=sha, repo_type="dataset"))
    candidates = {
        f for f in sha_files
        if f.startswith("submissions/") and f.endswith(".yaml")
        and f.rsplit("/", 1)[-1] not in {"README.md", "results_template.yaml"}
    }
    added = candidates - main_files
    return sorted(added)


def _primary_metric_at(api: HfApi, repo: str, revision: str) -> str:
    local = _download_at_revision(repo, "eval.yaml", revision=revision, repo_type="dataset")
    data = yaml.safe_load(Path(local).read_text())
    tasks = (data or {}).get("tasks") or []
    if not tasks:
        raise badge.BadgeError(f"{repo}@{revision}: eval.yaml has no tasks")
    metrics = tasks[0].get("metrics") or []
    if not metrics:
        raise badge.BadgeError(f"{repo}@{revision}: eval.yaml task[0] has no metrics")
    return str(metrics[0])


def _sentinel_for(sha: str, path: str) -> str:
    return f"<!-- ssb:badge --> sha={sha} path={path}"


def _already_posted(api: HfApi, repo: str, pr: int, sentinel: str) -> bool:
    try:
        details = api.get_discussion_details(
            repo_id=repo, repo_type="dataset", discussion_num=pr,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("get_discussion_details failed for %s#%d: %s", repo, pr, exc)
        return False
    for ev in getattr(details, "events", []) or []:
        body = getattr(ev, "content", "") or ""
        if sentinel in body:
            return True
    return False


def _post_comment(repo: str, pr: int, body: str) -> None:
    token = os.environ.get("HF_BOT_TOKEN")
    if not token:
        logger.warning("HF_BOT_TOKEN not set; printing comment instead of posting")
        print(body)
        return
    api = HfApi(token=token)
    api.comment_discussion(repo_id=repo, repo_type="dataset",
                           discussion_num=pr, comment=body)


def run(*, repo: str, pr: int, sha: str,
        api: HfApi | None = None,
        gh_run_url: str | None = None) -> int:
    api = api or HfApi()
    gh_run_url = gh_run_url or os.environ.get(
        "GH_RUN_URL", "https://github.com/lab260ru/speech_spoof_bench/actions"
    )

    paths = _changed_submissions(api, repo, sha)
    if not paths:
        logger.info("no new submissions in %s@%s; nothing to do", repo, sha)
        return 0

    errors = 0
    for path in paths:
        sentinel = _sentinel_for(sha, path)
        if _already_posted(api, repo, pr, sentinel):
            logger.info("badge comment already present for %s; skipping", path)
            continue
        try:
            local = _download_at_revision(repo, path, revision=sha, repo_type="dataset")
            data = submission.parse_submission(Path(local).read_text())
            dataset_id = data["dataset"]["id"]
            dataset_rev = data["dataset"]["revision"]
            primary = _primary_metric_at(api, dataset_id, dataset_rev)
            body = badge.build_paste_comment(
                data,
                arena_url=ARENA_URL,
                dataset_canonical_id=dataset_id,
                primary_metric=primary,
                submission_path=path,
                merge_sha=sha,
                gh_run_url=gh_run_url,
            )
            _post_comment(repo, pr, body)
        except Exception as exc:  # noqa: BLE001
            logger.error("badge generation failed for %s: %s", path, exc)
            errors += 1
    return 0 if errors == 0 else 1
