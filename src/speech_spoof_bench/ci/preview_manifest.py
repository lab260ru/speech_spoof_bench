"""Preview an arena-manifest candidate before merging it."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from huggingface_hub import HfApi

from .. import hf_fetch, manifest, submission

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreviewWarning:
    dataset_id: str
    path: str
    reason: str


@dataclass(frozen=True)
class PreviewResult:
    dataset_count: int
    rows: int
    warnings: list[PreviewWarning] = field(default_factory=list)
    added_datasets: list[str] = field(default_factory=list)
    removed_datasets: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.warnings


def _dataset_ids(data: dict) -> list[str]:
    return [entry["id"] for entry in data.get("core_set", []) + data.get("extended", [])]


def _dataset_entries(data: dict) -> list[dict]:
    return list(data.get("core_set", []) + data.get("extended", []))


def preview(candidate_manifest: dict, *, base_manifest: dict | None = None) -> PreviewResult:
    """Build an Arena-like manifest preview summary.

    This mirrors the Arena ingest acceptance rule: only submissions with a
    reproduction block are counted as displayable rows; invalid or unverified
    submissions become warnings.
    """
    entries = _dataset_entries(candidate_manifest)
    dataset_ids = [entry["id"] for entry in entries]
    base_ids = _dataset_ids(base_manifest) if base_manifest is not None else []
    rows = 0
    warnings: list[PreviewWarning] = []

    for entry in entries:
        dataset_id = entry["id"]
        revision = entry["revision"]
        revision_failed = False
        try:
            files_at_revision = hf_fetch.list_repo_files(
                dataset_id,
                repo_type="dataset",
                revision=revision,
            )
            if "eval.yaml" not in files_at_revision:
                warnings.append(
                    PreviewWarning(
                        dataset_id,
                        f"<revision:{revision}>",
                        "eval.yaml missing at manifest-pinned revision",
                    )
                )
        except Exception as exc:  # noqa: BLE001
            revision_failed = True
            warnings.append(PreviewWarning(dataset_id, f"<revision:{revision}>", str(exc)))

        try:
            paths = submission.list_submission_files(dataset_id)
        except Exception as exc:  # noqa: BLE001
            warnings.append(PreviewWarning(dataset_id, "<list>", str(exc)))
            continue
        if not paths and not revision_failed:
            warnings.append(
                PreviewWarning(
                    dataset_id,
                    "<submissions>",
                    "no submission YAMLs found; Arena will have no display rows",
                )
            )
            continue
        for path in paths:
            try:
                data = submission.fetch_submission(dataset_id, path)
            except submission.SubmissionValidationError as exc:
                warnings.append(PreviewWarning(dataset_id, path, f"schema: {exc}"))
                continue
            except Exception as exc:  # noqa: BLE001
                warnings.append(PreviewWarning(dataset_id, path, f"{type(exc).__name__}: {exc}"))
                continue
            repro = data.get("reproduction") or {}
            if not repro.get("match"):
                warnings.append(
                    PreviewWarning(
                        dataset_id,
                        path,
                        "missing reproduction block - submission is unverified, skipped",
                    )
                )
                continue
            rows += 1

    return PreviewResult(
        dataset_count=len(dataset_ids),
        rows=rows,
        warnings=warnings,
        added_datasets=sorted(set(dataset_ids) - set(base_ids)),
        removed_datasets=sorted(set(base_ids) - set(dataset_ids)),
    )


def _md_cell(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def format_markdown(result: PreviewResult, *, gh_run_url: str | None = None) -> str:
    lines = [
        "**speech-spoof-bench ci arena-manifest preview**",
        "",
        f"- Datasets: {result.dataset_count}",
        f"- Rows: {result.rows}",
        f"- Warnings: {len(result.warnings)}",
    ]
    if result.added_datasets:
        lines.append(f"- Added datasets: {', '.join(result.added_datasets)}")
    if result.removed_datasets:
        lines.append(f"- Removed datasets: {', '.join(result.removed_datasets)}")
    if result.warnings:
        lines.extend(["", "| Dataset | Path | Reason |", "|---|---|---|"])
        for warning in result.warnings:
            lines.append(
                f"| {_md_cell(warning.dataset_id)} | {_md_cell(warning.path)} "
                f"| {_md_cell(warning.reason)} |"
            )
    if gh_run_url:
        lines.extend(["", f"_[view CI run]({gh_run_url})_"])
    return "\n".join(lines)


def _post_comment(repo: str, pr: int, body: str) -> None:
    token = os.environ.get("HF_BOT_TOKEN")
    if not token:
        logger.warning("HF_BOT_TOKEN not set; printing manifest preview instead of posting")
        print(body)
        return
    api = HfApi(token=token)
    api.comment_discussion(repo_id=repo, repo_type="dataset", discussion_num=pr, comment=body)


def _load_candidate(
    *,
    manifest_path: Path | None,
    repo: str | None,
    branch: str | None,
) -> dict:
    if manifest_path is not None:
        return manifest.load_manifest(manifest_path)
    if repo is None or branch is None:
        raise ValueError("provide either --manifest or both --repo and --branch")
    local = hf_fetch.hub_download(
        repo_id=repo,
        filename=manifest.MANIFEST_FILENAME,
        repo_type="dataset",
        revision=branch,
    )
    return manifest.load_manifest(local)


def _load_base(repo: str | None) -> dict | None:
    if repo is None:
        return None
    local = hf_fetch.hub_download(
        repo_id=repo,
        filename=manifest.MANIFEST_FILENAME,
        repo_type="dataset",
        revision="main",
    )
    return manifest.load_manifest(local)


def run(
    *,
    manifest_path: Path | None = None,
    repo: str | None = None,
    pr: int | None = None,
    branch: str | None = None,
    gh_run_url: str | None = None,
) -> int:
    candidate = _load_candidate(manifest_path=manifest_path, repo=repo, branch=branch)
    result = preview(candidate, base_manifest=_load_base(repo))
    gh_run_url = gh_run_url or os.environ.get("GH_RUN_URL")
    body = format_markdown(result, gh_run_url=gh_run_url)
    if repo and pr is not None:
        _post_comment(repo, pr, body)
    else:
        print(body)
    return 0 if result.ok else 1
