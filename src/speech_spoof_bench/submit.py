"""Phase 7b — `submit` command implementation.

Public surface (used by cli.py and tests):
  - load_meta(path) -> dict
  - build_submission_payload(...) -> dict
  - upload_scores(...) -> (url, oid)
  - open_submission_pr(...) -> url
  - submit_one(...) -> url
  - submit(...) -> {dataset_spec: url}

Module-level seams (monkeypatched by tests):
  - _resolve_dataset_slug, _run_benchmark, _read_result_yaml
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
from importlib import resources
from io import BytesIO
from pathlib import Path
from typing import Any

import yaml
from huggingface_hub import CommitOperationAdd, HfApi
from jsonschema import ValidationError, validate

_META_SCHEMA_PACKAGE = "speech_spoof_bench.data"
_META_SCHEMA_FILENAME = "submission_meta.schema.json"

_LOG = logging.getLogger(__name__)


class MetaValidationError(ValueError):
    """Raised when a submission meta YAML fails schema validation."""


def _load_meta_schema() -> dict[str, Any]:
    with resources.files(_META_SCHEMA_PACKAGE).joinpath(_META_SCHEMA_FILENAME).open("r") as f:
        return json.load(f)


def load_meta(path: Path | str) -> dict[str, Any]:
    """Parse and validate a submission meta YAML.

    Raises:
      FileNotFoundError: path doesn't exist.
      MetaValidationError: YAML parses but fails the schema.
    """
    p = Path(path)
    text = p.read_text()  # raises FileNotFoundError as desired
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise MetaValidationError(f"{p}: not a YAML mapping")
    try:
        validate(instance=data, schema=_load_meta_schema())
    except ValidationError as exc:
        raise MetaValidationError(f"{p}: {exc.message}") from exc
    return data


def build_submission_payload(
    *,
    result_yaml: dict[str, Any],
    meta: dict[str, Any],
    scores_url: str,
    scores_sha256: str,
    hf_username: str,
    contact: str,
    submitted_at: str,
) -> dict[str, Any]:
    """Merge a result.yaml + meta into a fully-formed v4 submission dict.

    The `reproduction` block is left empty by design (§1.7) — the maintainer
    fills it via `reproduce --scoring` at merge time.
    """
    sys_meta = meta["system"]
    payload: dict[str, Any] = {
        "schema_version": 4,
        "system": {
            "name": sys_meta["name"],
            "slug": sys_meta["slug"],
            "description": sys_meta["description"],
            "code": sys_meta["code"],
            "checkpoint": sys_meta["checkpoint"],
            "paper": dict(sys_meta["paper"]),
        },
        "dataset": dict(result_yaml["dataset"]),
        "scores": dict(result_yaml["scores"]),
        "artifact": {
            "scores_url": scores_url,
            "scores_sha256": scores_sha256,
            "bench_version": result_yaml["artifact"]["bench_version"],
        },
        "reproduction": {},
        "submitter": {"hf_username": hf_username, "contact": contact},
        "submitted_at": submitted_at,
    }
    if "notes" in meta:
        payload["notes"] = meta["notes"]
    return payload


def upload_scores(
    *,
    api: HfApi,
    model_repo: str,
    dataset_canonical_id: str,
    local_path: Path | str,
) -> tuple[str, str]:
    """Upload scores.txt to the model repo's main branch.

    Returns (scores_url, commit_oid). The URL pins the returned commit oid.
    """
    p = Path(local_path)
    if not p.is_file():
        raise FileNotFoundError(f"scores file not found: {p}")
    path_in_repo = f".eval_results/{dataset_canonical_id}/scores.txt"
    info = api.upload_file(
        path_or_fileobj=str(p),
        path_in_repo=path_in_repo,
        repo_id=model_repo,
        repo_type="model",
        commit_message=f"upload scores for {dataset_canonical_id}",
    )
    oid = info.oid
    url = f"https://huggingface.co/{model_repo}/resolve/{oid}/{path_in_repo}"
    return url, oid


def open_submission_pr(
    *,
    api: HfApi,
    dataset_id: str,
    parent_commit: str,
    slug: str,
    yaml_text: str,
) -> str:
    """Open an HF PR on the dataset repo carrying submissions/<slug>.yaml.

    Returns the PR URL.
    """
    ops = [
        CommitOperationAdd(
            path_in_repo=f"submissions/{slug}.yaml",
            path_or_fileobj=BytesIO(yaml_text.encode("utf-8")),
        )
    ]
    info = api.create_commit(
        repo_id=dataset_id,
        repo_type="dataset",
        operations=ops,
        commit_message=f"submissions: add {slug}",
        create_pr=True,
        parent_commit=parent_commit,
    )
    url = getattr(info, "pr_url", None)
    if not url:
        raise RuntimeError("HF create_commit returned no PR url")
    return url


def _resolve_dataset_slug(spec: str, api: HfApi) -> tuple[str, str, str, str]:
    """Resolve `spec` to (canonical_id, slug, revision, split).

    Slug is the last path segment (matches DatasetSource.slug for HF specs).
    Revision is the current main-branch sha from HfApi.repo_info — we want the
    state the run scored against, even though `loader.resolve` returns None for
    HF specs today.
    """
    from .loader import resolve as _resolve

    source, _ = _resolve(spec, streaming=True)
    info = api.repo_info(repo_id=source.canonical_id, repo_type="dataset")
    return source.canonical_id, source.slug, info.sha, source.split


def _run_benchmark(
    *, model_module_spec: str, dataset_spec: str, output_dir: Path,
) -> None:
    """Import the model class and run the benchmark for a single dataset."""
    import importlib

    from .benchmark import Benchmark

    mod_name, cls_name = model_module_spec.split(":", 1)
    cls = getattr(importlib.import_module(mod_name), cls_name)
    model = cls()
    Benchmark.run(
        model,
        datasets=[dataset_spec],
        output_dir=str(output_dir),
        skip_existing=False,
    )


def _read_result_yaml(out_dir: Path) -> dict[str, Any] | None:
    p = out_dir / "result.yaml"
    if not p.is_file():
        return None
    return yaml.safe_load(p.read_text())


def submit_one(
    *,
    model_module_spec: str,
    dataset_spec: str,
    output_dir: Path,
    meta: dict[str, Any],
    model_repo: str,
    hf_username: str,
    contact: str,
    submitted_at: str,
    api: HfApi,
) -> str:
    """Run one (model, dataset) submission end-to-end. Returns the PR URL."""
    from . import submission as _sub

    canonical_id, slug, revision, _split = _resolve_dataset_slug(dataset_spec, api)
    out_dir = Path(output_dir) / slug

    existing = _read_result_yaml(out_dir)
    if existing is None or existing.get("dataset", {}).get("revision") != revision:
        _LOG.info("running benchmark for %s (revision %s)", canonical_id, revision)
        _run_benchmark(
            model_module_spec=model_module_spec,
            dataset_spec=dataset_spec,
            output_dir=Path(output_dir),
        )
        existing = _read_result_yaml(out_dir)
        if existing is None:
            raise RuntimeError(
                f"benchmark run produced no result.yaml under {out_dir}"
            )

    # Sanity-check: every metric from the result must be present.
    source_metrics = [
        k for k in existing.get("scores", {}) if k not in {"n_trials", "n_skipped"}
    ]
    if not source_metrics:
        raise RuntimeError(f"result.yaml at {out_dir} has no metric values")

    # Override revision in the result we pass to build_submission_payload so
    # the dataset block matches the freshly-resolved sha (loader.resolve
    # currently returns None for HF specs).
    existing = dict(existing)
    existing["dataset"] = dict(existing["dataset"])
    existing["dataset"]["revision"] = revision

    scores_path = out_dir / "scores.txt"
    scores_url, _commit_oid = upload_scores(
        api=api,
        model_repo=model_repo,
        dataset_canonical_id=canonical_id,
        local_path=scores_path,
    )

    payload = build_submission_payload(
        result_yaml=existing,
        meta=meta,
        scores_url=scores_url,
        scores_sha256=existing["artifact"]["scores_sha256"],
        hf_username=hf_username,
        contact=contact,
        submitted_at=submitted_at,
    )

    yaml_text = yaml.safe_dump(payload, sort_keys=False)
    _sub.parse_submission(yaml_text)  # raises if invalid

    return open_submission_pr(
        api=api,
        dataset_id=canonical_id,
        parent_commit=revision,
        slug=meta["system"]["slug"],
        yaml_text=yaml_text,
    )


def _expand_dataset_specs(specs: list[str]) -> list[str]:
    """Expand `--datasets all` against the arena manifest (core_set + extended)."""
    if specs == ["all"]:
        from . import manifest as _mf
        m = _mf.fetch_manifest()
        return [e["id"] for e in m.get("core_set", [])] + [
            e["id"] for e in m.get("extended", [])
        ]
    if "all" in specs:
        raise ValueError("'--datasets all' must be used alone, not mixed with explicit ids")
    return list(specs)


def submit(
    *,
    model_module_spec: str,
    dataset_specs: list[str],
    output_dir: Path,
    meta_path: Path,
    model_repo: str,
    hf_username: str,
    contact: str,
    continue_on_error: bool = False,
    api: HfApi | None = None,
) -> dict[str, str]:
    """Run `submit_one` for each dataset; return {dataset_spec: pr_url}."""
    meta = load_meta(meta_path)
    expanded = _expand_dataset_specs(dataset_specs)
    api = api or HfApi()
    submitted_at = _dt.date.today().isoformat()

    results: dict[str, str] = {}
    for spec in expanded:
        try:
            url = submit_one(
                model_module_spec=model_module_spec,
                dataset_spec=spec,
                output_dir=Path(output_dir),
                meta=meta,
                model_repo=model_repo,
                hf_username=hf_username,
                contact=contact,
                submitted_at=submitted_at,
                api=api,
            )
            _LOG.info("submitted %s → %s", spec, url)
            results[spec] = url
        except Exception as exc:
            if not continue_on_error:
                raise
            _LOG.error("submission failed for %s: %s", spec, exc)
            results[spec] = f"ERROR: {exc}"
    return results
