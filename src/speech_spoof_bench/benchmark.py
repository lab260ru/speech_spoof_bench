"""Orchestrator: load → run → score → write result.yaml."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from . import __version__ as _BENCH_VERSION
from .cache import purge_hf_cache
from .loader import resolve
from .metrics import get_metric
from .model import AntiSpoofingModel
from .runner import run_dataset

_LOG = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    dataset_slug: str
    canonical_id: str
    metrics: dict[str, float] = field(default_factory=dict)
    metric_extras: dict[str, dict[str, Any]] = field(default_factory=dict)
    n_trials: int = 0
    n_skipped: int = 0
    scores_path: Path | None = None
    result_yaml_path: Path | None = None


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_result_yaml(
    *,
    out_path: Path,
    model: AntiSpoofingModel,
    source,
    metrics: dict[str, float],
    n_trials: int,
    n_skipped: int,
    scores_sha256: str,
) -> None:
    payload: dict[str, Any] = {
        "schema_version": 4,
        "system": {
            "name": getattr(model, "name", "unknown"),
            "slug": None,
            "description": None,
            "code": None,
            "checkpoint": None,
            "paper": None,
        },
        "dataset": {
            "id": source.canonical_id,
            "revision": source.revision,
            "split": source.split,
        },
        "scores": {
            **{k: float(v) for k, v in metrics.items()},
            "n_trials": int(n_trials),
            "n_skipped": int(n_skipped),
        },
        "artifact": {
            "scores_url": None,
            "scores_sha256": scores_sha256,
            "bench_version": f"speech-spoof-bench=={_BENCH_VERSION}",
        },
        "reproduction": {},
        "submitter": {},
        "submitted_at": None,
        "notes": None,
    }
    out_path.write_text(yaml.safe_dump(payload, sort_keys=False))


class Benchmark:
    """Static entry point. ``Benchmark.run(model, datasets=...)``."""

    @staticmethod
    def run(
        model: AntiSpoofingModel,
        datasets: list[str] | str = "all",
        output_dir: str | Path = "./results",
        *,
        streaming: bool = True,
        cleanup: bool = True,
        skip_existing: bool = True,
        force_remote: bool = False,
    ) -> dict[str, BenchmarkResult]:
        if isinstance(datasets, str) and datasets == "all":
            raise NotImplementedError(
                "datasets='all' requires the arena manifest; lands in phase 4"
            )
        if isinstance(datasets, str):
            datasets = [datasets]
        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)

        results: dict[str, BenchmarkResult] = {}

        model.load()
        try:
            for spec in datasets:
                source, ds = resolve(spec, streaming=streaming, force_remote=force_remote)
                out = output_root / source.slug
                result_yaml = out / "result.yaml"

                if skip_existing and result_yaml.exists():
                    parsed = yaml.safe_load(result_yaml.read_text()) or {}
                    if parsed.get("dataset", {}).get("revision") == source.revision:
                        _LOG.info("skipping %s (result.yaml present)", source.slug)
                        existing_scores = parsed.get("scores", {}) or {}
                        results[source.slug] = BenchmarkResult(
                            dataset_slug=source.slug,
                            canonical_id=source.canonical_id,
                            metrics={
                                k: float(v)
                                for k, v in existing_scores.items()
                                if k not in {"n_trials", "n_skipped"}
                                and isinstance(v, (int, float))
                            },
                            n_trials=int(existing_scores.get("n_trials", 0)),
                            n_skipped=int(existing_scores.get("n_skipped", 0)),
                            scores_path=out / "scores.txt",
                            result_yaml_path=result_yaml,
                        )
                        continue

                run_res = run_dataset(model, source, ds, out)

                # Compute all metrics declared by the dataset's eval.yaml.
                scores_map = _load_scores_txt(run_res.scores_path)
                metric_values: dict[str, float] = {}
                metric_extras: dict[str, dict[str, Any]] = {}
                for mid in source.metrics:
                    spec_m = get_metric(mid)
                    mr = spec_m.fn(scores_map, run_res.labels)
                    metric_values[mid] = mr.value
                    metric_extras[mid] = dict(mr.extras)

                scores_sha256 = _sha256_of_file(run_res.scores_path)
                _write_result_yaml(
                    out_path=result_yaml,
                    model=model,
                    source=source,
                    metrics=metric_values,
                    n_trials=run_res.n_total,
                    n_skipped=run_res.n_skipped,
                    scores_sha256=scores_sha256,
                )

                results[source.slug] = BenchmarkResult(
                    dataset_slug=source.slug,
                    canonical_id=source.canonical_id,
                    metrics=metric_values,
                    metric_extras=metric_extras,
                    n_trials=run_res.n_total,
                    n_skipped=run_res.n_skipped,
                    scores_path=run_res.scores_path,
                    result_yaml_path=result_yaml,
                )

                if cleanup and not source.is_local:
                    purge_hf_cache(source.canonical_id)
        finally:
            model.unload()

        return results


def _load_scores_txt(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        utt_id, score = line.split()
        out[utt_id] = float(score)
    return out
