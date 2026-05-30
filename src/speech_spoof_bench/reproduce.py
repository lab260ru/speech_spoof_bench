"""Maintainer-side reproduction of submission scores (--scoring).

Workflow per §1.7 / spec §6 of phase-7a:
  1. Parse YAML.
  2. Fetch scores_url, verify sha.
  3. Stream labels from pinned dataset revision (no audio decode) — Task 8.
  4. Recompute every metric in the YAML — Task 9.
  5. Diff against claimed values — Task 9.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path

from datasets import load_dataset

logger = logging.getLogger(__name__)

from . import hf_fetch, submission
from .metrics import get_metric


def _parse_scores_txt(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        utt_id, score = line.split()
        out[utt_id] = float(score)
    return out


# Process-level memo of immutable labels, keyed by (dataset_id, revision).
_LABEL_CACHE: dict[tuple[str, str], dict[str, int]] = {}

_SHARD_RE = re.compile(r"^data/test-\d{5}-of-\d{5}\.parquet$")


def _shard_paths(api, dataset_id: str, revision: str) -> list[str]:
    files = api.list_repo_files(
        dataset_id,
        repo_type="dataset",
        revision=revision,
    )
    return sorted(path for path in files if _SHARD_RE.match(path))


def _info_fingerprint(info) -> tuple[str, int]:
    lfs = getattr(info, "lfs", None)
    sha = getattr(lfs, "sha256", None) if lfs is not None else None
    size = getattr(lfs, "size", None) if lfs is not None else None
    if sha is None:
        sha = getattr(info, "blob_id", "")
    if size is None:
        size = getattr(info, "size", -1)
    return str(sha), int(size)


def _shard_fingerprints(api, dataset_id: str, revision: str):
    paths = _shard_paths(api, dataset_id, revision)
    if not paths:
        return None
    infos = api.get_paths_info(
        dataset_id,
        paths,
        repo_type="dataset",
        revision=revision,
    )
    out = {info.path: _info_fingerprint(info) for info in infos}
    if set(out) != set(paths):
        return None
    return out


def _download_main_labels_if_shards_match(dataset_id: str, revision: str):
    """Use main's labels.parquet only when its test shards match revision.

    Some legacy submissions are pinned to dataset commits created before the
    labels fast-path artifact existed. If the current main branch still has
    byte-identical test shards, its labels file is equivalent to one generated
    at the pinned revision and avoids re-streaming multi-GB audio parquet.
    """
    if revision == "main":
        return None
    try:
        from huggingface_hub import HfApi, hf_hub_download

        api = HfApi()
        pinned = _shard_fingerprints(api, dataset_id, revision)
        current = _shard_fingerprints(api, dataset_id, "main")
        if pinned is None or pinned != current:
            return None
        local = hf_hub_download(
            repo_id=dataset_id,
            filename="data/labels.parquet",
            repo_type="dataset",
            revision="main",
        )
        return Path(local)
    except Exception as exc:
        logger.debug(
            "compatible main labels.parquet unavailable for %s@%s: %s",
            dataset_id, revision, exc,
        )
        return None


def _download_labels_file(dataset_id: str, revision: str):
    """Download ``data/labels.parquet`` from the pinned dataset revision.

    Returns the local file path, or ``None`` if the file is absent (older
    datasets) or any fetch error occurs — the caller then streams shards.
    Never raises: the labels file is a pure optimization.
    """
    try:
        from huggingface_hub import hf_hub_download
        local = hf_hub_download(
            repo_id=dataset_id,
            filename="data/labels.parquet",
            repo_type="dataset",
            revision=revision,
        )
        return Path(local)
    except Exception as exc:
        logger.debug("labels.parquet unavailable for %s@%s: %s", dataset_id, revision, exc)
        return _download_main_labels_if_shards_match(dataset_id, revision)


def _stream_labels_from_shards(dataset_id, split, revision, *, mapped):
    """Original behavior: stream (notes, label) from local or remote shards."""
    if mapped is not None:
        import glob
        shards = sorted(glob.glob(str(mapped / "data" / "test-*.parquet")))
        if not shards:
            raise FileNotFoundError(
                f"{mapped}/data/test-*.parquet not found for {dataset_id}"
            )
        ds = load_dataset(
            "parquet", data_files={"train": shards}, split="train",
            streaming=True, columns=["notes", "label"],
        )
    else:
        ds = load_dataset(
            dataset_id, split=split, streaming=True, revision=revision,
            columns=["notes", "label"],
        )
    labels: dict[str, int] = {}
    for row in ds:
        note = json.loads(row["notes"])
        labels[note["utterance_id"]] = int(row["label"])
    return labels


def _stream_labels(
    dataset_id: str, split: str, revision: str, *,
    force_remote: bool = False, force_shards: bool = False,
) -> dict[str, int]:
    """Return ``{utterance_id: int_label}`` for a pinned dataset revision.

    Resolution order (shards stay authoritative; the labels file and cache are
    pure optimizations and never cause a failure when absent):
      1. process cache (unless ``force_shards``)
      2. ``data/labels.parquet`` — local mapped copy, or one HF download
      3. stream ``data/test-*.parquet`` shards (today's behavior)

    ``force_remote`` ignores the local-dataset registry. ``force_shards``
    bypasses the labels file and the cache to verify against shards directly.
    Results are still cached so subsequent non-force calls benefit from the shard read.
    """
    from . import local_registry, labels as labels_mod

    cache_key = (dataset_id, revision)
    if not force_shards and cache_key in _LABEL_CACHE:
        return _LABEL_CACHE[cache_key]

    mapped = None if force_remote else local_registry.lookup(dataset_id)

    result: dict[str, int] | None = None
    if not force_shards:
        if mapped is not None:
            lf = mapped / "data" / labels_mod.LABELS_FILENAME
            if lf.is_file():
                result = labels_mod.load_labels_file(lf)
        else:
            lf = _download_labels_file(dataset_id, revision)
            if lf is not None:
                result = labels_mod.load_labels_file(lf)

    if result is None:
        result = _stream_labels_from_shards(
            dataset_id, split, revision, mapped=mapped
        )

    _LABEL_CACHE[cache_key] = result
    return result


def run_scoring(
    yaml_path: Path | str,
    *,
    tolerance: float = 1e-6,
    force_remote: bool = False,
    force_shards: bool = False,
    label_stream=None,
) -> int:
    """Run --scoring reproduction. Returns exit code (0 success, 1 fail).

    ``label_stream`` is injectable for tests. Defaults to _stream_labels.
    """
    if label_stream is None:
        def label_stream(did, split, rev):
            return _stream_labels(
                did, split, rev,
                force_remote=force_remote, force_shards=force_shards,
            )
    yaml_path = Path(yaml_path)
    try:
        data = submission.parse_submission(yaml_path.read_text())
    except Exception as e:
        print(f"FAIL: schema: {e}", file=sys.stderr)
        return 1

    url = data["artifact"]["scores_url"]
    claimed_sha = data["artifact"]["scores_sha256"]
    try:
        local, observed_sha = hf_fetch.download(url)
    except Exception as e:
        print(f"FAIL: fetch: {e}", file=sys.stderr)
        return 1
    if observed_sha != claimed_sha:
        print(
            f"FAIL: sha256 mismatch\n"
            f"  claimed:  {claimed_sha}\n"
            f"  observed: {observed_sha}",
            file=sys.stderr,
        )
        return 1

    scores = _parse_scores_txt(local)

    try:
        labels = label_stream(
            data["dataset"]["id"],
            data["dataset"]["split"],
            data["dataset"]["revision"],
        )
    except Exception as e:
        print(f"FAIL: dataset revision unreachable: {e}", file=sys.stderr)
        return 1

    scored_ids = set(scores)
    label_ids = set(labels)
    n_trials_claim = data["scores"]["n_trials"]
    n_skipped_claim = data["scores"]["n_skipped"]

    if scored_ids - label_ids:
        extra = sorted(scored_ids - label_ids)[:5]
        print(
            f"FAIL: coverage: scored {len(scored_ids - label_ids)} utterances not "
            f"in dataset (e.g. {extra})",
            file=sys.stderr,
        )
        return 1
    if len(scored_ids) + n_skipped_claim != n_trials_claim:
        print(
            f"FAIL: n_trials mismatch: "
            f"len(scores)={len(scored_ids)} + n_skipped={n_skipped_claim} "
            f"!= n_trials={n_trials_claim}",
            file=sys.stderr,
        )
        return 1
    if len(label_ids - scored_ids) > n_skipped_claim:
        print(
            f"FAIL: more skipped than claimed: "
            f"{len(label_ids - scored_ids)} unscored > n_skipped={n_skipped_claim}",
            file=sys.stderr,
        )
        return 1

    metric_keys = [
        k for k in data["scores"]
        if k not in {"n_trials", "n_skipped"}
    ]
    if not metric_keys:
        print("FAIL: no metrics in submission.scores to recompute",
              file=sys.stderr)
        return 1

    scores_subset = {k: scores[k] for k in scored_ids if k in label_ids}
    labels_subset = {k: labels[k] for k in scores_subset}

    diffs: list[tuple[str, float, float]] = []
    for mid in metric_keys:
        try:
            spec = get_metric(mid)
        except KeyError:
            print(
                f"FAIL: metric {mid!r} not registered in this version of "
                f"speech-spoof-bench",
                file=sys.stderr,
            )
            return 1
        result = spec.fn(scores_subset, labels_subset)
        claimed = float(data["scores"][mid])
        if abs(result.value - claimed) > tolerance:
            print(
                f"FAIL: metric {mid!r}: claimed {claimed!r} recomputed "
                f"{result.value!r} (Δ {result.value - claimed:.3e}, "
                f"tolerance {tolerance:.0e})",
                file=sys.stderr,
            )
            return 1
        diffs.append((mid, claimed, result.value))

    sha_short = claimed_sha[:4] + "…" + claimed_sha[-4:]
    rev = data["dataset"]["revision"]
    print(f"OK reproduced: {data['dataset']['id']} @ {rev}")
    print(f"  scores_sha256: matched ({sha_short})")
    for mid, claimed, recomputed in diffs:
        delta = recomputed - claimed
        print(
            f"  {mid}: claimed {claimed!r}  recomputed {recomputed!r}  "
            f"(Δ {delta:.1e})"
        )
    print(
        f"  n_trials:      {n_trials_claim} (skipped {n_skipped_claim})"
    )
    return 0
