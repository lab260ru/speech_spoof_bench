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
import sys
from pathlib import Path

from datasets import load_dataset

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


def _stream_labels(
    dataset_id: str, split: str, revision: str, *, force_remote: bool = False,
) -> dict[str, int]:
    """Stream labels-only from the pinned dataset revision (or local copy).

    If `dataset_id` is in the local-dataset registry and `force_remote` is
    False, reads the labels from local parquet shards. Otherwise streams
    from HF at the pinned revision.

    Passes ``columns=["notes", "label"]`` to ``load_dataset`` so the parquet
    builder's ``ParquetConfig.columns`` is set. This propagates to
    ``ParquetFileFormat.to_batches(columns=...)``, which IS a true PyArrow
    column projection — only those column chunks are read from each row
    group. Audio column bytes are not transferred.

    Critical: passing columns post-construction via ``ds.select_columns(...)``
    is a CPU/memory projection only; the parquet read still pulls every
    column's bytes. Only the load-time form achieves network-level pushdown.
    """
    from . import local_registry

    mapped = None if force_remote else local_registry.lookup(dataset_id)
    if mapped is not None:
        import glob
        shards = sorted(glob.glob(str(mapped / "data" / "test-*.parquet")))
        if not shards:
            raise FileNotFoundError(
                f"{mapped}/data/test-*.parquet not found for {dataset_id}"
            )
        ds = load_dataset(
            "parquet",
            data_files={"train": shards},
            split="train",
            streaming=True,
            columns=["notes", "label"],
        )
    else:
        ds = load_dataset(
            dataset_id,
            split=split,
            streaming=True,
            revision=revision,
            columns=["notes", "label"],
        )

    labels: dict[str, int] = {}
    for row in ds:
        note = json.loads(row["notes"])
        labels[note["utterance_id"]] = int(row["label"])
    return labels


def run_scoring(
    yaml_path: Path | str,
    *,
    tolerance: float = 1e-6,
    force_remote: bool = False,
    label_stream=None,
) -> int:
    """Run --scoring reproduction. Returns exit code (0 success, 1 fail).

    ``label_stream`` is injectable for tests. Defaults to _stream_labels.
    """
    if label_stream is None:
        def label_stream(did, split, rev):
            return _stream_labels(did, split, rev, force_remote=force_remote)
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
