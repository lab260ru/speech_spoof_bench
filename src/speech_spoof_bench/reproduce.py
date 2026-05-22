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


def _parse_scores_txt(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        utt_id, score = line.split()
        out[utt_id] = float(score)
    return out


def _stream_labels(dataset_id: str, split: str, revision: str) -> dict[str, int]:
    """Stream labels-only from the pinned dataset revision.

    Calls IterableDataset.select_columns(["notes", "label"]) BEFORE iteration
    so the audio column is projected away at the parquet reader. No audio
    bytes are fetched, no decode happens. This is the only correct invocation
    pattern — see spec §6.2.
    """
    ds = load_dataset(
        dataset_id, split=split, streaming=True, revision=revision
    )
    ds = ds.select_columns(["notes", "label"])
    labels: dict[str, int] = {}
    for row in ds:
        note = json.loads(row["notes"])
        labels[note["utterance_id"]] = int(row["label"])
    return labels


def run_scoring(
    yaml_path: Path | str,
    *,
    tolerance: float = 1e-6,
    label_stream=None,
) -> int:
    """Run --scoring reproduction. Returns exit code (0 success, 1 fail).

    ``label_stream`` is injectable for tests. Defaults to _stream_labels.
    """
    if label_stream is None:
        label_stream = _stream_labels
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

    # Task 9: metric recomputation.
    return 0
