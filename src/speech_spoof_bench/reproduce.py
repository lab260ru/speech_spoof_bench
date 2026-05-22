"""Maintainer-side reproduction of submission scores (--scoring).

Workflow per §1.7 / spec §6 of phase-7a:
  1. Parse YAML.
  2. Fetch scores_url, verify sha.
  3. Stream labels from pinned dataset revision (no audio decode) — Task 8.
  4. Recompute every metric in the YAML — Task 9.
  5. Diff against claimed values — Task 9.
"""

from __future__ import annotations

import sys
from pathlib import Path

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


def run_scoring(
    yaml_path: Path | str,
    *,
    tolerance: float = 1e-6,
) -> int:
    """Run --scoring reproduction. Returns exit code (0 success, 1 fail)."""
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

    scores = _parse_scores_txt(local)  # noqa: F841 — Tasks 8–9 consume this

    # Tasks 8–9: label stream, coverage, metric recompute.
    return 0
