"""End-to-end: reproduce --scoring against the live on-HF random-baseline.

NOT marked `network` — required HF reachability is intentional for this suite.
"""

from __future__ import annotations

import os
from pathlib import Path

from speech_spoof_bench import hf_fetch, reproduce


def _hf_cache_size(root: Path, substring: str) -> int:
    total = 0
    if not root.is_dir():
        return total
    for p in root.rglob("*"):
        if substring in str(p) and p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                continue
    return total


def test_random_baseline_real(capsys):
    # Pull the live submission YAML to a temp file.
    local = hf_fetch.hub_download(
        repo_id="SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA",
        filename="submissions/random-baseline.yaml",
        repo_type="dataset",
    )
    yaml_path = Path(local)

    cache_root = Path(
        os.environ.get("HF_DATASETS_CACHE")
        or Path.home() / ".cache" / "huggingface" / "datasets"
    )
    before = _hf_cache_size(cache_root, "ASVspoof2019_LA")

    rc = reproduce.run_scoring(yaml_path, tolerance=1e-6)
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "OK reproduced" in out

    after = _hf_cache_size(cache_root, "ASVspoof2019_LA")
    growth_mb = (after - before) / (1024 * 1024)
    assert growth_mb < 50, (
        f"HF cache grew by {growth_mb:.1f} MB — audio shards may have been "
        f"downloaded. The select_columns invariant is broken."
    )
