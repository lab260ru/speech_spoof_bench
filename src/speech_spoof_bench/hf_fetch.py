"""Tiny wrapper around huggingface_hub for commit-pinned resolve URLs.

Centralises:
  - URL parsing (must be /resolve/<sha>/, not /resolve/main/)
  - HF_TOKEN env honoring
  - sha256 of the fetched file
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

from huggingface_hub import hf_hub_download

_HF_RESOLVE_RE = re.compile(
    r"^https://huggingface\.co/(?P<repo>[^/]+/[^/]+)/resolve/"
    r"(?P<sha>[0-9a-f]{7,40})/(?P<path>.+)$"
)


def parse_hf_resolve_url(url: str) -> tuple[str, str, str]:
    """Parse a commit-pinned HF resolve URL into (repo_id, commit_sha, path)."""
    m = _HF_RESOLVE_RE.match(url)
    if not m:
        raise ValueError(f"not a commit-pinned HF resolve URL: {url!r}")
    return m["repo"], m["sha"], m["path"]


def download(url: str) -> tuple[Path, str]:
    """Download a commit-pinned HF resolve URL.

    Returns (local_path, sha256_hex). Honors $HF_TOKEN if set.
    """
    repo, sha, path = parse_hf_resolve_url(url)
    token = os.environ.get("HF_TOKEN") or None
    local = hf_hub_download(
        repo_id=repo,
        filename=path,
        repo_type="model",
        revision=sha,
        token=token,
    )
    p = Path(local)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return p, h.hexdigest()
