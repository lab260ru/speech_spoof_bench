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
import time
from pathlib import Path
from urllib.parse import quote

import requests
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

_HF_RESOLVE_RE = re.compile(
    r"^https://huggingface\.co/(?P<repo>[^/]+/[^/]+)/resolve/"
    r"(?P<sha>[0-9a-f]{7,40})/(?P<path>.+)$"
)
_DEFAULT_TIMEOUT = float(os.environ.get("SSB_HF_TIMEOUT", "20"))
_DEFAULT_ATTEMPTS = int(os.environ.get("SSB_HF_ATTEMPTS", "3"))
_DEFAULT_SLEEP = float(os.environ.get("SSB_HF_RETRY_SLEEP", "1"))


def parse_hf_resolve_url(url: str) -> tuple[str, str, str]:
    """Parse a commit-pinned HF resolve URL into (repo_id, commit_sha, path)."""
    m = _HF_RESOLVE_RE.match(url)
    if not m:
        raise ValueError(f"not a commit-pinned HF resolve URL: {url!r}")
    return m["repo"], m["sha"], m["path"]


def _auth_headers(token: str | None = None) -> dict[str, str]:
    token = token if token is not None else (os.environ.get("HF_TOKEN") or None)
    return {"authorization": f"Bearer {token}"} if token else {}


def _repo_api_prefix(repo_type: str | None) -> str:
    if repo_type in (None, "model"):
        return "models"
    if repo_type == "dataset":
        return "datasets"
    if repo_type == "space":
        return "spaces"
    raise ValueError(f"unsupported HF repo_type: {repo_type!r}")


def _retryable(exc: BaseException) -> bool:
    if isinstance(exc, HfHubHTTPError):
        status = getattr(getattr(exc, "response", None), "status_code", None)
        return status == 429 or (status is not None and status >= 500)
    return isinstance(exc, (requests.RequestException, TimeoutError, OSError))


def _with_retries(fn, *, attempts: int, sleep: float):
    last: BaseException | None = None
    for attempt in range(1, max(1, attempts) + 1):
        try:
            return fn()
        except BaseException as exc:
            if not _retryable(exc) or attempt >= max(1, attempts):
                raise
            last = exc
            if sleep > 0:
                time.sleep(sleep)
    assert last is not None
    raise last


def hub_download(
    *,
    attempts: int = _DEFAULT_ATTEMPTS,
    timeout: float = _DEFAULT_TIMEOUT,
    sleep: float = _DEFAULT_SLEEP,
    **kwargs,
) -> Path:
    """Bounded/retried wrapper around `hf_hub_download`."""
    kwargs.setdefault("etag_timeout", timeout)
    return Path(
        _with_retries(
            lambda: hf_hub_download(**kwargs),
            attempts=attempts,
            sleep=sleep,
        )
    )


def repo_sha(
    repo_id: str,
    *,
    repo_type: str | None = None,
    token: str | None = None,
    attempts: int = _DEFAULT_ATTEMPTS,
    timeout: float = _DEFAULT_TIMEOUT,
    sleep: float = _DEFAULT_SLEEP,
) -> str:
    """Fetch the current repo sha via a bounded direct Hub API request."""
    prefix = _repo_api_prefix(repo_type)
    url = f"https://huggingface.co/api/{prefix}/{repo_id}"

    def _request():
        response = requests.get(url, headers=_auth_headers(token), timeout=timeout)
        response.raise_for_status()
        return response

    data = _with_retries(_request, attempts=attempts, sleep=sleep).json()
    sha = data.get("sha")
    if not sha:
        raise RuntimeError(f"HF repo API returned no sha for {repo_id}")
    return str(sha)


def list_repo_files(
    repo_id: str,
    *,
    revision: str | None = None,
    repo_type: str | None = None,
    token: str | None = None,
    attempts: int = _DEFAULT_ATTEMPTS,
    timeout: float = _DEFAULT_TIMEOUT,
    sleep: float = _DEFAULT_SLEEP,
) -> list[str]:
    """List repo files with request timeouts instead of HfApi's unbounded pagination."""
    prefix = _repo_api_prefix(repo_type)
    rev = quote(revision or "main", safe="")
    url = f"https://huggingface.co/api/{prefix}/{repo_id}/tree/{rev}?recursive=true"
    headers = _auth_headers(token)
    files: list[str] = []

    while url:
        def _request():
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response

        response = _with_retries(_request, attempts=attempts, sleep=sleep)
        for item in response.json():
            if item.get("type") == "file":
                files.append(str(item["path"]))
        url = response.links.get("next", {}).get("url")

    return files


def download(url: str) -> tuple[Path, str]:
    """Download a commit-pinned HF resolve URL.

    Returns (local_path, sha256_hex). Honors $HF_TOKEN if set.
    """
    repo, sha, path = parse_hf_resolve_url(url)
    token = os.environ.get("HF_TOKEN") or None
    p = hub_download(
        repo_id=repo,
        filename=path,
        repo_type="model",
        revision=sha,
        token=token,
    )
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return p, h.hexdigest()
