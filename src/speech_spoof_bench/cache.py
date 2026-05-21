"""HF cache purger. No-op for repos not present in the cache."""

from __future__ import annotations

import logging

from huggingface_hub import scan_cache_dir

_LOG = logging.getLogger(__name__)


def purge_hf_cache(repo_id: str) -> None:
    """Remove all cached revisions for ``repo_id`` (datasets only).

    Silent no-op if the repo isn't in the cache.
    """
    info = scan_cache_dir()
    revisions_to_delete: list[str] = []
    for repo in info.repos:
        if repo.repo_id == repo_id and repo.repo_type == "dataset":
            revisions_to_delete.extend(rev.commit_hash for rev in repo.revisions)
    if not revisions_to_delete:
        _LOG.debug("no cached revisions for %s; nothing to purge", repo_id)
        return
    delete_strategy = info.delete_revisions(*revisions_to_delete)
    _LOG.info(
        "purging %d revisions for %s (~%s)",
        len(revisions_to_delete),
        repo_id,
        delete_strategy.expected_freed_size_str,
    )
    delete_strategy.execute()
