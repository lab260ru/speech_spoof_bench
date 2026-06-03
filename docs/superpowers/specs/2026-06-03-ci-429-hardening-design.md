# CI 429 hardening — design

**Date:** 2026-06-03
**Status:** approved, ready for implementation

## Problem

The `verify-hf-pr` and `post-merge-badge` GitHub Actions workflows intermittently
fail with `429 Too Many Requests` from the Hugging Face API. The failure reliably
appears when several dataset PRs are merged at the same time (e.g. one model
scored across several datasets, each merge dispatching its own CI run).

Observed traceback originates in `verify_pr._changed_submissions` →
`HfApi.list_repo_files` → `paginate` → `hf_raise_for_status`, but every HF call
in these workflows is exposed to the same throttle:

- `list_repo_files` (recursive tree listing) — `verify_pr.py`, `post_merge_badge.py`
- `hf_hub_download` — `verify_pr.py`, `post_merge_badge.py`, `hf_fetch.py`
- `list_repo_commits`, `get_discussion_details` — `post_merge_badge.py`
- `comment_discussion` — both
- `load_dataset(streaming=True)` — `reproduce.py` (labels stream)

## Root cause

The two workflows are `workflow_dispatch` jobs with **no concurrency control**.
Simultaneous merges spawn parallel runs, and their aggregate HF traffic trips
HF's throttle. HF rate-limiting is enforced at the account/IP/edge level, not
per dataset repo — so runs targeting *different* datasets still compete.

## Approach

Two independent, complementary levers. Both are implemented.

### Layer 1 — Serialize runs (root-cause fix)

Add the same `concurrency` block to **both** workflow files:

```yaml
concurrency:
  group: hf-ci
  cancel-in-progress: false
```

- A **fixed literal group name shared across both workflows** ⇒ GitHub runs at
  most one HF-touching job at a time, account-wide (global scope, matching the
  account-level throttle).
- `cancel-in-progress: false` ⇒ queued runs wait their turn rather than being
  cancelled; we never want to drop a pending verdict/badge.

**Known caveat — pending-queue depth.** GitHub keeps only 1 running + 1 pending
run per concurrency group; dispatching N>2 runs causes older *pending* runs to
be cancelled. Serialization therefore reduces but does not by itself guarantee
every run completes — which is the main reason Layer 2 is also required, and why
the existing `refs/pr/N` re-dispatch recovery path remains relevant for any run
that is dropped or fails.

### Layer 2 — Retry with backoff (residual resilience)

New helper `src/speech_spoof_bench/ci/_hf_retry.py`:

```python
def retry_on_429(fn, /, *args, _attempts=6, _base=2.0, _cap=60.0, _sleep=time.sleep, **kwargs):
    """Call fn(*args, **kwargs), retrying only on HF 429.

    - Catches HfHubHTTPError / httpx.HTTPStatusError with status_code == 429.
    - Any other status (or non-HTTP error) re-raises immediately.
    - Honors the Retry-After response header when present; otherwise
      exponential backoff with jitter (min(_cap, _base * 2**n) + jitter).
    - Re-raises the last 429 after _attempts exhausted (job fails with a clear log).
    - _sleep is injectable so unit tests run instantly.
    """
```

Wrap (do not rewrite) the HF calls at these sites:

- `verify_pr.py`: both `list_repo_files`, the `hf_hub_download`, `comment_discussion`
- `post_merge_badge.py`: `list_repo_files`, `list_repo_commits`,
  `hf_hub_download`, `get_discussion_details`, `comment_discussion`
- `hf_fetch.py`: the `hf_hub_download` in `download()`

**Out of scope:** the `load_dataset(streaming=True)` labels stream in
`reproduce.py`. It is an iterator inside the `datasets` library, not a single
wrappable call. With Layer-1 serialization it runs alone, so 429 there is
unlikely. Documented as a known residual rather than wrapped. (We deliberately
do not tune `datasets`' own retry config.)

## Testing

Unit test for `retry_on_429` (with an injected no-op sleep):

1. Callable raises 429 N times then returns a value ⇒ returns the value; assert
   it slept/retried the expected number of times.
2. Callable raises a non-429 `HfHubHTTPError` ⇒ re-raised immediately, no retry.
3. Callable always raises 429 ⇒ raises after `_attempts`, no infinite loop.
4. `Retry-After` header present ⇒ sleep honors it.

A fake 429 error object is constructed with the minimal `response.status_code`
/ `response.headers` surface the helper inspects.

## Docs

Add a section to `docs/developing/testing-and-pitfalls.md` covering:

- the 429 cause (concurrent merges, account-level throttle),
- the `hf-ci` global concurrency group and why `cancel-in-progress: false`,
- the `retry_on_429` helper and which calls it wraps,
- the pending-queue-depth drop risk and recovery via `refs/pr/N` re-dispatch.
