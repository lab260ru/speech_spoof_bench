# Phase 4 — Manifest: Design

**Date**: 2026-05-21
**Scope**: ROADMAP Phase 4. Implements the manifest defined in PLAN §4.
**Status**: Approved (pending spec self-review + user review).

## Goal

Land the one-file `manifest.yaml` that the Arena and pip package read, and replace the Phase 2 manifest stubs in the pip package with a working reader (fetch + validate + tiny CLI surface).

Phase 4 DoD per ROADMAP: `speech-spoof-bench manifest` prints the manifest contents.

## Deliverables

Two independent pieces of work, in this order:

### A. `arena-manifest/manifest.yaml`

Working copy at `/home/kirill/speech-spoof-bench/arena-manifest/`. Pushed to
`huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/arena-manifest`.

Concrete launch content:

```yaml
ranking_version: v1
schema_version: 1

metrics_in_use:
  - eer_percent

tiers:
  - {name: gold,   min_coverage: 1.0}
  - {name: silver, min_coverage: 0.5}
  - {name: bronze, min_coverage: 0.0}

core_set:
  - id: SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA
    revision: 9b2040e8c57749dcd9a4f16ad61b4f47626b89ec

extended: []
```

Notes:
- `revision` uses the full 40-char sha (the spec example shows 8 chars; the schema accepts 7–40, and we prefer full shas for unambiguity).
- `extended: []` is explicit so consumers can iterate without a None check.

The repo's `README.md` is updated: drop the "Phase 0 — not yet written" line, add a short description of the file's shape and link to PLAN §4.

### B. Pip package: schema + reader + CLI

Three files touched, one new.

**New: `speech-spoof-bench/src/speech_spoof_bench/schema/manifest.schema.json`**

JSON Schema (draft-07 or 2020-12, matching what's already used in the package). Shape:

- Top-level object, `additionalProperties: false`.
- `ranking_version`: string, required.
- `schema_version`: integer, required, `const: 1` for now (bump when the schema changes).
- `metrics_in_use`: array of strings, required, `minItems: 1`.
- `tiers`: array, required, `minItems: 1`. Each item:
  - object, `additionalProperties: false`
  - `name`: string, required
  - `min_coverage`: number, required, `minimum: 0`, `maximum: 1`
- `core_set`: array, required, `minItems: 1`. Each item:
  - object, `additionalProperties: false`
  - `id`: string, required, pattern `^[^/]+/[^/]+$`
  - `revision`: string, required, pattern `^[a-f0-9]{7,40}$`
- `extended`: array, required (may be empty). Item shape identical to `core_set`.

**Replace: `speech-spoof-bench/src/speech_spoof_bench/manifest.py`**

```python
MANIFEST_REPO = "SpeechAntiSpoofingBenchmarks/arena-manifest"
MANIFEST_FILENAME = "manifest.yaml"

def fetch_manifest() -> dict:
    """Download manifest.yaml from HF, parse, validate, return dict."""

def load_manifest(path: str | Path) -> dict:
    """Load + validate a local manifest file (used in tests and offline dev)."""

def core_dataset_ids(manifest: dict) -> list[str]: ...
def all_dataset_ids(manifest: dict) -> list[str]: ...   # core + extended, in order
def revision_for(manifest: dict, dataset_id: str) -> str | None: ...
```

- `fetch_manifest` uses `huggingface_hub.hf_hub_download(repo_id=..., repo_type="dataset", filename=...)`. No auth.
- Validation: `jsonschema.validate` against the bundled schema. Errors bubble up unchanged.
- Network / HF errors bubble up unchanged.

**Replace stubs in `speech-spoof-bench/src/speech_spoof_bench/cli.py`**

Two subcommands stop being stubs:

- `speech-spoof-bench manifest` — fetches and prints the raw YAML file contents verbatim to stdout. (Raw bytes, not a re-dump, so users see the upstream file exactly.)
- `speech-spoof-bench list` — fetches, then prints one dataset id per line. Format: `[core] org/name` or `[ext]  org/name`. Core first, then extended, in manifest order.

Both honor the same exit codes as the rest of the CLI (0 on success, non-zero on fetch/parse/validation errors).

## Tests

`speech-spoof-bench/tests/test_manifest.py` (new):

- Fixtures: a valid `manifest.yaml` (same shape as launch content) + a directory of invalid variants (missing `tiers`, bad revision regex, extra top-level key, empty `core_set`, `min_coverage > 1`).
- `load_manifest` accepts the valid fixture, raises `jsonschema.ValidationError` on each invalid variant.
- `core_dataset_ids`, `all_dataset_ids`, `revision_for` against the fixture (including `revision_for` returning `None` for unknown id).
- CLI smoke test: monkeypatch `fetch_manifest` to return the fixture; run `manifest` and `list` subcommands via the CLI entrypoint; assert stdout shape (raw YAML for `manifest`; `[core]`/`[ext]` lines for `list`).

No live-HF test in the unit suite — manual `speech-spoof-bench manifest` against the real repo is the integration check at end of phase.

## Order of operations

1. Write `manifest.yaml` + update `README.md` in `arena-manifest/` working copy, commit, push to HF.
2. Add `manifest.schema.json`, implement `manifest.py`, wire CLI, add tests.
3. Run the test suite.
4. Run `speech-spoof-bench manifest` end-to-end against the real HF repo as Phase 4 DoD verification.

## Out of scope (deferred)

- Caching the downloaded manifest. `hf_hub_download` already caches in `~/.cache/huggingface`; nothing additional needed.
- `--manifest-path` / `ARENA_MANIFEST_REVISION` overrides. Not needed at launch; `load_manifest(path)` is exposed for tests/offline dev but not surfaced in the CLI.
- Manifest versioning beyond `schema_version: 1`. Bumping is a future PR.
- Anything that consumes the manifest beyond the two CLI commands (e.g. the Arena's ingest). That lands in Phase 5.
