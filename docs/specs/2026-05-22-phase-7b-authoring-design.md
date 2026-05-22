# Phase 7b — Authoring (`submit`, `scaffold-dataset`)

**Date**: 2026-05-22
**Scope**: Roadmap Phase 7b (`docs/roadmap/ROADMAP.md` §Phase 7b)
**Spec reference**: `docs/roadmap/PLAN.md` §1.1, §1.5, §1.6, §2.5, §2.4 step 7

## Goal

Add two CLI commands to `speech-spoof-bench` so humans no longer hand-author
the artifacts that Phase 3 produced manually:

- **`submit`** — automates the end-to-end submitter flow: run model →
  upload scores to a model repo → build a v4 submission YAML → open an HF
  PR on the dataset repo.
- **`scaffold-dataset`** — generates the §1.1 dataset-repo skeleton
  (directories + stub files) so adding a new dataset starts from a valid
  layout, not a copy-and-edit of LA.

Verification is a unit-test suite with `HfApi` mocked plus one manual
end-to-end smoke test against the live
`SpeechAntiSpoofingBenchmarks/random-baseline-asas` model repo and
`SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA` dataset repo.

## Non-goals

- No `reproduce --inference` (Phase 8+).
- No arXiv auto-fetch in `scaffold-dataset` (deferred).
- No idempotency on the dataset PR — re-running `submit` opens another PR.
  Acceptable while volume is low.
- No upfront duplicate-slug check on the dataset repo; the PR commit
  simply overwrites `submissions/<slug>.yaml` and the maintainer resolves
  any collision at merge time.
- No persistent submitter config (no `~/.config/...` file). Identity is
  always passed via CLI flags.
- No CI/webhook wiring. Phase 8.

## Decisions

| Decision | Choice | Rationale |
|---|---|---|
| `submit` execution shape | Hybrid: reuse existing `result.yaml` if its `dataset.revision` matches; otherwise call `Benchmark.run` for that one dataset. | Cheap re-runs, one command from zero in the cold case. |
| Dataset write target | Always open an HF PR via `HfApi.create_commit(create_pr=True)`. | Matches the eventual submitter flow even when we are the maintainer; cleaner audit trail; no `--direct` flag needed. |
| Model-repo existence | Required up front. `submit` does not call `create_repo`. | One less permission to manage; clear failure mode for new submitters. |
| Submitter identity | CLI flags only: `--hf-username`, `--contact`. | Explicit, no hidden state, no implicit dependency on `huggingface-cli whoami` or git config. |
| Submission metadata | `--submission-meta meta.yaml` file (reusable across datasets). | One model → many datasets is the realistic case; co-locating in the class would force boilerplate on every model. |
| `reproduction:` block | Always empty in `submit`'s output. | §1.7 — only the maintainer fills it, via `reproduce --scoring` at merge time. |
| Multi-dataset support | `--datasets` repeatable; `--datasets all` reads the manifest. | Matches `run`'s flag shape. Each dataset is its own repo, so each gets its own PR. |
| Scores upload | Direct commit to `main` of the model repo via `HfApi.upload_file`; capture returned commit OID. | Submitter owns the model repo; PR ceremony there is unnecessary. |
| Scores upload idempotency | Always upload. HF deduplicates identical content; we use whatever commit sha the API returns. | Simpler than fetching latest commit and re-verifying sha. |
| `scaffold-dataset` scope | Skeleton + stub files only. No network. | YAGNI; arXiv fetch can be added later without changing the command shape. |
| Duplicate-slug handling on dataset repo | No upfront check; the PR overwrites the file; maintainer resolves at merge. | Less code in `submit`; matches the "always open a PR" decision. |
| Refusal on non-empty scaffold target | Default refuses; `--force` bypasses. | Standard scaffold ergonomics. |

## Architecture

Two HF repos are involved per `submit` call:

```
<owner>/<model-repo>                                  ← scores.txt (submitter owns; direct commit)
└── .eval_results/
    └── <dataset-canonical-id>/
        └── scores.txt

SpeechAntiSpoofingBenchmarks/<dataset>                ← submission YAML (org owns; PR-only)
└── submissions/
    └── <slug>.yaml
```

The dataset YAML references the scores file by **pinned URL** (`/resolve/<commit-oid>/...`) plus sha256. `submit`'s job is to populate those two fields correctly and produce a schema-valid v4 YAML.

### Module layout

```
src/speech_spoof_bench/
├── submit.py                               # new — submit core
├── scaffold.py                             # new — scaffold-dataset core
├── data/                                   # new — package data
│   ├── submission_meta.schema.json
│   └── dataset_skeleton/
│       ├── README.md
│       ├── LICENSE.txt
│       ├── eval.yaml
│       ├── build_parquet.py
│       └── submissions/
│           ├── README.md
│           └── results_template.yaml
└── cli.py                                  # extended — two new subcommands
```

`data/dataset_skeleton/submissions/README.md` and
`data/dataset_skeleton/submissions/results_template.yaml` are the canonical
versions that all dataset repos must carry verbatim; they're shipped inside
the pip package and copied by `scaffold-dataset`. (Existing copies in LA
remain authoritative as the source for this template — Phase 7b copies
them into the package as part of implementation.)

### `submit.py` function decomposition

Pure-functional pieces so HF calls are mockable in tests:

```python
def load_meta(path: Path) -> dict: ...
    # YAML parse + jsonschema validate against submission_meta.schema.json.

def build_submission_payload(
    *,
    result_yaml: dict,
    meta: dict,
    scores_url: str,
    scores_sha256: str,
    hf_username: str,
    contact: str,
    submitted_at: str,            # ISO date
) -> dict: ...
    # Merge into a full v4 submission dict; reproduction is {}.

def upload_scores(
    *, api: HfApi, model_repo: str,
    dataset_canonical_id: str, local_path: Path,
) -> tuple[str, str]: ...
    # Returns (scores_url, commit_oid).

def open_submission_pr(
    *, api: HfApi, dataset_id: str, parent_commit: str,
    slug: str, yaml_text: str,
) -> str: ...
    # Returns PR URL.

def submit_one(
    *, model: AntiSpoofingModel, dataset_spec: str,
    output_dir: Path, meta: dict,
    model_repo: str, hf_username: str, contact: str,
    api: HfApi,
) -> str: ...
    # Resolve → hybrid run → upload → build YAML → validate → open PR.
    # Returns PR URL.

def submit(
    *, model_module_spec: str, dataset_specs: list[str] | str,
    output_dir: Path, meta_path: Path,
    model_repo: str, hf_username: str, contact: str,
    continue_on_error: bool = False,
) -> dict[str, str]: ...
    # Iterates datasets sequentially; returns {dataset_slug: pr_url}.
```

### `meta.yaml` shape

The only new artifact authors hand-write per system. One file, reusable
across datasets.

```yaml
system:
  name: AASIST
  slug: aasist-clovaai-default
  description: |
    Reference AASIST, default config, FP32.
  code: https://github.com/clovaai/aasist
  checkpoint: https://huggingface.co/<owner>/<aasist-repo>
  paper:
    arxiv_id: "2110.01200"
    url: https://arxiv.org/abs/2110.01200
    bibtex: |
      @inproceedings{jung2022aasist, ... }

notes: |
  Optional free-form notes; copied into the submission YAML verbatim.
```

Validated against `data/submission_meta.schema.json` — a strict subset of
`submission.schema.json` covering exactly `system` (all required fields,
including `paper`) and optional `notes`. Missing fields → fail fast before
any model is loaded.

### `submit_one` execution

1. **Resolve dataset.** `loader.resolve(spec)` → `DatasetSource` with `canonical_id`, `revision`, `metrics`, `slug`.
2. **Hybrid run.** If `<output-dir>/<slug>/result.yaml` exists *and* its `dataset.revision` equals the resolved revision, reuse it. Otherwise call `Benchmark.run(model, datasets=[spec], output_dir=output_dir, skip_existing=True)`.
3. **Sanity-check result.** Every metric id listed in `source.metrics` must appear under `scores` in the result YAML. If not, raise — the runner is misbehaving.
4. **Upload scores.txt.** `upload_scores(api, model_repo, source.canonical_id, scores_path)` → `(scores_url, commit_oid)`. The URL is built from the returned OID, not from `main`.
5. **Build YAML.** `build_submission_payload(...)` merges the result YAML, the meta file, the URL, the local sha256 (already in `result.yaml`), and the submitter identity flags. `reproduction: {}`.
6. **Validate locally.** Pass the rendered YAML through `submission.parse_submission`. Abort if it fails the schema.
7. **Open PR.** `open_submission_pr(api, dataset_id, parent_commit=source.revision, slug=meta['system']['slug'], yaml_text=...)`. Uses `HfApi.create_commit(repo_type='dataset', create_pr=True, operations=[CommitOperationAdd(path_in_repo='submissions/<slug>.yaml', path_or_fileobj=BytesIO(yaml_bytes))])`. Print the PR URL.

The `parent_commit` is set to the resolved dataset revision so the PR is
anchored to the exact dataset state the run scored against — protecting
against races if upstream changes between resolve and PR open.

### `scaffold-dataset`

```
speech-spoof-bench scaffold-dataset \
    --name InTheWild \
    --output-dir ./dataset-builders/InTheWild \
    [--force]
```

1. If `output-dir` exists and is non-empty and `--force` not given → abort.
2. `mkdir -p output-dir/submissions`.
3. Copy every file under `speech_spoof_bench/data/dataset_skeleton/` into `output-dir`, replacing the literal token `{{NAME}}` in `README.md` and `eval.yaml` with the value of `--name`.

Template files (shipped in the package):

| File | Content |
|---|---|
| `README.md` | YAML frontmatter (`pretty_name: "{{NAME}}"`, `language: [en]`, `task_categories: [audio-classification]`, `size_categories: [unknown]`, `configs:` with the canonical default, `tags:` including `anti-spoofing`, `audio-deepfake-detection`, `speech`, `benchmark`, `arena-ready`, empty `arxiv: []`). Section headers per §1.3 with `TODO` placeholders for stats, license, citation. |
| `LICENSE.txt` | One line: `TODO: replace with upstream license verbatim before pushing.` |
| `eval.yaml` | `name: "{{NAME}}"`, description placeholder, `evaluation_framework: inspect-ai`, one task with `field_spec: {input: audio, target: label}`, solver/scorer matching §1.5, `metrics: [eer_percent]`. |
| `build_parquet.py` | Stub with module docstring listing the required schema (`{path, audio (16kHz mono), label (ClassLabel[bonafide, spoof]), notes (JSON string with utterance_id)}`) and a `def main(): raise NotImplementedError` body. |
| `submissions/README.md` | Verbatim canonical version (sourced from LA at implementation time). |
| `submissions/results_template.yaml` | Verbatim canonical template. |

No network. No state. Easy to re-run with `--force` after edits to the
template files.

## CLI surfaces

```
speech-spoof-bench submit \
    --model-module <mod:ClassName> \
    --datasets <dataset-id> [--datasets <dataset-id> ...]  # or --datasets all
    --model-repo <owner>/<repo> \
    --submission-meta <path-to-meta.yaml> \
    --hf-username <user> \
    --contact <email> \
    [--output-dir ./results] \
    [--continue-on-error]

speech-spoof-bench scaffold-dataset \
    --name <Pretty Name> \
    --output-dir <path> \
    [--force]
```

`--datasets all` is handled by `submit` (not pushed into `Benchmark.run`)
so each dataset gets its own PR; `submit` calls `fetch_manifest()` and
iterates `core_set + extended`.

## Testing

### Unit tests (mocked HF)

| File | Covers |
|---|---|
| `tests/test_submit_meta.py` | `load_meta` accepts a fully-populated meta; rejects missing `system.paper`, missing `system.slug`, wrong types, extra top-level keys. |
| `tests/test_submit_payload.py` | `build_submission_payload` round-trips through `submission.parse_submission` cleanly; `reproduction` is `{}`; `scores_url` is exactly the value passed in; `system.*` mirrors `meta.system`; `dataset.{id,revision,split}` come from the result YAML. |
| `tests/test_submit_upload.py` | `upload_scores` calls `HfApi.upload_file` with the expected `repo_id`, `path_in_repo` (= `.eval_results/<canonical_id>/scores.txt`), and `repo_type='model'`. Returned `commit_oid` is substituted into the returned URL verbatim. |
| `tests/test_submit_pr.py` | `open_submission_pr` calls `HfApi.create_commit` with `create_pr=True`, `parent_commit=<revision>`, exactly one `CommitOperationAdd` at `submissions/<slug>.yaml`, and returns the PR URL the mock yields. |
| `tests/test_submit_one.py` | End-to-end with all HF calls patched: hybrid-run skip path (existing result.yaml with matching revision) and cold path (calls `Benchmark.run`); failure cases: meta-schema fail, missing metric in result, schema fail on rendered YAML. |
| `tests/test_scaffold.py` | Writes the full tree into `tmp_path`; `{{NAME}}` replaced in `README.md` and `eval.yaml`; refuses non-empty target without `--force`; with `--force` overwrites. |
| `tests/test_cli.py` (extended) | `argparse` smoke for both new subcommands; missing required flags exit nonzero. |

### Manual smoke test (real HF)

Run after unit tests pass. Uses random-baseline + LA.

1. Author `/tmp/random-baseline-meta.yaml` matching the existing `random-baseline.yaml` but with **slug `random-baseline-phase7b`** (so it doesn't collide with the merged Phase 3 submission).
2. ```
   speech-spoof-bench submit \
       --model-module speech_spoof_bench.examples.random_baseline:RandomBaseline \
       --datasets SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA \
       --model-repo SpeechAntiSpoofingBenchmarks/random-baseline-asas \
       --submission-meta /tmp/random-baseline-meta.yaml \
       --hf-username SpeechAntiSpoofingBenchmarks \
       --contact k.n.borodin@mtuci.ru
   ```
3. Confirm: a new commit lands on `main` of the model repo at `.eval_results/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA/scores.txt`; an HF PR opens on the LA dataset repo touching only `submissions/random-baseline-phase7b.yaml`; `scores_url` in that YAML pins the new commit sha; `reproduction:` is `{}` (empty); `submitter.hf_username` and `submitter.contact` come from the flags; `submitted_at` is today.
4. Download the YAML from the PR branch: `speech-spoof-bench validate-submission <local-copy>` → exits 0.
5. `speech-spoof-bench reproduce --scoring <local-copy>` → exits 0; EER matches.
6. **Close the PR without merging** to avoid polluting the dataset repo with a duplicate submission.
7. Separately: `speech-spoof-bench scaffold-dataset --name TestScaffold --output-dir /tmp/test-scaffold` → inspect the tree, `cat` the eval.yaml and README to confirm `{{NAME}}` substitution and that all §1.1 files are present. `rm -rf /tmp/test-scaffold` after.

### Stop-the-line

- If the PR diff shows changes outside `submissions/<slug>.yaml`, fix before merging anything.
- If `validate-submission` or `reproduce --scoring` fails on the generated YAML, the bug is in `submit`, not in the validators.
- If `parent_commit` is not honored (PR opens against a stale dataset state), revisit `open_submission_pr`.

## Open items

1. **`--datasets all` source**: read from `arena-manifest`'s `core_set + extended`, or only `core_set`? Lean: both (`core_set + extended`), since a real submitter wants maximum coverage. Document in `submit --help`.
2. **Hybrid run revision mismatch**: if a stale `result.yaml` exists with a different revision, today we'd silently re-run. Should we instead error and require `--force-rerun`? Lean: silent re-run is fine — `result.yaml` is local artifact and the runner already handles output overwrite cleanly.

Both can be settled during implementation without re-doing the design.
