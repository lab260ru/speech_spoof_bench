# Submission Lifecycle & Schemas

Three YAML shapes flow through the system. Knowing which is which is the key to not
getting confused:

| Shape | `schema_version` | Authored by | Lives where | Schema file |
|-------|------------------|-------------|-------------|-------------|
| **meta** | *(none)* | the submitter (by hand) | local file, e.g. `meta.yaml` | `data/submission_meta.schema.json` |
| **result** | `1` | the badge layer (machine) | pasted into the model card | `schema/result.schema.json` |
| **submission** | `4` | `submit` (machine) | `submissions/<slug>.yaml` in the dataset repo | `schema/submission.schema.json` |

> There is also a transient `result.yaml` written by `Benchmark.run()` that carries
> `schema_version: 4` and the benchmark's view of the run — don't confuse it with the
> badge-layer **result** (`schema_version: 1`) projected by `badge.py`. The benchmark's
> file is an intermediate the `submit` command reads and merges with the meta.

All three are validated with `jsonschema` (draft-07) against the **bundled** schema JSON
files, so the validation rules ship with the package version.

---

## 1. The meta file (what the human writes)

The only file a submitter authors by hand. It describes the *system*, not the results.

Required: a `system` object with `name`, `slug`, `description`, `code`, `checkpoint`.
Optional: `system.paper` (`arxiv_id` / `url` / `bibtex`), `system.params_millions`,
and a top-level `notes` string.

- `slug` must match `^[a-z0-9][a-z0-9-]*$` (lowercase, digits, dashes — no spaces, no
  underscores, no capitals). The slug is the system's identity everywhere (badge URLs,
  arena rows, file name).
- `code` and `checkpoint` must be URIs.
- **`paper` is optional.** A system *with* a paper can compete in the ranked tiers
  (gold/silver/bronze). A system *without* one is welcome but lands in the unranked
  **`unpublished`** tier no matter how good its scores. (This is the project's paper
  policy — see the bottom of [submitting/submit-model.md](../submitting/submit-model.md).)

```yaml
system:
  name: "AASIST"
  slug: "aasist-clovaai-default"
  description: "Reference AASIST, default config, FP32."
  code: "https://github.com/clovaai/aasist"
  checkpoint: "https://huggingface.co/owner/aasist/blob/main/model.pth"
  paper:
    arxiv_id: "2110.01200"
    url: "https://arxiv.org/abs/2110.01200"
    bibtex: "@article{jung2021aasist, ...}"
  params_millions: 7.5
notes: "Trained on ASVspoof2019 LA train, no augmentation."
```

---

## 2. The submission file (the pointer)

Built by `submit.build_submission_payload()` by merging the **meta** with the
benchmark's **result.yaml**, then uploaded as a PR. It is a *pointer*: it contains the
scores' URL + hash, never the score file or any audio.

`schema_version: 4`. Required top-level keys: `system`, `dataset`, `scores`, `artifact`,
`submitter`, `submitted_at`. Optional: `reproduction`, `notes`.

Key field rules (from `submission.schema.json`):

- `dataset.id` matches `^[^/]+/[^/]+$` (`org/name`); `dataset.revision` matches
  `^[0-9a-f]{7,40}$` (a commit SHA, lowercase hex).
- `scores` needs `n_trials` + `n_skipped` **and at least one metric** (so ≥3 properties).
- `artifact.scores_url` must be a **commit-pinned** HF resolve URL
  (`…/resolve/<sha>/…`) — a `…/resolve/main/…` URL fails validation. `scores_sha256` is
  64 hex chars. `artifact.bench_version` is the `speech-spoof-bench==X.Y.Z` string copied
  from the result.
- `reproduction` is `oneOf`: an **empty `{}`** (as opened by the submitter) **or** a
  fully populated block (`reproduced_by`, `reproduced_at`, `reproduced_bench_version`,
  `match`). It can't be partial — the maintainer fills the whole block at merge.

```yaml
schema_version: 4
system:        # copied from meta
  name: "AASIST"
  slug: "aasist-clovaai-default"
  description: "Reference AASIST, default config, FP32."
  code: "https://github.com/clovaai/aasist"
  checkpoint: "https://huggingface.co/owner/aasist/blob/main/model.pth"
  paper: {arxiv_id: "2110.01200", url: "https://arxiv.org/abs/2110.01200", bibtex: "@article{...}"}
  params_millions: 7.5
dataset:
  id: SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA
  revision: 9b2040e8c57749dcd9a4f16ad61b4f47626b89ec
  split: test
scores:
  eer_percent: 4.23
  n_trials: 71237
  n_skipped: 0
artifact:
  scores_url: https://huggingface.co/owner/aasist/resolve/abc123def456/.eval_results/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA/scores.txt
  scores_sha256: 4f9b0e1a...   # 64 hex
  bench_version: "speech-spoof-bench==0.1.0"
reproduction: {}               # ← maintainer fills this at merge
submitter:
  hf_username: "owner"
  contact: "me@example.com"
submitted_at: "2026-05-29"
notes: "..."
```

> **Gotcha — dates.** PyYAML parses `submitted_at: 2026-05-29` into a `datetime.date`
> object. `submission.parse_submission()` recursively coerces these back to ISO strings
> before validating, because jsonschema's `format: date` only accepts strings.

---

## 3. The result file (the badge layer)

A minimal projection of the submission that the contributor pastes into their HF model
card. `schema_version: 1` (independent of the submission's `4`). Produced by
`badge.build_result_yaml()` / `_project_submission_for_result()`, which keeps `system`
(name/slug/`paper.arxiv_id` only), `dataset`, `scores`, an `arena` block (`url`,
`system_url = {arena_url}?system={slug}`), and `artifact.scores_url` only. It drops the
submitter, the reproduction block, `bench_version`, and notes. → [badges.md](badges.md)

---

## The dataset repo layout

Each dataset PR targets a repo shaped like this:

```
SpeechAntiSpoofingBenchmarks/<name>/        (HF dataset repo)
├── README.md                 # YAML front-matter = HF dataset card; must include tag `arena-ready`
├── eval.yaml                 # task + metrics; metrics[0] is the primary metric
├── data/
│   └── test-*.parquet        # the canonical 4-column schema
└── submissions/
    ├── README.md             # submitter instructions
    ├── results_template.yaml # copy-paste starting point
    └── <slug>.yaml           # one per merged system
```

Canonical parquet schema (enforced by validation): `path: string`, `audio: Audio(16000)`,
`label: ClassLabel(["bonafide", "spoof"])`, `notes: string` (JSON with a unique
`utterance_id`).

---

## Validation: the D- and S-checks (`validate.py`)

`validate_dataset(spec, skip_submissions=False)` returns a `ValidationReport` of
`CheckResult(id, passed, message)`. It is **all-or-nothing**: `report.ok` is true only if
every check passes.

**Dataset checks (D1–D7):**

| Check | What it verifies |
|-------|------------------|
| D1 | Columns are exactly `{path, audio, label, notes}`. |
| D2 | `label` is a `ClassLabel` named `[bonafide, spoof]` (int 0/1 soft-passes). |
| D3 | **First row only** (spot-check, not every clip): sample rate is 16000 Hz and duration ≥ 1.0 s. |
| D4 | First ~100 rows: `notes` is valid JSON containing `utterance_id`. |
| D5 | Uniqueness: no duplicate `utterance_id` or `path`. |
| D6 | README front-matter has `{license, language, pretty_name, task_categories, size_categories, configs, tags, arxiv}` and `tags` includes `arena-ready`. |
| D7 | Every metric id in `eval.yaml` is registered in the package (delegated to `loader.resolve`). |

**Submission checks (S1–S4)**, skipped if `--skip-submissions`:

| Check | What it verifies |
|-------|------------------|
| S1 | The submission YAML passes schema validation. |
| S2 | A `reproduction` block is present (i.e. it has been reviewed). |
| S3 | `scores_url` is reachable. |
| S4 | `sha256(scores.txt)` matches the claimed `scores_sha256`. |

`validate-submission PATH` is the offline-only subset: schema check, no network.

---

## Reproduction (`reproduce.py`)

`run_scoring(yaml_path, tolerance=1e-6, force_remote=False)` is the verification gate
used by PR verification and the nightly job:

1. Schema-validate the submission YAML.
2. `hf_fetch.download(scores_url)` → re-check SHA-256 against `scores_sha256`.
3. Parse `scores.txt` into `{utt_id: score}`.
4. Stream the dataset's **labels** at the pinned `revision`, using parquet column
   projection (`columns=["notes", "label"]`) so the audio bytes are never transferred.
   Honours the local registry unless `force_remote`.
5. Coverage check: every scored id must be in the dataset; `len(scores) + n_skipped == n_trials`.
6. For each metric in `scores` (excluding `n_trials`/`n_skipped`): recompute and compare
   to the claimed value within `tolerance`. Any mismatch → exit `1`.

`reproduce --inference` (re-run the model from the checkpoint) raises
`NotImplementedError` today.

---

## Scaffolding a dataset (`scaffold.py`)

`scaffold_dataset(name, output_dir, force=False)` materialises the packaged
`data/dataset_skeleton/` template, substituting `{{NAME}}` in `README.md` and `eval.yaml`
only (other files copied verbatim; `__pycache__`/`__init__.py` skipped). It refuses to
write into a non-empty dir unless `force=True`. The template ships **inside the wheel**
(read via `importlib.resources`), so editing the skeleton requires reinstalling the
package. → [developing/new-dataset.md](../developing/new-dataset.md)

---

### Label fetch fast path

**Label fetch fast path.** `reproduce --scoring` resolves labels in this order:
in-process cache → `data/labels.parquet` (one request) → `data/test-*.parquet`
shard stream (fallback). Labels at a pinned `dataset.revision` are immutable, so
they are memoized per `(dataset_id, revision)` for the life of the process.

**Nightly skip-unchanged.** `nightly-revalidate` skips a submission entirely when
`(scores_sha256, dataset.revision, installed bench_version)` is unchanged since
the last green run (state cached via GitHub Actions). A package release bumps
`bench_version` and re-verifies everything; a weekly `--full` sweep and the
`--full` flag force a complete pass. Trade-off: artifact drift at `scores_url`
on an otherwise-unchanged submission is only detected on the next full sweep.
</content>
