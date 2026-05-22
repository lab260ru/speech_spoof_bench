# Phase 7a — Validators Design

**Status**: approved, ready for plan
**Date**: 2026-05-22
**Scope**: `validate-submission`, full `validate-dataset`, `reproduce --scoring`
**Out of scope**: `submit`, `scaffold-dataset`, `reproduce --inference` (Phase 7b / 8)

Phase 7 of `ROADMAP.md` is split into 7a (validators, this spec) and 7b
(authoring tools). 7a delivers the maintainer-side gate the spec marks
mandatory before merging any submission (§1.7). All three commands are
pure-read: no HF writes, no PR creation.

---

## 1. Goals

1. Submitters can sanity-check their submission YAML offline before pushing.
2. Dataset maintainers can run one command that exhaustively checks a dataset
   repo against the v4 spec (§1.9), including all submission YAMLs it
   contains.
3. Arena maintainers can reproduce any submission's metrics from its pinned
   `scores.txt` in seconds, with no audio decode and no GPU — the mandatory
   gate before merging an HF PR (§1.7 level "scoring").

## 2. CLI surface

```
speech-spoof-bench validate-submission <path-to-yaml>
speech-spoof-bench validate-dataset    <repo-id-or-local-path> [--skip-submissions]
speech-spoof-bench reproduce --scoring <path-to-yaml> [--tolerance 1e-6]
speech-spoof-bench reproduce --inference <path-to-yaml>        # raises NotImplementedError
```

`--scoring` and `--inference` are a mutually exclusive flag group under
`reproduce`. `--inference` is wired now so 7b/8 is a pure addition.

Exit codes: `0` on success, `1` on any failure. No other codes used.

## 3. Module layout

```
src/speech_spoof_bench/
├── validate.py        # NEW — dataset-side checks (D1–D7) + submission
│                      #       cross-checks (S1–S4). Aggregating runner.
├── reproduce.py       # NEW — --scoring impl: fetch, sha, recompute, diff.
├── hf_fetch.py        # NEW — wraps hf_hub_download:
│                      #         - parses HF resolve URL into (repo, sha, path)
│                      #         - honors HF_TOKEN env var if set
│                      #         - returns local path + sha256 of file
├── submission.py      # EXISTING — parse_submission, schema loader
└── cli.py             # MODIFIED — three new subcommands wired up
```

Public functions in `validate.py` / `reproduce.py` take Python objects
(parsed YAML dicts, paths, `DatasetSource`) so they are testable without
invoking the CLI.

No new third-party dependencies. Uses `huggingface_hub`, `datasets`, `yaml`,
`jsonschema`, `scipy`, `numpy` already on the install line.

## 4. `validate-submission` (offline)

Single-file, pure schema check. Reuses `submission.parse_submission`, which
already validates the JSON Schema (slug pattern, `scores_url` regex pinning
to `/resolve/<sha>/`, sha256 hex pattern, ISO `date` format, schema_version,
`reproduction:` block required).

```python
def _cmd_validate_submission(args):
    text = Path(args.path).read_text()
    try:
        submission.parse_submission(text)
    except SubmissionValidationError as e:
        print(f"FAIL {args.path}: {e}", file=sys.stderr)
        return 1
    print(f"OK: {args.path}")
    return 0
```

No flags. No network. Milliseconds. Submitters run this before opening a PR.

## 5. `validate-dataset` (full §1.9, network)

Accepts either an HF repo id (`org/name`) or a local directory path.
**Aggregating**: collects every failure, prints a per-check report, exits 1
if any check failed.

### 5.1 Dataset-side checks (D1–D7)

Always run. For local paths, files are read from disk; for repo-ids, each
auxiliary file is `hf_hub_download`'ed individually and the parquet is read
via `load_dataset(..., streaming=True)`.

| #  | Check | Spec |
|----|-------|------|
| D1 | First row has exactly `{path, audio, label, notes}` | §1.2 / §1.9 |
| D2 | `label` `ClassLabel` names are `["bonafide", "spoof"]` | §1.2 / §1.9 |
| D3 | First row's `audio.sampling_rate == 16000` and `len(array)/sr ≥ 1.0` | §1.9 |
| D4 | For 100 sampled rows: `json.loads(notes)` succeeds, `utterance_id` non-empty | §1.9 |
| D5 | Full-scan: all `utterance_id` unique, all `path` unique | §1.9 |
| D6 | `README.md` YAML frontmatter has all keys from §1.4: `license`, `language`, `pretty_name`, `task_categories`, `size_categories`, `configs`, `tags` (must contain `arena-ready`), `arxiv` | §1.4 / §1.9 |
| D7 | `eval.yaml` parses, matches §1.5 shape, every id in `metrics:` is registered in `speech_spoof_bench.metrics` | §1.5 / §1.9 / §2.6 |

D3 decodes one audio row to verify the sampling rate and length. D5 streams
the dataset in full and accumulates `utterance_id` and `path` into Python
sets — for the largest planned datasets (~120k rows) this is a few MB of RAM
and one full streamed pass.

### 5.2 Submission-side checks (S1–S4)

Run unless `--skip-submissions`. For each YAML under `submissions/`
(excluding `README.md`, `results_template.yaml`):

| #  | Check | Spec |
|----|-------|------|
| S1 | YAML parses + schema-validates via `submission.parse_submission` | §1.6 |
| S2 | `reproduction:` block is non-empty | §1.7 |
| S3 | `artifact.scores_url` is fetchable via `hf_fetch.download` | §1.9 |
| S4 | `sha256(fetched file) == artifact.scores_sha256` | §1.9 |

S2 is structurally redundant with the schema (which marks `reproduction` as
required) but kept as a distinct line in the report so its mandatory status
is visible to humans.

`dataset.revision` reachability is **not** checked. `validate-dataset`
operates on the current HEAD of the dataset, while submissions pin
historical revisions; verifying historical revisions belongs in
`reproduce`, not here.

### 5.3 Report format

```
Dataset: SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA
  [✓] D1 schema matches v4
  [✓] D2 label classes
  [✓] D3 audio sampling rate
  [✓] D4 notes JSON sample
  [✓] D5 uniqueness scan
  [✓] D6 README frontmatter
  [✗] D7 eval.yaml: metric 'min_tdcf' not registered

Submissions (2):
  [✓] submissions/random-baseline.yaml
  [✗] submissions/aasist.yaml
        S3 scores_url unreachable: HTTP 404
        S4 skipped (depends on S3)

2 checks failed.
```

### 5.4 `--skip-submissions`

Escape hatch for dataset maintainers iterating on `build_parquet.py` who
don't want to wait on N network fetches. Skips S1–S4 only; D1–D7 still run.

## 6. `reproduce --scoring`

Single-submission, sequential, fail-fast (this is a precise verification,
not an audit).

### 6.1 Workflow

1. Parse YAML — `submission.parse_submission`. Schema error → exit 1.
2. Fetch `scores.txt` — `hf_fetch.download(yaml['artifact']['scores_url'])`.
   `HF_TOKEN` env honored if set. 404/network error → exit 1.
3. Verify `sha256(fetched) == yaml['artifact']['scores_sha256']`. Mismatch
   → exit 1 with both hashes.
4. Parse `scores.txt` into `{utt_id: float}`.
5. Stream labels from the pinned dataset revision (see §6.2).
6. Cross-check coverage (see §6.3).
7. Recompute every metric (see §6.4).
8. Print report; exit 0 on full match.

### 6.2 Streaming labels without audio decode

```python
ds = load_dataset(
    yaml["dataset"]["id"],
    split=yaml["dataset"]["split"],
    streaming=True,
    revision=yaml["dataset"]["revision"],
    columns=["notes", "label"],  # load-time pushdown — drops audio at parquet level
)
labels: dict[str, int] = {}
for row in ds:
    notes = json.loads(row["notes"])
    labels[notes["utterance_id"]] = int(row["label"])
```

Passing `columns=["notes", "label"]` at `load_dataset()` time sets
`ParquetConfig.columns`, which propagates to `ParquetFileFormat.to_batches(columns=...)`.
This IS a genuine PyArrow column filter at the row-group level — the parquet
reader issues HTTP range requests for only those column chunks. Audio bytes
are not requested, not downloaded, not decoded. For ASVspoof2019_LA this is
a few MB of HTTP traffic.

**Important**: calling `ds.select_columns(["notes", "label"])` post-construction
is a CPU/memory projection only; the underlying parquet read still pulls every
column's bytes from the network. Only the load-time `columns=` form achieves
network-level pushdown.

`streaming=True` is hardcoded in `reproduce.py` — not configurable —
because non-streaming mode downloads the full parquet shards (including
audio) to `~/.cache/huggingface/datasets/`.

If `revision` isn't reachable, the `load_dataset` call raises; the command
exits 1 with a message naming the revision.

### 6.3 Coverage checks

Let `S = set(scores)`, `L = set(labels)`, `n_skipped_claim = yaml['scores']['n_skipped']`,
`n_trials_claim = yaml['scores']['n_trials']`.

- `S - L != ∅` → fail ("submitter scored utterances not in dataset").
- `len(S) + n_skipped_claim != n_trials_claim` → fail.
- `len(L - S) > n_skipped_claim` → fail ("more skipped than claimed").

`L - S` ≤ `n_skipped_claim` is tolerated (submitter declared they couldn't
score some items).

### 6.4 Metric recomputation

For every key in `yaml['scores']` other than `n_trials` and `n_skipped`:

- `metric = metrics.get_metric(key)`. Unknown id → fail ("metric X not
  registered in this version of speech-spoof-bench").
- Score-side dict is restricted to `{k: scores[k] for k in scores if k in labels}`
  so the metric sees only utterances the dataset actually contains.
- Labels-side dict is restricted to the same key set.
- `result = metric.fn(scores_subset, labels_subset)`.
- `abs(result.value - yaml['scores'][key]) <= tolerance` (default `1e-6`).
  Mismatch → fail with claimed vs recomputed printed to full precision.

### 6.5 Success report

```
OK reproduced: SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA @ 151aa4c6
  scores_sha256: matched (71ac…5587)
  eer_percent:   claimed 49.870836  recomputed 49.870836  (Δ 0.0e+00)
  n_trials:      71237 (skipped 0)
```

### 6.6 `--inference` placeholder

```python
if args.inference:
    raise NotImplementedError("--inference lands in Phase 7b/8")
```

Argparse defines the mutually-exclusive group so the CLI shape is locked.

## 7. HF fetch helper

`hf_fetch.py` is small but centralised so caching + auth behavior is
consistent across both consumers (`validate-dataset` S3/S4 and `reproduce`
step 2).

```python
import re
from huggingface_hub import hf_hub_download
import os, hashlib
from pathlib import Path

_HF_RESOLVE_RE = re.compile(
    r"^https://huggingface\.co/(?P<repo>[^/]+/[^/]+)/resolve/(?P<sha>[0-9a-f]{7,40})/(?P<path>.+)$"
)

def parse_hf_resolve_url(url: str) -> tuple[str, str, str]:
    m = _HF_RESOLVE_RE.match(url)
    if not m:
        raise ValueError(f"not a commit-pinned HF resolve URL: {url}")
    return m["repo"], m["sha"], m["path"]

def download(url: str) -> tuple[Path, str]:
    """Download a commit-pinned HF resolve URL. Returns (local_path, sha256)."""
    repo, sha, path = parse_hf_resolve_url(url)
    token = os.environ.get("HF_TOKEN") or None
    local = hf_hub_download(
        repo_id=repo, filename=path, repo_type="model",
        revision=sha, token=token,
    )
    h = hashlib.sha256()
    p = Path(local)
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return p, h.hexdigest()
```

`repo_type="model"` reflects current `submission.schema.json` — `scores_url`
points at submitter model repos. If we later add support for scores hosted
in other repo types, this stays a one-line change.

## 8. Testing

### 8.1 Layout

```
tests/
├── fixtures/
│   ├── valid_submission.yaml
│   ├── invalid_submission_no_reproduction.yaml
│   ├── invalid_submission_unpinned_url.yaml
│   ├── invalid_submission_bad_sha.yaml
│   ├── invalid_submission_bad_slug.yaml
│   ├── invalid_submission_wrong_schema_version.yaml
│   ├── invalid_submission_malformed_yaml.yaml
│   ├── scores_known.txt
│   └── mini_dataset/                     # local-path dataset for validate-dataset
│       ├── data/test-00000-of-00001.parquet
│       ├── README.md
│       ├── eval.yaml
│       └── submissions/random.yaml
├── test_validate_submission.py
├── test_validate_dataset.py
├── test_reproduce.py
└── test_hf_fetch.py
```

### 8.2 Unit tests (offline; default `pytest`)

- **`validate-submission`**: one passing fixture, one fixture per schema
  failure branch (~6). Each asserts exit code and a substring of the
  message.
- **`validate-dataset` D1–D7**: happy path via `mini_dataset`, plus one
  test per D-check failure produced by mutating the fixture
  (monkeypatch a feature, swap eval.yaml, etc.). Uses
  `--skip-submissions`.
- **`validate-dataset` S1–S4**: monkeypatch `hf_fetch.download` to return
  local fixture paths; test the aggregator with mixed pass/fail
  submissions and assert the report shape.
- **`reproduce --scoring`**: monkeypatch `hf_fetch.download` to return
  `tests/fixtures/scores_known.txt`; monkeypatch the dataset stream to
  yield a tiny in-memory iterable of `{notes, label}` dicts. Covers: sha
  mismatch, metric mismatch, unknown metric id, missing utterances,
  `n_trials` mismatch, success. Plus a test that asserts
  `load_dataset` receives `columns=["notes", "label"]` at call time (not
  via post-construction `select_columns`), enforcing the network-level
  pushdown invariant.
- **`hf_fetch`**: pure parsing tests for the HF-resolve-URL regex
  (positive + negative cases); one test mocks `hf_hub_download` and
  asserts the `token=` kwarg is forwarded from `HF_TOKEN`.

### 8.3 Integration test (default `pytest`, not skipped)

`tests/test_reproduce_integration.py::test_random_baseline_real`

- Reads the on-HF `submissions/random-baseline.yaml` from
  `SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA`.
- Runs `reproduce --scoring` end-to-end against it.
- Asserts exit 0 and a successful match report.
- Snapshots `~/.cache/huggingface/datasets/` size before and after;
  asserts no growth attributable to ASVspoof2019_LA audio shards.

No `@pytest.mark.network` marker. HF unreachable → test fails (not
skips). This is the canary for HF API drift, URL pattern changes, and
the audio-not-downloaded guarantee.

## 9. Deferred (out of scope for 7a)

Captured here so they aren't re-discovered as gaps later:

- `submit` (Phase 7b) — run + upload scores + build YAML + open PR.
- `scaffold-dataset` (Phase 7b).
- `reproduce --inference` (Phase 8) — wired as `NotImplementedError`.
- `reproduce --scoring` accepting `<repo>@<ref>:<file>` to fetch a YAML
  directly from an HF dataset PR branch — file-path only at launch. Will
  be tracked in `ROADMAP.md` under 7a's Deferred bullet.
- CI/webhook integration (Phase 8).
- Badge generation (Phase 9).

## 10. Done when

- [ ] `validate-submission` exits 0 on the existing on-HF
      `random-baseline.yaml` (downloaded locally) and exits 1 on each
      invalid fixture.
- [ ] `validate-dataset SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA`
      exits 0 with all D1–D7 and S1–S4 checks green for the current
      dataset state.
- [ ] `validate-dataset --skip-submissions <local-mini>` exits 0.
- [ ] `reproduce --scoring submissions/random-baseline.yaml` (against
      the on-HF dataset) exits 0 with `eer_percent` matching the
      claimed `49.870836…` within `1e-6`.
- [ ] `reproduce --inference` raises `NotImplementedError`.
- [ ] Network traffic for one `reproduce --scoring` run on LA is under
      10 MB (verified by integration test cache snapshot).
- [ ] Full `pytest` invocation passes (offline unit tests + the one
      network integration test).
