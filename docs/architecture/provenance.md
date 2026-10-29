# Corpus provenance & "trained on" (n/a\*) cells

A leaderboard cell is **flagged** when the model was trained on data that overlaps the
test dataset. Flagged cells show **n/a\*** on the Overview (score + shared corpus in the
tooltip), are **left out of the Mean / rank** in both views (no penalty), still **count
toward coverage / tier**, are never highlighted as best, and on the Per-dataset tab get a
`*` and no rank place.

Design record: `docs/plans/2026-10-29-arena-contamination-provenance-design.md`.

## The data: `provenance.yaml` (arena-manifest repo)

Lives next to `manifest.yaml` in `SpeechAntiSpoofingBenchmarks/arena-manifest`. Optional:
no file ⇒ nothing is flagged. Not validated by the package's strict manifest schema, so
editing it needs **no package release** — commit, and the next Arena refresh picks it up.

```yaml
version: 1
corpora:                              # shared vocabulary
  LibriSpeech: {parts: [train, dev, test]}
  LibriTTS: {from: LibriSpeech, parts: [train, dev, test]}
  ASVspoof2019-LA: {from: VCTK, parts: [train, dev, eval]}
  ASVspoof2021-LA: {from: "ASVspoof2019-LA:eval", parts: [eval]}   # derived from a split
  Common-Voice: {parts: [ar, de, en, ...]}                          # parts = any disjoint subsets
datasets:                             # Arena dataset id -> what it is built from
  SpeechAntiSpoofingBenchmarks/SONAR: ["LibriTTS:test"]
models:                               # systems that don't declare training data themselves
  "*-antideepfake*":                  # fnmatch glob on the slug
    source: https://arxiv.org/abs/2506.21090
    trained_on: ["ASVspoof2019-LA:train", "ASVspoof2019-LA:eval", VCTK, ...]
```

An **entry** is `Corpus` (whole corpus) or `Corpus:part` (an official split or other
disjoint subset listed in that corpus's `parts`).

**What a dataset lists:** the corpora of its bona fide **audio** and of the **speakers**
cloned/converted for its spoofs. Not text-only reuse, not the TTS/VC systems' own training
data. Datasets with no public source corpus (original/web recordings) are omitted.

## The matching rule: same data only

A training entry flags a dataset entry iff it is the **same corpus** with compatible parts
(equal, or either side whole) — the model trained on that very data. `from` links are
**documentation only**: training on a parent pool (MLS, Common Voice, VCTK) or on a sibling
built from the same split (ASVspoof2021 DF vs 2021 LA / 2019 LA) does **not** flag a dataset
(maintainer decision 2026-10-29).

So a dataset lists **what it is** — its own corpus/split plus any exact official split it
literally consists of (SONAR bona fide = `LibriTTS:test`; DECRO English = `ASVspoof2019-LA:eval`)
— not the pools it samples from.

| Training entry | Dataset entry | Result |
|---|---|---|
| `ASVspoof2019-LA:eval` | `ASVspoof2019-LA:eval` | flagged |
| `ASVspoof2019-LA:train` | `ASVspoof2019-LA:eval` | clean (different split) |
| `LibriTTS` (whole) | `LibriTTS:test` (SONAR) | flagged |
| `MLS` / `Common-Voice` (XLS-R pretraining) | `ASVspoof5:eval`, `PyAra`, `CVoiceFake` | clean |
| `ASVspoof2021-DF:eval` | `ASVspoof2019-LA:eval` | clean |

A system's training entries = matching `models:` globs (several may match; they add up)
**+** each matching entry's `backbone:` pretraining data **+** its submissions' optional
`system.trained_on` (Arena dataset ids → that dataset's `datasets:` entries, or the id itself).

### Backbones (pretraining counts)

Pretraining counts like any training data: each front-end's pretraining corpora are listed
once under `backbones:` and referenced by `backbone: <name>`. Under the same-data rule it only
flags a dataset that *is* one of those corpora — pools like MLS / Common Voice flag nothing.

### Not-run cells

A dataset the model trained on but was never run on also shows **n/a\*** (tooltip: "not
run"). Display only: it still counts as not run for the tier and the pooled absence
penalty — the model does not move tiers.

## Code (Arena)

| File | Role |
|---|---|
| `provenance.py` | `fetch_raw()` (HF, or `ARENA_PROVENANCE_PATH` local file), `parse()` (never raises: problems → warnings; resolves `backbones:`), `overlap(prov, train, test)`, `contaminated(rows, prov, dataset_ids)` |
| `contamination.py` | display helpers: n/a\* cell, per-dataset `*` marker, legend; `contaminated_by_slug(rows, manifest)` |
| `ingest.py` | fetches the file each refresh, attaches the raw dict to the cached manifest under `_provenance`, adds parse warnings to the state |
| `ranking.global_scores` | skips flagged (system, dataset) pairs — no penalty |
| `leaderboard.py` | n/a\* cells (overview), `*` + unranked (per-dataset); best-value highlighting ignores non-numeric cells |

## Maintaining it

- **New dataset:** add its `datasets:` entry (and any new `corpora:`) in the same manifest
  PR that adds it to the Core Set. Cite the source in a YAML comment.
- **New model that trained on public corpora:** prefer the submitter declaring
  `system.trained_on`; otherwise add a `models:` entry with a `source:` URL.
- **Check locally before committing:** `cd arena-live && ./dev_local.sh` (picks up
  `../arena-manifest/provenance.yaml` automatically), or in Python:
  `provenance.parse(yaml.safe_load(open(path)))` — the warnings list must be empty.
