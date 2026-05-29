# Submit a dataset

Add a new anti-spoofing benchmark to the org so models can be evaluated on it. The
flow is: **check license → scaffold → build parquet → validate → push → manifest PR**.

> 💡 Not sure if your dataset fits, or need help hosting it? Email
> **[k.n.borodin@mtuci.ru](mailto:k.n.borodin@mtuci.ru)**.

## 1. Redistribution check (do this first)

We only host datasets we can **redistribute under their upstream license**. If you
can't legally rehost the audio, it's out of scope — loader-only / proxy repos are not
accepted. Ship the verbatim `LICENSE.txt`.

## 2. Scaffold the repo skeleton

```bash
speech-spoof-bench scaffold-dataset \
  --name <Source><Year>_<Partition> --output-dir ./<name>
```

Naming is `<Source><Year>_<Partition>` (e.g. `ASVspoof2019_LA`); real-world sets use a
plain name (e.g. `InTheWild`).

## 3. Build the parquet to the canonical schema

Every row must be exactly:

| field | type | notes |
|---|---|---|
| `path` | string | stable, unique archive-relative path |
| `audio` | Audio(16 kHz) | mono, resampled at build time |
| `label` | ClassLabel `[bonafide, spoof]` | index 0 = bonafide |
| `notes` | string (JSON) | must parse and contain a unique `utterance_id` |

## 4. README, eval.yaml, citation

- **README frontmatter:** include the `arena-ready` tag (this is how the Arena
  discovers the dataset) and the `arxiv:` list.
- **`eval.yaml`:** the task config and the `metrics:` list (e.g. `eer_percent`). The
  first metric is the dataset's *primary* metric used for ranking.
- **Citation:** a section with the paper's arXiv link and a BibTeX block.

## 5. Validate until green

```bash
speech-spoof-bench validate-dataset ./<name>
```

This checks the schema, the label classes, `utterance_id` uniqueness, the 16 kHz
sample rate, the README frontmatter, and `eval.yaml`. Fix everything it reports before
pushing.

## 6. Push, then open a manifest PR

Push to `huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/<name>`, then open a PR on
the **`arena-manifest`** repo adding your dataset under `core_set` or `extended` with a
**pinned `revision`** (a commit sha — this is the exact version the Arena scores
against).

## 7. Core vs Extended

- **Core** datasets count toward tier coverage and the global rank.
- **Extended** datasets are shown and rankable per-dataset, but don't gate tiers.

New datasets usually start in **Extended** and are promoted to **Core** once stable.

---

Questions? Email **[k.n.borodin@mtuci.ru](mailto:k.n.borodin@mtuci.ru)**.
