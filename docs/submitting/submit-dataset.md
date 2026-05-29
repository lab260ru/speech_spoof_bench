# Submit a dataset

1. **Redistribution check:** the dataset must be redistributable under its
   upstream license (§1.8). Loader-only repos are out of scope.
2. **Scaffold:**
   ```bash
   speech-spoof-bench scaffold-dataset --name <Source><Year>_<Partition> --output-dir ./<name>
   ```
3. **Build parquet** to the canonical schema `{path, audio(16kHz), label[bonafide,spoof], notes(JSON w/ utterance_id)}` (§1.2).
4. **README frontmatter** (incl. `arena-ready` tag + `arxiv:` list), **`eval.yaml`** (metrics list), and a **Citation** block with arXiv link + BibTeX.
5. **Validate** until green:
   ```bash
   speech-spoof-bench validate-dataset ./<name>
   ```
6. **Push** to `huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/<name>`, then
   open a PR on `arena-manifest` adding it under `core_set` or `extended` with a pinned `revision`.
7. **Core vs Extended:** Core datasets count toward tier coverage and global rank;
   Extended are shown but don't gate tiers.
