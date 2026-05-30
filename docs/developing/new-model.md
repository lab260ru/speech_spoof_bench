# Developing a New Model

This is the **developer-focused** companion to the step-by-step
[submit-model.md](../submitting/submit-model.md) guide. That guide walks the submission
clicks; this one is about *building the wrapper so it actually reproduces*, and catching
the failures before CI does.

The whole job is one Python class. Everything else (running, scoring, EER, uploading,
the PR) is done for you.

## Step 1 — Write the class

Pick the base that fits:

- **`SimpleAntiSpoofingModel`** — you implement `score(audio, sr) -> float`. Use this
  unless you have a real batching win.
- **`AntiSpoofingModel`** — you implement `score_batch(audios, srs) -> list[float]` for
  true batched inference.

```python
# my_model.py
from __future__ import annotations
import numpy as np
from speech_spoof_bench.model import SimpleAntiSpoofingModel

class MyModel(SimpleAntiSpoofingModel):
    name = "my-detector"             # human label; the *slug* comes from meta.yaml
    expected_sample_rate = 16000     # the runner resamples to this for you
    batch_size = 1                   # raise it only if score_batch is overridden

    def load(self) -> None:
        # called ONCE before any scoring. Load weights here, not in __init__.
        self.net = _load_checkpoint("my_model.pt")

    def score(self, audio: np.ndarray, sr: int) -> float:
        # audio: float32, mono, already at expected_sample_rate. sr == expected_sample_rate.
        # RETURN: higher == more bona fide.
        logits = self.net(audio[None, :])
        return float(logits.softmax(-1)[0, BONAFIDE_INDEX])

    def unload(self) -> None:
        del self.net                 # called once at the end, even on error
```

### The three things that bite everyone

1. **Score direction.** Higher = more bona fide. Label 0 = bona fide, 1 = spoof. If your
   model outputs a *spoof* probability, return `-p` or `1 - p`. Getting this backwards
   gives you an EER of `~100 - true_eer` (e.g. 97% instead of 3%) — a dead giveaway.
2. **Load in `load()`, not `__init__()`.** The runner constructs the model, then calls
   `load()` once. Heavy work in `__init__` runs at import time and breaks
   `--model-module` discovery.
3. **Don't resample.** You're handed float32 mono at `expected_sample_rate`. Resampling
   again double-processes the audio and shifts your numbers.

### Batched inference (optional)

```python
from speech_spoof_bench.model import AntiSpoofingModel

class MyBatchedModel(AntiSpoofingModel):
    name = "my-detector"
    batch_size = 16

    def load(self): self.net = _load_checkpoint("my_model.pt")

    def score_batch(self, audios: list[np.ndarray], srs: list[int]) -> list[float]:
        # Must handle ANY k in 1..batch_size (the final short batch included).
        batch = _pad_stack(audios)
        return self.net(batch).softmax(-1)[:, BONAFIDE_INDEX].tolist()
```

If `score_batch` raises, the runner falls back to scoring those items **one at a time** —
so a batch-only bug can hide as a silent slowdown. Test with a real batch size.

## Step 2 — Run it locally (offline)

Register the dataset locally (see [setup.md](setup.md)), then:

```bash
speech-spoof-bench run \
  --model-module my_model:MyModel \
  --datasets SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA \
  --output-dir ./results
```

`--model-module` is `import.path:ClassName`. The file must be importable (on `PYTHONPATH`
or in CWD). Outputs:

- `results/<slug>/scores.txt` — `utt_id score` per line.
- `results/<slug>/result.yaml` — the metrics, `n_trials`, `n_skipped`, and `bench_version`.

Sanity checks:
- EER in a sane range (single-digit % for a good model; ~50% for random).
- `n_skipped` near 0. If >5% skip, the run **aborts** with `TooManySkips` — your
  `score()` is throwing on some inputs.

## Step 3 — Prove it reproduces (the step CI will run)

This is the gate that catches most "works on my machine" submissions. After you've
uploaded `scores.txt` to your model repo and authored the submission YAML (the
[submit-model.md](../submitting/submit-model.md) flow), verify it the way CI will:

```bash
speech-spoof-bench reproduce ./submission.yaml --scoring --no-local
```

`--no-local` forces it to use the canonical pinned HF revision (exactly what
`verify-hf-pr` does). It re-downloads your `scores.txt`, checks the SHA-256, streams the
labels, recomputes the EER, and compares to your claimed value within `1e-6`. If it
disagrees, **your PR will be rejected** — fix it now, locally.

Common reasons reproduce fails:

| Symptom | Cause |
|---------|-------|
| `scores_sha256 mismatch` | You edited `scores.txt` after computing the hash, or uploaded a different file. Re-upload and re-hash. |
| EER off by a lot | Score direction flipped, or you ran against a different dataset revision than the one pinned. |
| `coverage` / `n_trials` mismatch | `len(scores) + n_skipped != n_trials`; you scored a different set of utterances than the official split. Run with `--no-local` so you're on the pinned revision. |
| metric "not registered" | The dataset's `eval.yaml` lists a metric your installed package doesn't have. Update the package. |

## Step 4 — Decide on the paper

Per the project's paper policy: a **paper is what lets you compete in the ranked tiers**
(gold/silver/bronze). Proprietary or unpublished models are welcome but sit in the
unranked **`unpublished`** tier regardless of score. Set `system.paper` in `meta.yaml`
(`arxiv_id` is enough) if you have one; you can add it later to move into the ranked tiers.

## A complete worked example

```python
# rawnet2.py
import numpy as np, torch
from speech_spoof_bench.model import SimpleAntiSpoofingModel

class RawNet2(SimpleAntiSpoofingModel):
    name = "rawnet2-baseline"
    expected_sample_rate = 16000

    def load(self):
        self.net = torch.jit.load("rawnet2.ts").eval()

    @torch.no_grad()
    def score(self, audio: np.ndarray, sr: int) -> float:
        x = torch.from_numpy(audio).unsqueeze(0)          # (1, T)
        logit_bonafide = self.net(x)[0, 1]                # class 1 = bona fide here
        return float(logit_bonafide)
```

```yaml
# meta.yaml
system:
  name: "RawNet2 (baseline)"
  slug: "rawnet2-baseline"
  description: "RawNet2, ASVspoof2019 LA pretrained, FP32, no augmentation."
  code: "https://github.com/eurecom-asp/rawnet2-antispoofing"
  checkpoint: "https://huggingface.co/me/rawnet2/blob/main/rawnet2.ts"
  paper: {arxiv_id: "2011.01108", url: "https://arxiv.org/abs/2011.01108", bibtex: "@inproceedings{...}"}
  params_millions: 17.6
notes: "Default config, no TTA."
```

```bash
# offline iterate
speech-spoof-bench run --model-module rawnet2:RawNet2 \
  --datasets SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA --output-dir ./results
# upload scores.txt to your model repo (commit-pinned), then:
speech-spoof-bench submit --model-module rawnet2:RawNet2 \
  --datasets SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA \
  --model-repo me/rawnet2 --submission-meta meta.yaml \
  --hf-username me --contact me@example.com
# the PR is open; mirror CI before you walk away:
speech-spoof-bench reproduce ./results/ASVspoof2019_LA/submission.yaml --scoring --no-local
```

See [testing-and-pitfalls.md](testing-and-pitfalls.md) for the full failure catalogue.
</content>
