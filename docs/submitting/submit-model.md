# Submit a model

This guide walks you through evaluating an anti-spoofing model and getting it onto
the Arena leaderboard. The whole flow is: **wrap → run → upload scores → open a PR →
maintainer verifies → merged**.

> 💡 **Can't run the full benchmark yourself?** If you can run your model over the
> complete dataset(s), email **[k.n.borodin@mtuci.ru](mailto:k.n.borodin@mtuci.ru)**
> and we'll consider running it for you.

## Prerequisites

- Python 3.10+ and a Hugging Face account.
- Your model's inference code and checkpoint (weights) reachable from Python.
- A public HF **model repo** you own (it will host your `scores.txt`).

## 1. Wrap your model

Subclass `AntiSpoofingModel`. The contract: audio arrives as float32 mono at 16 kHz,
and **a higher score must mean "more bonafide"** (less likely spoofed).

```python
from speech_spoof_bench.model import AntiSpoofingModel
import numpy as np

class MyModel(AntiSpoofingModel):
    name = "my-model"

    def load(self):                       # called once before scoring
        self.net = load_my_checkpoint()
    def score(self, audio: np.ndarray, sr: int) -> float:
        return float(self.net(audio))     # higher = more bonafide
    def unload(self):                     # called once after scoring
        del self.net
```

## 2. Install the toolkit

```bash
pip install speech-spoof-bench
```

## 3. Run the benchmark

```bash
speech-spoof-bench run --model-module my_pkg.my_module:MyModel --datasets all
```

For each dataset this writes `results/<dataset>/scores.txt` (one `utterance_id score`
per line) and a `result.yaml` with the computed EER. Use `--datasets <id>` to target a
single dataset instead of `all`.

## 4. Upload your scores to your model repo

Upload each `scores.txt` to your HF model repo under this exact path:

```
.eval_results/<dataset-org>/<dataset-name>/scores.txt
```

e.g. `.eval_results/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA/scores.txt`. The
submission pins your scores by commit sha, so they stay immutable after merge.

## 5. Describe your system in `meta.yaml`

```yaml
system:
  name: My Model
  slug: my-model                 # lowercase, hyphenated; unique on the board
  description: One-liner — architecture, config, precision.
  code: https://github.com/you/my-model
  checkpoint: https://huggingface.co/you/my-model
  params_millions: 52.3          # optional; powers the "By model size" tab
  paper:
    arxiv_id: "2110.01200"
    url: https://arxiv.org/abs/2110.01200
    bibtex: |
      @inproceedings{your2022paper, ... }
```

## 6. Open the submission PR

```bash
speech-spoof-bench submit \
  --model-module my_pkg.my_module:MyModel --datasets all \
  --model-repo you/my-model --submission-meta meta.yaml \
  --hf-username you --contact you@example.com
```

This re-runs, uploads the scores, builds `submissions/<slug>.yaml`, and opens a pull
request on the dataset repo — you never edit the dataset repo by hand.

## 7. Maintainer verification

A maintainer runs `reproduce --scoring`: it fetches your `scores.txt`, re-checks the
sha, and recomputes EER (must match within 1e-6). That earns the **✔ scoring** badge
and the PR is merged. Optionally they re-run your model end-to-end
(`reproduce --inference`), upgrading you to the **★ inference** badge.

## 8. Show the badge

After merge, CI posts a ready-to-paste `result.yaml` and a shields badge snippet.
Paste the badge into your model README to link back to your Arena row.

---

Questions or problems? Email **[k.n.borodin@mtuci.ru](mailto:k.n.borodin@mtuci.ru)**.
