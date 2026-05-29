# Submit a model

1. **Wrap your model** as an `AntiSpoofingModel` subclass (`load`, `score`/`score_batch`, `unload`). Higher score = more bonafide.
2. **Install:** `pip install speech-spoof-bench`.
3. **Run the benchmark:**
   ```bash
   speech-spoof-bench run --model-module mypkg.mymod:MyModel --datasets all
   ```
   This writes `results/<dataset>/scores.txt` + `result.yaml`.
4. **Upload `scores.txt`** to your HF model repo under
   `.eval_results/<dataset-org>/<dataset-name>/scores.txt`.
5. **Author `meta.yaml`** describing your system, including the optional
   `system.params_millions` (your model's parameter count, in millions — used by
   the "By model size" tab).
6. **Submit** (runs + uploads + opens the PR):
   ```bash
   speech-spoof-bench submit \
     --model-module mypkg.mymod:MyModel --datasets all \
     --model-repo <you>/<repo> --submission-meta meta.yaml \
     --hf-username <you> --contact <you@example.com>
   ```
7. **Verification:** a maintainer runs `reproduce --scoring` (fast; mandatory)
   and optionally `--inference` (full re-run; upgrades the ★ badge), then merges.
8. **After merge:** paste the badge snippet from the post-merge comment into your
   model README to link back to the Arena.

> Can't run the full benchmark yourself? See the note at the top of the Submit tab.
