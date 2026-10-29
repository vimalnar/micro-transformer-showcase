# Three-showcase end-to-end review — 30 October 2026

> Historical review. A subsequent [fresh-install review](clean-install-2026-10-30.md) closes the clean-install gap noted below; other platforms remain unqualified.

**Passed locally:** all 20 automated workflows, a separate real-server checkpoint/restart review, and actual HTML training/Playground review of Chatbot, Device messaging and Ava. No publishing, deployment or GitHub push.

## Fixes

- **Chatbot stale answers:** an answer generated before a weight update previously remained labelled as current. It now shows its original prompt and is explicitly marked as a previous-checkpoint answer until asked again. Raw predictions are retained, not replaced.
- **Invalid numerical output:** a diagnostic language model with a NaN weight previously generated `........` through argmax instead of reporting a numerical failure. The adapter now rejects non-finite logits before decoding, and rejects saved non-finite model/optimizer tensors without replacing the checkpoint. Device inference failures still retain their event and expose the error.
- **Asynchronous UI state:** older status/simulation responses cannot overwrite newer interaction state. Ava settings saves and action/training requests are queued so fast edits are not silently dropped or raced against Train.
- **Service restart feedback:** pages explicitly show a disconnected Python service and clear that transport warning when reconnection succeeds. Actual model/input errors remain visible. A live Ava page recovered without reloading after its server was stopped and restarted.

The canonical transformer, Ava architecture/world mathematics, checkpoint formats, supported practice objectives and reset rules were not changed. Existing checkpoints remain compatible; Ava's implementation identity was not changed by these adapter/UI fixes.

## Real-process checkpoint verification

`scripts/review_end_to_end.py` launches the actual local HTTP service in a subprocess with fresh temporary state. It trains and pauses each real model, downloads its checkpoint, stops the process, then starts a **new process using new directories populated only from those downloads** (plus the separate Device simulation file).

For each showcase it verifies restored tensor identity, update count, offset and non-empty Adam state. It then compares resumed HTTP training with a separately restored Python continuation: final learned-state hashes and every optimizer tensor/value must match exactly. Playground output must match that independently restored model, and inference must leave weights unchanged.

| Showcase | Pause point in process review | Final real updates | Observed Playground result | Exact resumed weights + optimizer | Frozen inference |
| --- | --- | --- | --- | --- | --- |
| Chatbot | 8 / 40 updates | 40 | `lab \|` on its practised question | Pass | Pass |
| Device messaging | 13 / 40 updates | 40 | `river \|` after delivered river, withheld dock | Pass | Pass |
| Ava | 608 / 896 examples, 19 updates | 28 (one round) | `forward` from the selected start | Pass | Pass |

Learned-state SHA-256 identities:

```text
Chatbot 900bff35e5eef93ddf6ed1df5416e1fab4ebeb75218c3e748bacc15a716ee78c
Devices bb1b02c94d5cb7129945c6563b4f6c7fb9f91cf08b3a75202ede453007b6952c
Ava     037dfac55e9202a33c651b6d027a43f1cc729697e7193bc19172e81b0a404d2c
```

The machine-readable result is retained at `artifacts/end-to-end-review-2026-10-30/process-restoration.json`. Temporary process-review checkpoints are removed by the script after verification; no large intermediate histories are created.

## Actual browser workflow

The HTML review used isolated state under `/tmp/mt-e2e-ui-20261030-01/`, served on loopback port 8879. It did not train the default saved user models or use Browser Lab storage.

- **Chatbot:** baseline `unknown |` on `ava move key one to lab. where key one?`; edit examples; actual training; pause/reload; resume to 40 updates; Playground returns `lab |`. The old baseline answer was visibly marked stale after training. Model and Playground showed the same learned identity.
- **Device messaging:** delivered river then withheld dock, with Python showing actual dock and model report `river |`; edit Missing updates practice; pause at 6/40; reload and resume; reinterpret the preserved stream using the final learned checkpoint. Old reports were correctly marked stale before reinterpretation. Model matched Playground's learned identity.
- **Ava:** actual baseline model action; edit round target and queue its settings save; train; pause at 160/896 examples, five actual updates. Stop and restart the server while the page remains open: it displayed the disconnection, reconnected, and retained the exact five-update tensor identity, offset and settings. Resume to one completed round and 28 updates; Playground executes an action using the final learned identity; read-only Model agrees. The actual room check remains **215/216**, retaining the unsuccessful start.

All three final browser-trained tensor hashes exactly matched the independent process review's final hashes above, despite different pause points. This is an additional check that training, restoration and Playground are connected to the same learned versions.

Browser evidence:

- `artifacts/end-to-end-review-2026-10-30/chatbot.png`
- `artifacts/end-to-end-review-2026-10-30/devices.png`
- `artifacts/end-to-end-review-2026-10-30/ava.png`
- `artifacts/end-to-end-review-2026-10-30/browser-before-restart.json`
- `artifacts/end-to-end-review-2026-10-30/browser-final-states.json`

## Regression checks and limits

The 20 passing workflows include canonical baseline and known-failure predictions; browser/Python Ava parity (maximum logit difference `2.86102294921875e-6`); real gradient updates; exact optimizer restoration; model/data separation; delivered/withheld and unknown message behaviour; world edits and inventory rules; numerical failures; and actual HTTP routes. Both JavaScript modules pass syntax checks.

These are implementation and practice checks, not a protected evaluation. Chatbot's `lab |` is a practised answer; Device messaging is a bounded simulated protocol; Ava's 215/216 is one known room with practice exposure. No claim of unseen-map transfer, broad language ability, general intelligence or consciousness follows. Clean dependency installation and other operating systems remain untested; the verified environment is the existing Python 3.12 / PyTorch 2.14.0 CPU environment. One writer process per state directory remains the supported operating model.

## Reproduce locally

From the showcase repository root:

```sh
PY=.venv/bin/python
$PY -m unittest discover -s tests -v
$PY scripts/review_end_to_end.py --report artifacts/my-local-review.json
$PY -m showcase serve
```

The review script chooses an unused loopback port and uses only temporary state. Normal startup remains at `http://127.0.0.1:8765/`. Architecture editing remains disabled in the interface. Existing default state-file identities are recorded and compared before and after this review.
