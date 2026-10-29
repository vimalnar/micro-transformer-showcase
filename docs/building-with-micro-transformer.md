# Building with Micro-Transformer

The showcases are applications of a small transformer, with their application
state kept outside the learned model. This guide follows that boundary so you
can remove the browser, inspect the Python, and build something of your own.

## Run the examples

Complete the [repository setup](../README.md#clean-setup), then run these commands
from the repository root. No browser or server is needed:

```sh
.venv/bin/python -m examples.toolkit_inference
.venv/bin/python -m examples.chatbot
.venv/bin/python -m examples.devices
.venv/bin/python -m examples.ava
```

The last three use a temporary state directory, removed when the example exits.
They start from the supplied weights and leave your saved Playground models alone.
To exercise genuine gradient training, checkpoint restoration and frozen inference:

```sh
.venv/bin/python -m examples.chatbot --train
.venv/bin/python -m examples.devices --train
.venv/bin/python -m examples.ava --train
```

Chatbot trains one example; Devices trains eight missing-update examples; Ava
trains one epoch of its selected practice and runs the application's normal checks.
These are demonstrations of the mechanics, not generalisation experiments. Raw
predictions remain visible even when wrong; examples do not assert correct answers.
They do assert actual weight changes during training, exact restored model and
optimizer values, and no weight changes during prediction or world actions.

To retain an example's checkpoint, add a separate path, for example
`--state state/examples/chatbot`. Use different paths for each showcase and one
writer per directory. The Devices example clears its chosen stream before sending
its two events; Ava retries its journey. Use temporary defaults if you want to
preserve an existing stream or journey. Repeated `--train` calls continue learning;
the language examples start another practice round, while Ava extends the epoch
target. Ava's existing supported epoch maximum still applies.

To inspect retained example weights through the HTML interface, stop the example
and start the corresponding server configuration:

```sh
# Chatbot
.venv/bin/python -m showcase --state state/examples/chatbot serve
# Devices (run separately)
.venv/bin/python -m showcase serve --device-state state/examples/devices
# Ava (run separately)
.venv/bin/python -m showcase serve --ava-state state/examples/ava
```

## What belongs where

| Layer | Canonical toolkit | Showcase application |
| --- | --- | --- |
| Architecture | `models/transformer.py`: configuration, Transformer, attention and feed-forward components | Ava's ordered extension scaffold; no copied vanilla Transformer |
| Language encoding | `training/data.py`: fixed vocabulary and encoding | Story examples; device-event-to-story adapter |
| Language inference | `training/harness.py`: checkpoint loading and token generation | Sessions, input validation, selected learned-state identity |
| Learning | PyTorch model supplied by the toolkit; toolkit also has its own standalone trainer | Answer-only language fine-tuning and Ava's action training loops |
| Saved state | Supplied canonical language checkpoint | Separate checkpoint envelopes with model, optimizer, progress and provenance; Ava also includes world/history |
| Application state | Toolkit has separate Micro-World facilities | Python device stream and faithful port of the browser's Ava world |
| Interface | No requirement for a browser | FastAPI plus plain HTML/JavaScript |

The source checkout is locked by [framework-lock.json](../framework-lock.json) to
revision `87cb7cd30695dd86cfb5586a41bd8951a2c9bef2`. On import,
[`showcase/core.py`](../showcase/core.py) checks revision and source/checkpoint hashes
before importing the toolkit. The toolkit packaging currently omits the architecture
file, so this is source-checkout integration rather than a standalone pip SDK.
See [checkpoint provenance](checkpoint-provenance.md) for weight and data identities.

## Start with the toolkit alone

[`examples/toolkit_inference.py`](../examples/toolkit_inference.py) makes the central
operation explicit:

```python
from showcase.core import ARCH, BASE  # Verified checkout paths and import bootstrap
from training.harness import InferenceHarness

harness = InferenceHarness(BASE, device='cpu', architecture=ARCH)
result = harness.predict('gia move lamp twelve to dock. where lamp twelve?')
```

`harness`, its model, tokenisation and generation come from Micro-Transformer.
`ARCH` identifies architecture code; `BASE` identifies learned weights. They are
different artifacts. No simulation or HTTP service is involved here.

For a project based only on the toolkit repository, run from that checkout and
replace `ARCH` and `BASE` with its `models/transformer.py` and
`baselines/v1.0.0/checkpoints/seed42-inference.pt` paths. The toolkit's
[training guide](https://github.com/vimalnar/micro-transformer/blob/87cb7cd30695dd86cfb5586a41bd8951a2c9bef2/training/README.md)
covers preparing data, training from scratch, inference and evaluation. The showcase
examples instead begin with supplied trained weights and demonstrate fine-tuning.

## Chatbot: model to application

Read [`examples/chatbot.py`](../examples/chatbot.py), then
[`Session`](../showcase/core.py). The web path is:

`web/app.js → showcase/server.py → Session.predict → toolkit InferenceHarness → Transformer`

The training path is `Session.start → Session.run → Session.update`. An example
such as `story question => answer |` is encoded using the toolkit vocabulary.
The full sequence enters the causal model; cross-entropy supervises only the
answer and end token. Backpropagation, norm clipping and Adam update the actual
model held by the inference harness. There is no second inference model to copy
weights into. `save()` records those weights and the optimizer; constructing a
new session with the same directory restores them.

The example shows the higher-level session because it is the exact path used by
HTML Training and Playground. To inspect the learning mechanics, `Session.update`
is the small implementation to follow. Its checkpoint envelope is showcase-specific;
do not pass `current.pt` directly to the toolkit harness as if it were a canonical
inference checkpoint. Restore it through its matching session.

This is a bounded story-question model with a fixed 128-token vocabulary and
128-position context. Arbitrary English prompts are not supported. Better practice
scores do not imply broad conversational ability.

## Devices: add an event adapter

Read [`examples/devices.py`](../examples/devices.py) and
[`showcase/devices.py`](../showcase/devices.py). The model is the same canonical
language architecture and starting checkpoint as Chatbot, with separate learned
weights, optimizer and state directory.

An event moving the flag to river encodes as `hal move flag six to river.`.
Delivery adds `hal tell gia flag six at river.`; withholding omits that sentence.
The final question is `where gia knows flag six?`. Python stores the stream, builds
this prompt and calls the transformer. It calculates the expected report only
after generation for assessment. The expected answer does not choose output tokens.

The example delivers river, then withholds dock. Python knows the actual location
is dock and the last delivered location is river; the printed transformer report
shows what the learned model actually says. Up to six events are explicitly
included in context. This is not a network service or hidden recurrent memory.

To build a related application, start with explicit event/state definitions and
an encoding compatible with the model. Keep your simulation transitions separate
from model prediction. Prepare real labelled practice and independent evaluation;
do not replace a wrong prediction with the simulator's expected answer.

## Ava: what the showcase adds

Read [`examples/ava.py`](../examples/ava.py), then
[`model.py`](../showcase/ava/model.py), [`world.py`](../showcase/ava/world.py) and
[`session.py`](../showcase/ava/session.py). Ava is not the vanilla language model
with a navigation prompt. It has a browser-trained extension architecture and a
different checkpoint.

`AvaModel` loads canonical components through the toolkit's `load_architecture`.
It reuses canonical embeddings, final normalisation, attention projections and
feed-forward components. The showcase adds an ordered stage scaffold supporting:

- Processing depth and a limited context lens.
- Object-focused attention and configurable attention heads.
- Recent and archive journey memory with cross-attention.
- Workspace compression/broadcast and optional feature refinement.

All eight capability types are supported. The supplied checkpoint uses its original
six-capability configuration; the optional capabilities are not silently inserted
into its learned architecture. Its 134,860 parameters and converted tensor identities
are distinct from the language model's 1,024,512 parameters.

Python encodes the current world into tokens and supplies recorded journey history
to the model. The small encoding has 25 tokens; the expanded encoding has 104.
The model receives the full encoded map, not only the displayed local vision.
It produces vocabulary logits; Playground chooses among action IDs 96–98 for
forward, left and right. Python then applies the world transition and records the
actual action. There is no planner correction or world-dependent legal-action mask.

World rules include lighting/occlusion, walls, reusable coloured keys and matching
doors. Automatic key collection and door passage match the browser extension;
they are not the canonical Micro-World's explicit pick-up/unlock interaction API.
The breadth-first teacher creates supervised training labels and reachability
checks; it does not choose Playground actions. Training uses full-vocabulary
action cross-entropy and AdamW, with the existing batching, clipping and warmup.

`AvaSession` keeps the world, history, weights and optimizer together and preserves
the existing reset rules. For example, changing encoding resets the model;
changing practice can also reset it. The example keeps the world, rate, seed and
selection unchanged. See the [audit](ava-audit-and-port.md) for the complete rules,
browser parity checks and limitations.

To reproduce full Ava, use these extension/world modules as well as the toolkit.
They have not been moved into the canonical toolkit. For a new application, first
ask whether the standard Transformer is sufficient; these extensions are not a
requirement for using Micro-Transformer.

## Build something different

Keep architecture, learned weights, training data and application rules explicit.
Define what the model predicts, encode only the information it should receive,
and decide how its output affects the application. Reuse the toolkit model and
harness where suitable, then train and evaluate the new task. A new vocabulary or
architecture needs compatible weights and training; an existing checkpoint does
not automatically acquire a new domain because an interface was added.

The runnable examples verify plumbing and restoration. Existing parity tests and
the [end-to-end review](end-to-end-review-2026-10-30.md) provide broader evidence.
Neither is evidence of general intelligence, consciousness or success on arbitrary
unseen tasks.
