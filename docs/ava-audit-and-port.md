# Ava audit and Python port — 30 October 2026

The active Ava showcase is now Python-backed. Playground opens first, Training edits worlds and selected practice, and Model is read-only. Everything runs locally. This is an application engineering review, not a Research Lab experiment or a claim about cognition or consciousness.

## Audit findings and scope

The executing Browser Lab worker imports `CausalMapAgent` from `causal-map-agent.js`. The older `map-agent-model.js` is a historical different model and was **not** used as the port target.

The active source has eight capability types and an expanded key/door world. Older six-block documentation omits Feature refinement and Attention perspectives, and predates the expanded-world rules. The audit followed executable source, configuration validation, the supplied checkpoint, worker training, and the actual world functions. Exact reference file hashes are recorded in `data/ava/provenance.json`, alongside the Browser Lab revision. The canonical framework revision and source hashes remain in `framework-lock.json`.

No active capability was substituted with a policy, dropped, or reinterpreted as a consciousness mechanism. Legacy teaching experiments (experts, repeated processing boards and their different models) are not the active Ava capability kit and are not imported. No browser videos are reused. The UI is a compact Python-backed rendering of the active navigation workflow, rather than a port of every historical comparison/teaching screen.

## Architecture and executable equivalents

The PyTorch adapter imports `ModelConfig`, `Transformer`, `CausalSelfAttention`, `FeedForward` and canonical embedding/normalization components from the hash-checked canonical `models/transformer.py`. It adds an ordered stage scaffold because vanilla `Transformer` does not support arbitrary ordering, cross-attention memory, object bias or workspace stages. It does not maintain a copied vanilla transformer. A depth-only instance is tested directly against canonical `Transformer` using the same complete state dictionary.

| Active capability | Faithful Python operation |
| --- | --- |
| Processing depth, 1/2/3/5 | Independently weighted canonical causal attention and GELU feed-forward pairs; pre-LayerNorm, residuals, tied token output. |
| Context lens, 8/16/32/128 | Additional causal attention; a query reads itself and at most window−1 earlier keys. This is local to the stage, not a smaller learned position table. |
| Object focus, lamp/Ava/both | Per-head trainable additive key-logit bias. The 25-token adapter identifies coordinate token ranges; the 104-token adapter identifies coordinate positions, exactly as the browser does. No object detector or route hint. |
| Short-term memory, 2/4/8 | Single-head learned Q/K/V/output cross-attention over the latest executed pose/action records. Shared token/position embeddings; padding masked; an entirely empty bank contributes exactly zero. |
| Journey memory, 4/8/16 | Same cross-attention, reading earliest distinct pose/facing records. Expanded records also distinguish carried keys. Repeated poses do not replace the archive. |
| Shared workspace, 4/8/16 | Causal prefix mean, learned compress projection, tanh, learned broadcast projection and residual. A functional extension; no claim of global-workspace-theory implementation. |
| Feature refinement, width multiplier 1/2/4, GELU/ReLU | Extra per-token canonical feed-forward projections at the selected stack position, with the selected activation. |
| Attention perspectives, 1/2/4 heads | Additional full causal attention stage; head count splits the same token width and does not change the other stages. |

Stack order, normalization, residuals, tensor shapes and original browser state-dictionary names are retained by the import mapping. Configuration validation rejects duplicate capabilities, invalid settings, empty stacks and more than 16 stages. Browser Mulberry32 and Box–Muller initialization are reproduced for new/reset models. Developers can supply a configuration file to the Python CLI with a new state directory; no architecture editing is offered in HTML.

The supplied model has width 64, one depth layer and context, object, recent, archive and workspace extensions: **134,860 trainable parameters**, not the 1,024,512-parameter vanilla language baseline. Its six blocks compile into seven executable stages. The other two capability types are implemented and tested in developer configurations, rather than silently added to this checkpoint's architecture.

## Task encoding and action selection

Small room: 70 features → 25 tokens. Row-major 4-bit wall chunks, Ava column/row/facing, lamp column/row and final action query. Expanded room: 520 features → 104 tokens, including 64 wall nibbles, up to ten object triplets, carried-key state and the final action query. Object slots use token zero; memory padding uses input-only token 128. Both use the original learned position table of 128.

Recent/history records are earlier pose/facing plus the **action actually executed**. Small records use four tokens; expanded records add inventory as a fifth. Recent/archive capacities therefore have fixed masked banks up to 32/64 small-room tokens or 40/80 expanded tokens. Teacher routes, current labels, distance and future observations never enter inference memory. Retry clears history; it is not persistent learning or hidden state.

The model predicts all 128 vocabulary scores. Training uses last-query cross entropy over all 128. Playground softmax/argmax uses only action token IDs **96, 97, 98**, mapped to forward, turn-left and turn-right. There is no world-dependent legal-move mask, planner rescue, teacher correction or scripted movement.

## World rules retained in Python

`showcase/ava/world.py` owns authoritative state, observation, transition, edit and encoding rules:

- Small editable area at coordinates 4–11 inside the same 16×16 tile array; expanded editable area at 1–14 with fixed outside walls.
- Forward collisions, left/right turns, supported simulator wait, ticks and events. Wait remains a simulator action, not a fourth model output.
- Day/dusk/night/dawn cycle over 100 ticks; visibility ranges, 9×9 egocentric view, Bresenham occlusion, lamp lighting and locked-door occlusion. The model still reads the full map; local vision is displayed for comparison.
- One lamp, one Ava, fixed walls; at most one key per colour and eight doors. Movement automatically collects reusable red/blue keys. Matching doors are passable, wrong-colour doors block. Keys are not consumed. Collected keys appear as floor without deleting the underlying layout object. These intentionally match the **browser extension**, not canonical explicit pick-up/unlock interaction actions.
- Forward/turn training labels come from reverse BFS over pose and, in expanded rooms, available inventory. Tie-breaking uses the first optimal action in forward/left/right order. Labels are used only for practice and reachability diagnostics.
- Editing within one encoding keeps weights, clears the current journey and room qualification, and allows testing the same weights in the changed room. Training on that changed room reinitializes. Switching small/expanded encoding immediately reinitializes, matching the browser boundary.
- Another start uses the original deterministic trial selection. Retry restores the selected initial agent and empty history. Memory models stop after eight occurrences of a resulting pose or 160 actions (small)/512 (expanded). Stateless policies stop on a repeated pose. A successful arrival is handled separately from these stopping rules.

## Practice, training and room checks

Selected rooms combine the edited layout, two alternative reachable lamp placements, open/detour obstacles, zigzag routes and (expanded only) red/blue lock scenarios. Worlds are deduplicated by exact identity. All reachable poses supply examples. With memory, the same eight deterministic teacher trajectories per room add prefix-history examples, stopping at 40 teaching steps. Histories contain only preceding actions. Expanded teacher states include acquired-key states; room checks start without gifted keys.

Training uses AdamW with betas `(0.9,0.999)`, epsilon `1e-8`, decoupled decay `.01`, batch size 32, norm-one clipping with the original `1e-6` denominator convention, 20-step linear warm-up and the original seeded Fisher–Yates ordering. Every tensor is trainable. Actual loss/update/round progress is reported. Practice move agreement uses the original restricted three-action probabilities; it is not held-out accuracy.

At each tenth or final round, room checks run **every reachable start**, with independently evolving memory per start. Stateless policies may cache model predictions by full pose/inventory; memory policies do not. Check progress is real, pause is available between batches/steps, and interrupted unfinished journeys remain unresolved. The user can also request an independent check without training. No sampled check is presented as full coverage.

Changing situations, learning rate or seed resets weights, like the browser configuration flow. Completing a requested round target and starting again also reinitializes. Raising a target on an incomplete compatible recipe continues. Editing only the round target preserves current weights and progress. These reset rules are visible in Training.

## Checkpoint provenance and persistence

Source Browser Lab `public/trained-example.json` SHA-256:

`8bbcac2e4bd756e87a6e7de63c0628c45c3d013a591a95e233a592e2966e0f3b`

Converted tensor payload `data/ava/base.pt` SHA-256:

`9a71b73a78696e848ad1ebe444b06823332b430472e8d45a15d854f14cb44281`

The original has 50 historical training rounds and 1,400 historical updates, but **no optimizer payload**. Conversion retains weights without retraining; it does not fabricate resumable browser moments or count historical updates as new Python updates. Python training begins at round/update zero with a fresh AdamW optimizer. Historical browser room evidence is labelled separately until a new Python check is run.

`state/ava/current.pt` saves the exact current PyTorch tensors, AdamW state, partial-round offset/loss, settings, selected practice, world/layout, trajectory, memory, decisions, qualification, source identities and implementation hashes. It is atomically replaced after every completed training batch, on Pause, on user world/play operations and at round/check boundaries. Process restart restores actual optimizer and world state. World actions save the model but do not train it.

Chatbot, Devices and Ava have separate directories, model instances and showcase identifiers. Loading a foreign payload or changed Ava implementation/base identity fails explicitly. This source binding is deliberate: after changing Python implementation, use a fresh state directory or prepare an explicit compatible migration; do not silently relabel old evidence. Only one writer process per directory is supported. Checkpoints use atomic replace rather than filesystem-fsync durability guarantees.

The locked source checkpoint stays in the new showcase as a small converted PyTorch asset; the original Browser Lab source and browser storage are never written. No dependency installation, publishing, GitHub push or deployment occurred.

## Verification

The full local suite contains 17 passing test workflows, including existing Chatbot and Device regressions. Ava tests compare:

- Seven room fixtures, canonical exact encoding, all teacher label/optimal-action/cost identities, trial selection, shuffle order and all generated history examples.
- Full combinations of small/expanded training situations against executable Browser Lab output.
- Day/dusk/night/dawn observations, world steps/events, inventory transitions, wrong-key blocking, impossible key cycles and editing boundaries.
- Twenty-eight browser/Python forward cases across seven assemblies: the supplied learned checkpoint, all eight capabilities, reordered stages, multiple depth/window/focus/head/FF/memory/workspace settings, both encodings and empty/nonempty memory. Maximum observed absolute logit difference: `2.86102294921875e-6`; tolerances `atol=1e-5, rtol=1e-5` (probabilities use `atol=2e-6`). Stage-state comparisons also pass.
- Exact canonical depth-only forward agreement; causal invariance, correct history selection and exactly zero contribution from empty memory.
- Full-model gradient agreement (`atol=1e-6, rtol=1e-4`) and one actual browser/Python clipped AdamW update (`atol=3e-6, rtol=1e-4`), with learned changes in every stage. The first diagnostic reference incorrectly used TensorFlow's default epsilon; it was corrected to Ava's actual `1e-8` before qualification.
- Actual weight changes, exact next-update/loss equality after Python optimizer restoration, world-edit retention, encoding-change reinitialization, inference immutability and cross-showcase isolation.
- HTTP training/pause/persistence/download/Playground semantics through the actual service.

### Actual HTML review

The browser review used isolated `/tmp/mt-ava-review-20261030/` state on port 8878. With untouched supplied weights (`0bbf4797c51af175…`), Ava reached the displayed lamp in 15 actual model-selected actions. An independent Python full-room check measured **205/216** arrivals, reproducing the supplied browser record.

A one-round `layout + lamps` training review paused at **128/896 examples, four optimizer updates**. Reload retained that offset, settings, weights and optimizer. Resume completed **28 updates** and the requested one round. Actual final-batch loss was `0.011078039184212685`; practice move agreement was about **98.8%**. Rechecked room outcome: **215/216**, with the remaining failure retained. Learned tensor identity:

`037dfac55e9202a33c651b6d027a43f1cc729697e7193bc19172e81b0a404d2c`

Playground used that exact learned identity for its next action. A world edit retained it. Switching to the expanded key/door layout reset weights and local update count as specified. Model inspection remained read-only. A screenshot and small review JSON are kept under ignored `artifacts/`; review training did not replace default user models.

These observations establish bounded implementation parity and a working local flow. They do not establish unseen-map transfer, memory-dependent competence, universal room success, cognition or consciousness. No runtime backend is TensorFlow; JavaScript/TensorFlow is used only by the optional read-only reference-fixture exporter.

## Commands

From the showcase repository root:

```sh
PY=.venv/bin/python
$PY -m showcase serve
# Open http://127.0.0.1:8765/ava.html

$PY -m showcase.ava info
$PY -m showcase.ava predict       # inspect current state, no world move or weight change
$PY -m showcase.ava step          # real prediction and Python world transition
$PY -m showcase.ava retry
$PY -m showcase.ava examples
$PY -m showcase.ava train --epochs 1
$PY -m showcase.ava check
$PY -m unittest discover -s tests -v
```

Stop the server before state-changing CLI commands against its directory. Use `-m showcase.ava --state /tmp/my-ava-version …` for an independent CLI version. A developer can supply `--config path/to/compiled-capabilities.json` with a **new** state directory; saved checkpoints reject a different supplied architecture. Both compact capabilities and matching executable blocks are supported through `compile_config`.

For isolated web work:

```sh
$PY -m showcase --state /tmp/ava-review/chatbot serve \
  --device-state /tmp/ava-review/devices --ava-state /tmp/ava-review/ava --port 8878
```

Normal state directories are siblings `state/chatbot`, `state/devices`, `state/ava`. Explicit `--state` keeps its existing Chatbot meaning; otherwise extra showcase directories default below that custom path unless separately specified. The new Ava CLI is intentionally separate from the language-model CLI because its actions, data and objective differ.

Reference regeneration (optional; requires the unchanged Browser Lab checkout and its installed TensorFlow.js):

```sh
$PY scripts/import_ava_checkpoint.py
node scripts/export_ava_reference.mjs
```

`BROWSER_LAB_ROOT` overrides the reference checkout location. Import regenerates the supplied conversion/provenance, not a user checkpoint. Neither script writes to the Browser Lab. Serving, training and inference require no Node or TensorFlow dependency. The test fixture is gzip-compressed; no large raw tensor dumps or training intermediates are kept in the project.
