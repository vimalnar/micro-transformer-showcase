# Python-backed Device messaging

Implemented locally on 30 October 2026. Open `/devices.html` from the showcase selector. The independent Ava port and its verification are documented in [the Ava audit](ava-audit-and-port.md).

## Meaning and model boundary

Python owns the bounded event history, actual asset position, delivery decisions, comparison state and report records. `showcase/devices.py` ports Browser Lab's `device-location-messages-v1` adapter:

- Tracker HAL moves asset `flag six` to one of river, lab, store, home, bridge, room, forest or dock.
- Every event encodes `hal move flag six to PLACE.`
- Only delivered events append `hal tell gia flag six at PLACE.`
- Python concatenates the complete stream and asks `where gia knows flag six?`
- The independent expected hub state is the last delivered place, or `unknown` when no message was delivered. It is assessed **after** generation; it never enters the inference call as an answer.
- Raw generated text, IDs, stop reason, update count and learned-state SHA-256 remain inspectable. A valid report is exactly `PLACE |` or `unknown |`; wrong/unsupported replies are retained, not corrected.

The full stream is supplied each time. This adds no recurrent model memory, network stack, physical device control or new architecture. Six delivered events use 96 input tokens within context 128. A seventh event is rejected before changing the stream. Clear removes only simulation events/reports and retains learned weights. Interpret same messages reuses the stream without appending an event. Inference failure retains the committed event and an explicit error; retry does not duplicate it.

## Learned version and training

Devices and Chatbot use separate Python model/Adam instances and showcase-tagged payloads; cross-showcase checkpoint loading is rejected. Both initially read the same immutable canonical base checkpoint, without copying its weights asset. Devices retain the exact architecture and existing answer-only cross-entropy, full-model Adam, norm-one clipping and supported learning rates. Training and simulation inference share the same selected Device model and lock. No teacher correction or rule-generated replacement report exists.

Default learned state: `state/devices/current.pt`. Default Python simulation: `state/devices/simulation.json`. Practice files: `data/devices/`. The exact editable recipe is saved in the training payload alongside optimizer, offset, loss, RNG and framework identities. Simulation persistence is separate from learned checkpoints. Checkpoint download exports learned state and practice, not the live event stream; back up both files to preserve the full application session. Old report identities are compared to current learned weights and shown as stale after training until explicitly reinterpreted.

Delivered updates, Missing updates and Mixed sequences each provide eight editable examples. Python generation of these starter sets exactly matches the original JS adapter. All 24 labels are independently checked against the pinned canonical interpreter in tests. The interpreter is not called by the runtime adapter or inference. Edited labels receive structural validation only. Practice loss is not generalisation evidence.

Web training supports actual progress, pause/resume after reload or restart, draft autosave and downloads. Changed recipes restart Adam from current weights; unchanged incomplete recipes resume saved Adam and offset. The application is a single local shared workspace: tabs see the same Python simulation, rather than owning conflicting browser copies. Use separate server/state directories for independent visitors or development probes. Only one writer process per state directory is supported.

## Local commands

From the showcase repository root:

```sh
PY=.venv/bin/python
$PY -m showcase serve
# Open http://127.0.0.1:8765/devices.html

$PY -m showcase --showcase devices event river
$PY -m showcase --showcase devices event dock --withheld
$PY -m showcase --showcase devices interpret
$PY -m showcase --showcase devices clear-stream
$PY -m showcase --showcase devices predict 'hal move flag six to dock. where gia knows flag six?'
$PY -m showcase --showcase devices train --examples data/devices/default.txt --rounds 2 --rate .0001
$PY -m showcase --showcase devices train  # resume the saved recipe
$PY -m showcase --showcase devices info
```

Stop the server before running state-changing CLI commands against its state directory. For isolated CLI work, add `--state /tmp/my-device-version` **before** the subcommand. Serve always exposes both showcases; `--showcase` selects non-server CLI operations. For an isolated server use:

```sh
$PY -m showcase --state /tmp/my-showcase/chatbot serve --device-state /tmp/my-showcase/devices --port 8877
```

With `--state` alone, that path retains its original Chatbot meaning and server Device state defaults to its `devices/` child. `--device-state` explicitly overrides this; normal startup keeps sibling `state/chatbot/` and `state/devices/` directories. The existing `SHOWCASE_STATE` environment override still affects Chatbot only. Server binds only to loopback and rejects cross-origin API calls.

## Interface

The existing Chatbot `/api/*` endpoints retain their meaning. Device counterparts use `/api/devices/{status,predict,train,pause,draft,practice,checkpoint}`. Device simulation adds GET `simulation` and POST `events` (`place`, boolean `delivered`), `interpret` and `clear`. JavaScript sends only user actions; the server creates authoritative events, language packets, comparisons and report records. It never trusts a browser-supplied expected answer for Playground inference.

## Verification and evidence limits

Eight tests pass (five Device workflows plus three existing Chatbot regressions). They check:

- Exact Browser adapter parity, canonical validation of 24 labels, supported locations and six-event/context limits.
- Baseline token predictions and last-token logits against retained Browser Lab canonical-Python fixtures at `atol=1e-5, rtol=1e-5`; interpreter fixture hash matches the framework lock.
- Delivered, withheld and initial unknown behaviour through the real model, with unchanged weights. Diagnostic wrong or malformed outputs remain wrong/invalid; the simulator does not replace them.
- Failed inference retains the event and exposes a structured HTTP error. A seventh event leaves the stream/report unchanged. Clear preserves weights. Restart reloads simulation.
- Device weights really change; restored Adam reproduces the next full-state hash and loss exactly. Cross-scope checkpoints are rejected and Chatbot saved bytes remain unchanged.
- HTTP training pauses, resumes after server reconstruction, uses its learned checkpoint for reports, marks older reports stale and downloads actual checkpoint bytes. Chatbot draft, weights and recipe stay unchanged.

Actual browser review used isolated `/tmp/mt-device-review-20261030/` state on port 8877. Initial delivered river → `river |`; subsequent dock withheld → `river |` while actual position showed dock. Forty real updates on two edited examples completed. A subsequent 160-update Missing updates run paused at 13/160, retained that offset and draft after reload, then resumed to 160/160 (200 cumulative updates). Reinterpreting the existing stream with those learned weights again returned `river |`. These are development/practice checks, not a held-out evaluation. The review checkpoint and screenshot are separate from default user state.

The framework and original Browser Lab were read only. No browser storage was accessed or replaced. No videos reused, no publishing, GitHub push or deployment. Source and fixture hashes are recorded in `device-port-provenance.json`; the canonical framework/checkpoint lock is unchanged so existing Chatbot learned payloads remain compatible.
