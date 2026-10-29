# Setup, commands and troubleshooting

Run these commands from the showcase repository root. Use Python 3.12 and Git. The verified clean-install profile is macOS arm64 with CPU execution; other platforms are not qualified by this local review.

## Framework checkout

```sh
python3.12 scripts/setup_framework.py
```

The default source is `https://github.com/vimalnar/micro-transformer.git`. The script creates the adjacent `../micro-transformer` only when it does not already exist, checks out revision `87cb7cd30695dd86cfb5586a41bd8951a2c9bef2`, and verifies the source/checkpoint hashes in `framework-lock.json`.

An existing directory is verified, never reset or switched to another revision. If it does not match, choose a new location:

```sh
python3.12 scripts/setup_framework.py --destination ../micro-transformer-pinned
export MICRO_TRANSFORMER_ROOT="$(pwd)/../micro-transformer-pinned"
```

For offline setup from an already available Git repository, pass `--source PATH/TO/REPOSITORY`. The exact revision and file checks still apply. This copies a checkout of the commit, not uncommitted source edits. Application startup never clones or fetches anything.

## Fresh Python environment

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
.venv/bin/python scripts/verify_setup.py
.venv/bin/python -m showcase serve
```

The `venv` command does not use `--system-site-packages`; packages from a previous environment are not inherited. Runtime requirements and test extras share the exact transitive pins in `constraints.txt`. Install test extras only when needed:

```sh
.venv/bin/python -m pip install -r requirements-test.txt
.venv/bin/python scripts/verify_setup.py --tests
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/review_end_to_end.py --report artifacts/review.json
```

The ordinary tests use supplied reference fixtures and the canonical framework checkout. Browser Lab and Node are not required. Optional fixture regeneration requires the separately audited Browser Lab working-tree snapshot, not merely its historical Git commit.

## Startup and isolated state

Open `http://127.0.0.1:8765/`. Use `serve --port 8878` if the port is occupied. The server binds to loopback and rejects cross-origin API calls. It does not contact a hosted model service.

Default state lives under `state/chatbot`, `state/devices` and `state/ava`. For an independent server:

```sh
.venv/bin/python -m showcase --state state/review/chatbot serve \
  --device-state state/review/devices --ava-state state/review/ava --port 8878
```

Stop with Ctrl-C. Training pauses after completed updates and graceful shutdown waits for the worker to save. Only one server/CLI writer process should use a state directory at a time. Do not run training CLI commands against an active server's directory.

## Training and checkpoint behaviour

Chatbot and Devices fine-tune separate full canonical models with answer/end-token cross entropy and clipped Adam updates. Each update saves weights, optimizer, exact recipe, offset, RNG and loss. An unchanged incomplete recipe resumes; a changed recipe starts a fresh Adam optimizer from the current weights. A completed recipe repeats from its current weights with a new optimizer. Each prompt is independent; no interpreter chooses Playground answers.

Ava uses full-vocabulary action cross entropy, clipped AdamW, warm-up and supervised world/history examples. Its reset boundaries differ deliberately: changing situations, seed or rate resets weights; training on an edited room resets; changing small/expanded encoding resets immediately. Same-encoding world edits retain weights for Playground testing. See the [Ava audit](ava-audit-and-port.md) for all rules and their source parity checks.

Download saves the actual selected model. To restore a trusted download:

1. Stop the server or CLI writer.
2. Back up the destination's existing `current.pt` and any associated simulation file.
3. Place the download at `current.pt` in its original showcase's state directory.
4. For Devices, also restore `simulation.json` if preserving live events/reports is required; its weights/draft are in the checkpoint. Ava includes its world and history in its checkpoint.
5. Restart and compare the read-only Model/Playground learned-state hash with the expected saved identity.

Cross-showcase payloads are rejected. Ava additionally checks its implementation source hashes: after changing its Python model/world/session code, use a fresh state directory or implement an explicit compatible migration. Do not disable that check to force incompatible saved evidence to load.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| Framework missing | Run `scripts/setup_framework.py`, or set `MICRO_TRANSFORMER_ROOT` to a verified checkout. |
| Framework revision/hash mismatch | Keep existing work intact and create a separate pinned checkout with `--destination`. Do not reset the reference project to make setup pass. |
| No matching wheel / Python mismatch | Use Python 3.12. The qualified macOS arm64 wheels require macOS 14+. Other platforms need their own validation before being claimed supported. |
| Dependency mismatch | Create a fresh environment; install the checked-in requirements/constraints; run `pip check`. Do not modify a working framework environment to satisfy this app. |
| Supplied checkpoint/data hash mismatch | Reacquire the unchanged supplied artifact from the repository and verify `checkpoints.json`. Do not rewrite expected hashes to hide damage. |
| Port already in use | Select another port; do not stop an unrelated service. |
| Unknown controlled-language token / too much context | Use the supplied vocabulary and shorter independent stories. This is a bounded model. |
| Non-finite logits or checkpoint tensors | Treat this as a numerical failure. No generated answer is substituted; preserve the error and last valid checkpoint for investigation. |
| Saved report/answer marked stale | Training changed the selected weights; ask or reinterpret again. The old raw output is retained as a previous-checkpoint result. |

Review artifacts and screenshots are local ignored outputs. Historical reports retain their original environments and results; the new [clean-install report](clean-install-2026-10-30.md) records the independently installed environment.
