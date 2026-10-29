# Micro-Transformer Showcase

Three local applications of a real Python/PyTorch Micro-Transformer:

- **Ava:** chooses actions in a Python-owned Micro-World, including the audited browser extensions, journey memory, and colour keys/doors.
- **Chatbot:** answers short-story questions in a fixed controlled language. It is not an open-ended conversational assistant.
- **Device messaging:** interprets delivered and withheld updates between simulated devices.

Try a supplied trained model in **Playground**, modify its practice in **Training**, and inspect its architecture in the read-only **Model** view. Training uses real gradients; Playground uses the same selected learned weights without updating them. Incorrect answers and failed journeys remain visible. No inference interpreter, teacher correction or replacement policy supplies answers.

## Requirements

- Git and **Python 3.12**.
- Internet access for the initial framework clone and Python package installation. After setup, all runtime work is local; no model/API service, Node.js, frontend build or Browser Lab installation is needed.
- The verified clean-install platform is macOS arm64, Python 3.12, CPU execution. Other platforms and Python versions are not qualified by that check. The pinned macOS PyTorch/NumPy wheels require macOS 14 or newer.

This is a source-checkout application: run commands from this repository root. It is not currently distributed as a Python wheel.

## Clean setup

Clone this repository, then enter its root:

```sh
git clone https://github.com/vimalnar/micro-transformer-showcase.git
cd micro-transformer-showcase
```

The repository is private; cloning requires repository access. You can also copy an authorised source snapshot. The framework setup script clones the canonical framework into the adjacent `../micro-transformer` directory and checks out its exact pinned revision. If that directory already exists, it only verifies it; it never resets, edits or updates an existing checkout.

```sh
python3.12 scripts/setup_framework.py
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
.venv/bin/python scripts/verify_setup.py
.venv/bin/python -m showcase serve
```

Open [the local showcase](http://127.0.0.1:8765/). The server binds only to loopback. Stop with Ctrl-C; graceful shutdown pauses active training at a completed update boundary. An occupied port can be changed with `serve --port 8878`.

For a framework in another location:

```sh
python3.12 scripts/setup_framework.py --destination ../micro-transformer-pinned
export MICRO_TRANSFORMER_ROOT="$(pwd)/../micro-transformer-pinned"
```

Set that environment variable in every shell running the app or tests. A missing checkout, wrong revision or altered locked source/checkpoint fails visibly. See [setup and troubleshooting](docs/setup.md).

## Development and verification

Install the additional test client into the same fresh environment:

```sh
.venv/bin/python -m pip install -r requirements-test.txt
.venv/bin/python scripts/verify_setup.py --tests
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/review_end_to_end.py --report artifacts/my-local-review.json
```

The end-to-end review creates temporary state, runs an actual HTTP server, trains each model, restores downloaded checkpoints into a new process, compares exact continued weights and optimizer state, and verifies frozen Playground inference. It does not train your default saved models. Browser Lab and Node are optional for regenerating browser-reference fixtures; the checked-in fixtures are sufficient for the normal suite.

Exact direct dependencies are in `requirements.txt`; test extras are in `requirements-test.txt`; `constraints.txt` pins the verified transitive resolver snapshot. These are the showcase's tested dependencies, rather than the framework's older optional-training pins. See [clean-install evidence](docs/clean-install-2026-10-30.md) and the [three-showcase review](docs/end-to-end-review-2026-10-30.md).

## Standalone Python

```sh
# Chatbot inference and real fine-tuning
.venv/bin/python -m showcase predict 'hal move flag six to river. hal tell gia flag six at river. where gia knows flag six?'
.venv/bin/python -m showcase train --examples data/chatbot/locations.txt --rounds 2 --rate .0001

# Separate Device model and Python-owned message stream
.venv/bin/python -m showcase --showcase devices event river
.venv/bin/python -m showcase --showcase devices event dock --withheld
.venv/bin/python -m showcase --showcase devices train --examples data/devices/default.txt --rounds 2

# Ava: inspect without moving, execute one real action, or train
.venv/bin/python -m showcase.ava predict
.venv/bin/python -m showcase.ava step
.venv/bin/python -m showcase.ava train --epochs 1
```

Stop the server before state-changing CLI commands against its state directory. Use a separate directory for experiments: language CLI `--state PATH`, Ava CLI `--state PATH`, or server `--state PATH/chatbot --device-state PATH/devices --ava-state PATH/ava` (the last two flags follow `serve`). Only one writer process per state directory is supported.

## Models, checkpoints and practice

The canonical transformer is imported from a separate hash-checked framework checkout at revision `87cb7cd30695dd86cfb5586a41bd8951a2c9bef2`. There is no copied or silently diverging vanilla transformer. The framework's packaging omits its architecture file, so this application deliberately requires its locked source checkout.

**[Checkpoint provenance](docs/checkpoint-provenance.md)** and machine-readable **[checkpoints.json](checkpoints.json)** identify source weights, converted assets, practice and reference fixtures. Chatbot and Devices each start with the canonical 1,024,512-parameter language checkpoint. Ava starts with a faithfully converted 134,860-parameter browser-trained extension model. None of the supplied inference checkpoints includes resumable source optimizer state; new local training starts fresh optimizers and records local progress separately from historical training.

Saved models, optimizer state, drafts and simulations live in separate ignored directories: `state/chatbot`, `state/devices`, `state/ava`. Pause and reload restore completed progress. Download exports the actual selected learned checkpoint. To restore one, stop the server, back up the current file, put the trusted download at `current.pt` in its original showcase's state directory, and restart. Device simulation also uses `simulation.json`; Ava includes its world/history in its payload. Checkpoints reject foreign showcase identities. Atomic file replacement preserves complete saves; filesystem-fsync durability is not claimed.

Architecture editing stays disabled in HTML. Developers can inspect the Python source/configurations. Ava validates its extension implementation identities; changing those files requires a new state directory or an explicit compatible migration. See the application guides for the exact resume/reset rules:

- [Ava audit, world rules and training](docs/ava-audit-and-port.md)
- [Device messaging](docs/device-messaging.md)
- [Chatbot training and persistence](docs/setup.md#training-and-checkpoint-behaviour)

Practice loss and success on known rooms do not establish generalisation. This local engineering showcase makes no claim of broad language ability, general intelligence or consciousness.

## Repository map

| Path | Purpose |
| --- | --- |
| `showcase/core.py`, `server.py` | Canonical language adapter, real training, persistence and local HTTP service |
| `showcase/ava/` | Ordered Ava extension stages and authoritative Python world |
| `showcase/devices.py` | Authoritative delivered/withheld simulation |
| `web/` | Plain HTML/CSS/JavaScript; no browser model runtime |
| `data/`, `tests/fixtures/` | Supplied assets and small retained reference fixtures |
| `scripts/` | Verified setup, optional reference conversion and real-process review |
| `state/`, `artifacts/` | Ignored user models and local review output; not for sharing |

See [sharing notes](CONTRIBUTING.md). Code and supplied materials retain the repository's existing [Apache-2.0 license](LICENSE) and [NOTICE](NOTICE). The source is hosted on GitHub; the application runs locally. No hosted application or deployment is provided.
