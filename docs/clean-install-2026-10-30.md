# Clean install and local GitHub preparation — 30 October 2026

**Verified:** a fresh test environment and a separate runtime-only environment, a stripped source copy, and an independently cloned canonical framework. All work remains local. No GitHub repository, release, remote for this showcase, push or deployment was created.

## Environment and isolation

| Item | Actual verification |
| --- | --- |
| Host | macOS 26.6.2, arm64 |
| Python | CPython 3.12.14 |
| Package isolation | Two newly created virtual environments; `include-system-site-packages = false` |
| Test profile | `requirements-test.txt`, with the exact `constraints.txt` resolver snapshot |
| Runtime profile | `requirements.txt` only; test HTTP client `httpx` confirmed absent |
| Application source | Separate copy excluding `.git`, environments, user state, artifacts and caches |
| Framework source | Fresh clone from `https://github.com/vimalnar/micro-transformer.git`, detached at `87cb7cd30695dd86cfb5586a41bd8951a2c9bef2` |
| Existing projects | The original framework and Browser Lab remained unchanged |

The Python interpreter installation was already available on this computer; the virtual environments and their packages were new. This was not a new operating-system installation or a cross-platform certification. Package-wheel downloads may use pip's cache; no packages were inherited from the existing development environments.

Both environment profiles installed successfully, `pip check` reported no broken requirements, and setup verification passed. The supplied macOS arm64 PyTorch/NumPy wheel tags require macOS 14+. Other operating systems, architectures and Python versions remain unqualified.

## What was verified

1. **Setup script:** a new pinned framework clone was created by `scripts/setup_framework.py`, then the same target was verified idempotently. An existing directory with a user-file sentinel was rejected without changing or cloning over it.
2. **Integrity:** `scripts/verify_setup.py` checked six locked framework files and all ten bundled checkpoint/practice/fixture files. A deliberately missing supplied checkpoint in the isolated copy was rejected, then restored unchanged.
3. **Dependencies:** clean install of runtime and test profiles, exact transitive constraints, and `pip check`. The supported legacy HTTPX test-client path emits a non-fatal Starlette deprecation warning; assertions are not suppressed. The production profile does not install the test client.
4. **Full suite:** all **20 workflows passed** using the clean source copy, clean framework and fresh test environment. No Browser Lab checkout or Node runtime was needed. Ava browser/Python maximum absolute logit difference remained `2.86102294921875e-6`.
5. **Real server:** with the runtime-only environment and stripped copy, all three models completed genuine training, checkpoint download, restoration into a new server process and resumed training. Final learned weights and all optimizer values exactly matched independent restored continuations. Playground matched the selected learned versions and left weights unchanged.
6. **Portable source:** developer-specific home paths were removed from runtime/test tooling and setup commands. Optional direct browser-source tests require explicit `BROWSER_LAB_ROOT`; ordinary tests use retained fixtures.
7. **Git boundaries:** source, documentation, supplied `data/ava/base.pt` and compressed fixtures are included. User `state/`, `artifacts/`, `.venv*/`, caches and `.env` files are ignored. Tensor/fixture files are marked binary. The local showcase repository uses branch `main`, has no commits or configured remote, and has not been pushed.

## Scheduling issue found during installation verification

The first real-server review failed its partial-pause prerequisite: the short training run had completed before the test could capture an incomplete checkpoint. No restoration success was claimed for that attempt.

The language worker now yields its time slice between completed real updates (`time.sleep(0)`). This allows status/pause handlers to run without inserting simulated updates or changing the model, objective, data order, optimizer or checkpoint format. The full suite and real-server check were rerun successfully after this change. The first failure classification is retained in the ignored local evidence directory.

## Reproducibility result

The clean installation produced exactly the learned tensor identities from the preceding development review, despite different pause points:

| Showcase | Final real updates | Learned-state SHA-256 |
| --- | --- | --- |
| Chatbot | 40 | `900bff35e5eef93ddf6ed1df5416e1fab4ebeb75218c3e748bacc15a716ee78c` |
| Device messaging | 40 | `bb1b02c94d5cb7129945c6563b4f6c7fb9f91cf08b3a75202ede453007b6952c` |
| Ava | 28 | `037dfac55e9202a33c651b6d027a43f1cc729697e7193bc19172e81b0a404d2c` |

These are engineering/practice checks, not evidence of generalisation, broad conversational ability, intelligence or consciousness.

## Sharing and provenance preparation

- `README.md` now starts with clean setup, supported environment, normal local startup and standalone commands.
- `docs/setup.md` covers existing-checkout protection, alternate framework paths, isolated state, checkpoint restoration and troubleshooting.
- `checkpoints.json` catalogs exact source/converted checkpoint identities and hashes every bundled asset. `docs/checkpoint-provenance.md` explains optimizer availability, historical versus local training, practice provenance and sharing boundaries.
- `constraints.txt` records all resolved runtime/test transitive pins; direct runtime requirements and test extras remain separate.
- Browser Lab's Git base commit is explicitly distinguished from its audited working-tree files. That commit does not contain the current browser application or original Ava JSON; file-level SHA-256 records identify the actual snapshot. Runtime and normal tests require neither that checkout nor an unavailable regeneration artifact.
- `LICENSE`, upstream attribution in `NOTICE`, `.gitattributes`, `.gitignore`, `.python-version` and `CONTRIBUTING.md` are ready for review before an eventual commit/push.

The existing `framework-lock.json`, supplied checkpoint bytes and Ava implementation identities were preserved, keeping previously saved compatible models valid.

## Local evidence

Detailed output is retained under ignored `artifacts/clean-install-2026-10-30/`, including:

- installation logs and complete installed-package lists;
- clean setup/runtime verification JSON and framework bootstrap log;
- `clean-copy-tests-final.log` with all 20 workflows passing;
- `clean-process-restoration.json` and log with exact resumed-model/optimizer checks;
- `first-attempt-result.json`, recording the partial-pause timing failure before the scheduling fix;
- `summary.json` and a source-file manifest.

Do not include those local logs, environments or user-trained review models in a public source snapshot. The retained test fixtures and supplied checkpoint assets remain part of the repository. The original [three-showcase review](end-to-end-review-2026-10-30.md) remains a historical record of its own environment and observations.
