# Working on the local showcase

Use the [clean setup](README.md#clean-setup), keep the canonical framework in a separate pinned checkout, and install `requirements-test.txt` into a fresh environment. Serving, training and ordinary tests do not require Browser Lab or Node.

Before proposing changes:

```sh
.venv/bin/python -m pip check
.venv/bin/python scripts/verify_setup.py --tests
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/review_end_to_end.py --report artifacts/review.json
```

For frontend edits, also check JavaScript syntax and the real HTML workflow. Node is optional for these developer checks:

```sh
node --check web/app.js
node --check web/ava.js
```

Use separate state directories for probes. Preserve real failures and distinguish practice results from generalisation. Keep learned weights, architecture, teaching data and application state distinct. Playground must use the selected learned checkpoint, not teacher labels or a substitute policy.

## Dependencies

`requirements.txt` declares the runtime; `requirements-test.txt` adds the test client. `constraints.txt` pins the verified transitive versions. To change them, resolve in a new Python 3.12 environment, run `pip check`, the full suite and real-process restoration review, then record the actual platform and results. Do not silently equate a changed dependency stack with earlier numerical evidence.

## Model and data changes

Keep `framework-lock.json` unchanged unless intentionally selecting and qualifying another canonical revision; local learned payloads bind to it. Do not copy a private fork of the vanilla transformer into this repository. Ava architecture editing stays disabled in HTML; developer configuration changes require new compatible weights and provenance.

Review changes to `checkpoints.json`, bundled assets and Ava provenance together. Do not update expected hashes merely to hide failed integrity checks. Original browser checkpoint and working-tree provenance must not be represented as files available from the base Git commit.

## Before sharing on GitHub

- Include code, web assets, supplied `data/` and `tests/fixtures/`, requirements/constraints, license/notice and documentation.
- Exclude user `state/`, local `artifacts/`, virtual environments, caches and `.env` files. Review the file list before staging.
- Supplied `data/ava/base.pt` and compressed fixture files are intentional versioned assets. They are small enough for ordinary Git.
- Do not include local review screenshots, downloaded learned checkpoints or developer-specific paths in the source snapshot.
- Check setup from a fresh copy and a separate clean pinned framework checkout. Describe the platform actually tested, rather than claiming unverified cross-platform support.

Source changes can be committed and pushed when authorised. Releases, public-visibility changes and application deployment remain separate decisions; pushing this repository does not deploy it.
