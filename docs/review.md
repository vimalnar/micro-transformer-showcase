# Local development review — 29 October 2026

> Historical first-Chatbot review. See the [fresh-install review](clean-install-2026-10-30.md) for current setup qualification.

Three unittest workflows passed in the existing Python 3.12 / PyTorch 2.14.0 CPU environment:

1. Four canonical harness examples retain their expected predictions. Full prompt logits agree with a separately loaded canonical harness at absolute tolerance 1e-7 and relative tolerance 1e-6. Both documented baseline failures retain their original predictions. Inference state hash is unchanged.
2. A real gradient update changes learned state. Reload restores non-empty Adam state, recipe and example offset. The next update matches the uninterrupted model's complete state hash and loss exactly. HTTP Playground output and tensor hash match direct inference from the trained persisted checkpoint.
3. HTTP validates errors, rejects cross-origin mutations, starts real asynchronous training, pauses below the target, restores on application restart and resumes to the exact total. Prediction after training preserves weights. Checkpoint download succeeds.

Actual IAB → HTML → Python review used isolated `/tmp/micro-transformer-showcase-ui-review` storage, never Browser Lab storage. The initial communication prompt returned `river |` with zero local updates. Editing practice to `ava move key one to lab. where key one? => lab |` and training 20 updates produced tensor identity `c6be2d9823a750f5e56013fa9b677ee68b5596459158ca9b88440c8bba9bff58`; Playground returned `lab |`. Reload retained that identity and answer. Read-only Model showed the same identity and 20 updates. A numeric select-value mismatch found in the first browser attempt was fixed before this successful run.

No broad task evaluation, unseen-example generalisation, clean dependency installation, concurrent-writer support, device messaging port, Ava fidelity audit or Ava Python execution is claimed. UI pause/resume correctness is covered through HTTP integration; the browser review exercised edit/train/predict/reload/Model. CPU logit comparison is an integration regression, not an independent reimplementation of the transformer mathematics.
