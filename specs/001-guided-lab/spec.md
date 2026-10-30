# 001: A hypothesis-driven LLM learning lab

## Purpose

Turn the date-normalization walkthrough into a small, inspectable course teaching
training mechanics, trustworthy evaluation, and engineering judgment. Keep
CPU-sized experiments and the existing model-agnostic scoring harness.

## Requirements and acceptance criteria

### R1: Explicit learning scope

Distinguish supervised training from random weights from general-purpose
pretraining. TinyGPT versus SmolLM2 compares systems, not just pretraining:
size, tokenizer, architecture, and instruction tuning also differ.

Acceptance: README maps the lifecycle and links guided exercises, a conceptual
reference, separate solutions, and a decision-making capstone.
Provide prerequisites, a core-versus-extension route, and definitions for terms
before relying on them in beginner exercises.

### R2: Progressive investigation

Cover non-ML baselines, token/target tracing, deliberately broken causality,
generalization predictions, fixed-budget data diversity, three-seed comparisons,
and prompting versus LoRA. Each challenge asks for a prediction, action,
observable evidence, and explanation; worked conclusions are separate.

Acceptance: runnable commands match the scripts, including a named diagnostic
option for removing the causal mask.
The causal-mask exercise includes a matched control; it must not confound
causality with one-batch versus full-data training.

### R3: Development versus final evaluation

Default evaluation uses `dev`, `dev_ood_format`, and `dev_ood_year`. Retain
`test`, `test_ood_format`, and `test_ood_year` for a final evaluation after
freezing a decision. Train, validation, development, and final date populations
are disjoint. ID/OOD-format pairs within a suite intentionally share dates.

Acceptance: final evaluation requires `--final --decision PATH`, a nonempty
decision document, and full suites (no `--limit`). Old datasets without dev
files fail with a migration message, never a fallback to test files.
Training augmentation leaves all evaluation files unchanged.

Decision: the final boundary is procedural and CLI-enforced, not access control.
Files are locally visible; do not claim reruns are impossible.

### R4: Fair paired comparisons

Reject different suite membership, item IDs, inputs, labels, or format annotations,
duplicate IDs, and empty comparisons. Compare matching decoding policies where
manifests exist; non-ML parsing has no decoding policy.

Acceptance: partial overlaps cannot silently produce a paired score. Legacy
reports remain comparable when their actual items match, with a provenance warning.
The existing statistical regression gate remains available.

### R5: Reproducible records

Training/evaluation records include configuration, seed where applicable,
dataset hashes, source identity, dependency versions, hardware/device, and
elapsed time. Evaluations identify checkpoints/adapters and snapshot the final
decision. SmolLM2 records its system prompt, chat template, and model revision.

Acceptance: TinyGPT training, TinyGPT evaluation, dateutil evaluation, and
SmolLM2 training/evaluation emit adjacent JSON manifests. Existing summary report
shape remains compatible and reports include actual item counts.

### R6: Durable material

README is navigation and quickstart. Separate guided exercises, reference,
illustrative historical results/solutions, and a capstone with submission template.

Acceptance: explain validation's distribution limits, dateutil's parsing-policy
mismatch, item uncertainty versus seed variation, and the final protocol.
The capstone has measurable quality/resource constraints and requires evidence,
limitations, and a justified system choice.
Explain how to restore full development reports after limited runs, preserve
the adapter's base revision, and distinguish processed predictions from raw output.

### R7: Small maintenance harness

Use standard-library `unittest` with `make check`, without new dependencies.
Ignore generated data, runs, local submissions, secrets, caches, and OS metadata
without hiding specifications.

Acceptance: focused tests cover isolation, final opt-in, pairing, provenance,
and diagnostic training. Offline smoke runs cover data generation, baseline
evaluation, TinyGPT training/evaluation, and the final gate.

## Non-goals

No broad-corpus pretraining, preference optimization/RLHF, production service,
automatic model download during checks, heavyweight spec tooling, or claims
about real SmolLM2 quality without running it.
