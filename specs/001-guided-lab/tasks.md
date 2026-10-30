# Implementation tasks

Check a box only after verification. Dependencies are explicit.

- [x] T1: Define requirements and the small spec workflow. (R1-R7)
- [x] T2: Generate isolated development/final populations and evaluation files
  independent of training augmentation. Depends on T1. (R3)
- [x] T3: Add final opt-in and strict paired comparisons. Depends on T2. (R3, R4)
- [x] T4: Record training/evaluation provenance, including final decision and
  prompt/model identity. Depends on T3. (R5)
- [x] T5: Add an explicit broken-causality diagnostic mode. Depends on T1. (R2)
- [x] T6: Publish navigation, challenges, reference, solutions, and capstone.
  Depends on T2-T5. (R1, R2, R6)
- [x] T7: Update artifact ignores and add focused tests with `make check`.
  Depends on T2-T5. (R7)
- [x] T8: Run automated checks and offline CLI smoke experiments; record evidence
  and confirm generated artifacts stay ignored. Depends on T6, T7. (R1-R7)
- [x] T9: Proofread as a beginner: add prerequisites/definitions, control the
  causal-mask experiment, align source explanations and artifact descriptions,
  and clarify full-suite/pinned-model capstone steps. Depends on T6, T8.
  (R1, R2, R6)

## Verification

- `make check PYTHON=.venv/bin/python`: 16 tests passed.
- Dataset checks: disjoint date populations, unchanged evaluation files under
  augmentation, zero leakage counts, invalid-size rejection.
- Protocol/comparison checks: development defaults, missing-data migration,
  final decision/full-suite requirements, strict pairing, legacy warnings,
  decoding mismatch, and statistical regression exit status.
- Training checks: shifted answer masking, padding, causal-prefix invariance,
  intentionally broken noncausal attention, and manifest hashes.
- Offline CLI smoke checks: parser development/final evaluation; two-step TinyGPT
  training, checkpoint loading, self-comparison gate, final evaluation without
  overwriting development reports, and diagnostic-option persistence.
- Offline LoRA smoke check: locally initialized tiny Llama/tokenizer; zero-shot,
  few-shot, two training steps, adapter save/reload, and frozen-adapter final
  evaluation. This checks mechanics, not real SmolLM2 quality.
- `git diff --check`: clean. Artifact ignore rules verified for existing data,
  checkpoints, virtual environment, and learner submissions.

Existing local datasets and runs were not regenerated or modified. Their old
split protocol must not be mixed with new benchmark results.

### Beginner proofreading pass

- Added prerequisites, a core/extension route, essential terminology, and an
  artifact-reading guide.
- Corrected the causal-mask exercise to use a matched full-data control, and
  explained validation subsets, step-zero loss, and training versus data seeds.
- Clarified processed-output scoring, system-specific batching, actual LoRA
  artifacts, full-development reruns, and adapter base-revision pinning.
- Corrected explanatory source comments/docstrings; runtime behavior is unchanged
  by this proofreading pass.
- Re-ran all 16 checks successfully. Verified local links/anchors across nine
  Markdown files and shell syntax for 18 command blocks. Executed the token trace
  and augmentation-hash examples on temporary data; the report-reading example
  parses and uses fields exercised by the existing suite.
- Full-length learning experiments and real SmolLM2 quality were not re-measured;
  historical results remain explicitly illustrative.
