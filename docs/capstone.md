# Capstone: ship a date normalizer under constraints

Choose a parser, TinyGPT, prompted model, LoRA model, or a justified hybrid.
You are delivering a decision and evidence, not a production web service.
You may complete this exercise after the core path using only the parser and
TinyGPT; the pretrained-model extension is not a prerequisite.

## Release brief

Accept the lab's supported date population under its DD/MM rule and produce
only valid `YYYY-MM-DD` answers.

Default teaching constraints:

- At least **95% exact match on each** of the three evaluation suites, not just
  on their pooled average.
- At least **99% valid-date outputs on each** suite.
- At most **1 second per item**, using measured batch-averaged prediction time
  on your declared device and batch size. This is not a p95 service-latency claim.
- At most **30 minutes total training time** on your declared hardware for the
  selected candidate's local task training/fine-tuning. This excludes the base
  model's original pretraining; list your selection experiments separately.

These are exercise targets, not claims that the supplied models meet them. An
instructor may change constraints **before** experiments, never after seeing final
scores. A no-ship conclusion is acceptable.

These thresholds apply to returned prediction strings, after the supplied
first-line/whitespace processing. They do not measure whether raw model output
contains extra lines. A strict raw-output contract needs an additional check.

## Develop, choose, then freeze

1. Use only training, validation, and development evidence to select your system.
2. Compare the selected candidate to the original parser baseline on full
   development suites. Record item counts, uncertainty, slices, cost, and seeds.
3. Copy [the decision template](decision-template.md) to
   `submissions/decision.md`, complete it, and freeze the exact code, prompt,
   model revision, data, and checkpoint/adapter.
4. Open final evaluation once for the frozen selection. Do not tune afterward
   and claim the same final holdout remains unseen.

### Restore full development results

Challenge 7's limited reports may have overwritten earlier full reports. Omit
`--limit` to restore full evaluation. For a TinyGPT candidate:

```bash
python step5_eval.py --dateutil --out runs/dateutil_full
python step5_eval.py --run runs/baseline
python step5_eval.py --compare runs/dateutil_full/report_dateutil.json runs/baseline/report_tinygpt.json
```

Replace `runs/baseline` with your selected run. For a prompting/adapter candidate,
run its corresponding evaluator without `--limit`, then compare that report
against `runs/dateutil_full/report_dateutil.json`. If you selected the original
parser itself, record that choice; there is no distinct candidate to compare.

### Pin the base model for adapter evaluation

An adapter is not standalone: it must be combined with its training base model.
Look up `base_model.name` and `base_model.revision` in
`runs/smollm/training_manifest_lora.json`. For the default SmolLM2 workflow, set
the following value to that **resolved commit hash**, not the moving name `main`:

```bash
BASE_REVISION="replace-with-the-commit-hash-from-your-training-manifest"
```

This assignment is a template: replace the quoted text before running the
adapter commands. If you trained a different model, also pass its exact
`--model_id`. For a local model with no remote revision, preserve the local base
files and use that same local path instead.

### Open final evaluation

Complete `submissions/decision.md` from the linked template first. Commands below
are alternatives: run the one for your selected system, not all of them.
Keep the same device, batch size, prompt, and other settings used for your frozen
development result; add their flags if they differ from defaults.

For TinyGPT:

```bash
python step5_eval.py --run runs/baseline --final --decision submissions/decision.md
```

For the parser:

```bash
python step5_eval.py --dateutil --final --decision submissions/decision.md
```

For a previously trained adapter (pin the base revision to your training record):

```bash
python step6_finetune_lora.py --mode lora --adapter runs/smollm/lora_adapter --revision "$BASE_REVISION" \
  --final --decision submissions/decision.md
```

Use the corresponding `--mode zeroshot` or `--mode fewshot` for those frozen
systems, pinning their model revision from the corresponding evaluation manifest
rather than an adapter training manifest. Do not use `--mode train` for final
evaluation. Do not use `--limit`.
A custom parser/hybrid must honor the same final protocol: call
`evaluation_plan`, `run_suite`, and `save_results`, and retain a provenance
manifest. The supplied `step5_eval.py` is the small integration example.

The final guard checks that a nonempty decision exists, not whether its reasoning
is complete or its declared configuration matches your implementation. Review
those manually. For strictly blind assessment, the instructor must host final
data outside the learner's workspace.

## Deliverable and assessment

Submit the frozen decision, relevant configs/manifests, development comparisons,
and final reports. Include an after-evaluation note without rewriting the frozen
decision. Explain whether release constraints were met and which claims remain
unsupported.

| Criterion | Evidence |
|---|---|
| Correct specification | Unambiguous dates, output contract, supported scope |
| Fair investigation | Fixed budgets, matched items, multiple seeds where relevant |
| Sound choice | Quality/cost/maintenance tradeoff against a non-ML baseline |
| Evaluation integrity | Separate development/final use, frozen artifact identity |
| Honest limitations | Failure slices, uncertainty, no tuning disguised as final testing |

After inspecting final failures, those items become development evidence.
Further iteration needs a fresh instructor-held holdout before another final
release claim.
