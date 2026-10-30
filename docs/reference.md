# Reference

[Entry point](../README.md) | [Guided lab](lab.md)

## Essential vocabulary

| Term | Meaning in this lab |
|---|---|
| Language model (here) | An autoregressive model that predicts a distribution over the next token, given earlier tokens |
| Token / tokenizer | A unit of text / the mapping between text and integer IDs; TinyGPT uses characters |
| Tensor | A multidimensional array of numbers, such as a batch of token IDs |
| Parameters / weights | Learned numbers that determine the model's predictions |
| Training / inference | Updating weights from examples / using weights to generate answers |
| Label / ground truth | The intended answer used to train or score a prediction |
| Loss / cross-entropy | A penalty for assigning low probability to the correct next token; lower is better |
| Gradient / optimizer | How loss changes with each parameter / the rule that uses gradients to update parameters |
| Batch / step / epoch | A group of examples / one optimizer update / one pass through the training dataset |
| Learning rate | The scale of an optimizer's parameter updates |
| Checkpoint | Saved model weights and configuration |
| ID / OOD | In-distribution / out-of-distribution relative to training data |
| Exact match (EM) | Fraction of predictions that match their complete reference answers |
| EOS / padding | End-of-sequence token / filler tokens used to align batch lengths |
| Teacher forcing | Predicting training targets with the correct earlier tokens supplied as input |
| Autoregressive generation | Predicting a token, appending it, then predicting the next token using that generated prefix |
| SFT / LoRA | Supervised fine-tuning on labeled pairs / low-rank adaptation, a parameter-efficient way to fine-tune |

**The core loop:** encode examples -> predict next-token probabilities -> measure
loss -> compute gradients -> update weights. During evaluation, generate an
answer without supplying the correct answer, then compare it to ground truth.
An attention mask controls which input positions a token can use; a loss mask
controls which prediction targets contribute to the loss. They are different.

## Task and split contract

Slash dates mean DD/MM/YYYY; outputs must be valid calendar dates in
`YYYY-MM-DD`. This labeling policy is a design decision, not a universal date rule.

Default generation produces:

| File stem | Rows | Use |
|---|---:|---|
| `train` | 20,000 | Seven formats, years 1950-2039 |
| `val` | 500 | TinyGPT checkpoint selection; LoRA loss inspection |
| `dev` | 1,000 | Repeated in-distribution development evaluation |
| `dev_ood_format` | 500 | Held-out `03-Mar-2024` format |
| `dev_ood_year` | 500 | Unseen years 2040-2099 |
| `test` | 1,000 | Final in-distribution evaluation |
| `test_ood_format` | 500 | Final unseen-format evaluation |
| `test_ood_year` | 500 | Final unseen-year evaluation |
| `fewshot` | 7 | Examples from the training date population |

Split by the underlying **date**, not by rendered string. Otherwise a differently
formatted training date could disclose a test answer. Train, validation,
development, and final populations are disjoint, including their future-year
populations. Within development or final, ID and OOD-format examples intentionally
share dates: these measure different renderings, not independent samples.
Few-shot examples come only from the training population, never held-out dates.
They are rendered from that date population; they are not necessarily rows
already present in `train.jsonl`. "Unseen format/year" refers to this lab's
training data, not to everything a pretrained model may have seen previously.

`--extra_formats` adds four training formats at the same row budget. Independent
randomness keeps all evaluation files unchanged. `data_card.json` records sizes,
protocol version, format mix, and leakage checks; nonzero leakage aborts generation.
Zero overlap does not establish that a synthetic benchmark represents real users.
The generated inputs are valid dates in predefined English-language formats;
invalid dates, arbitrary prose, other languages, and two-digit years are not
covered by this evaluation.

`val` and `dev*` are both held out from weight updates. This lab separates their
roles: `val` selects checkpoints, whereas `dev*` supports experiment decisions.
The word "development" is not a universally distinct split in other projects.

## Tokens and the objective

See [tokenizer](../step2_tokenizer.py) and [training tensors](../step4_train.py).
The fixed character vocabulary includes padding, letters, digits, punctuation,
the separator `=`, newline EOS, and an unknown-character token.

```text
sequence: input=answer\n
x:        sequence[:-1]
y:        sequence[1:]
loss:     mean(-log p(y[t] | x[:t+1])) over scored targets
```

With masking, targets predicting prompt characters are `-100` and ignored.
The input still contains the entire prompt. The position containing `=` predicts
the first answer character; the last answer position predicts EOS. Padding
targets are always ignored.

A uniform predictor has loss `ln(vocabulary size)`. Low loss measures next-token
probability under teacher forcing, not automatically successful full-answer
generation. Unmasked and masked training losses average over different targets;
validation uses answer-only targets for both.
Here `ln` is the natural logarithm; a random initialization should be near, not
necessarily exactly at, the uniform-prediction loss.

For TinyGPT, the shifted tensors are built explicitly. Hugging Face's causal-LM
loss shifts labels internally, so the LoRA script supplies unshifted labels.
Both train the same next-token prediction relationship; do not shift twice.

Broad-corpus pretraining uses next-token prediction on raw text. Part 1 instead
learns a narrow supervised task from random weights; removing prompt masking does
not turn this small date dataset into general-purpose pretraining.

## Model and training

See [TinyGPT](../step3_model.py):

```text
token embedding + learned positional embedding
  -> 4 x [residual causal attention + residual MLP, with pre-LayerNorm]
  -> LayerNorm -> tied embedding/output projection
```

Defaults: four heads, width 128, context length 48, about 0.8M parameters.
Causality prevents position t from observing tokens after t. The
`--diagnostic_no_causal_mask` training option deliberately violates this rule
to illustrate teacher-forced leakage. Its low loss is not valid evidence of
autoregressive competence.

Training uses AdamW, warmup/cosine learning-rate decay, and gradient clipping.
First overfit one batch to check that the loop can learn. TinyGPT saves the best
validation-loss checkpoint and the final checkpoint; `--ckpt last` evaluates the
latter. LoRA currently saves the last training step, not a best-validation model.

TinyGPT samples batches with replacement for a fixed number of steps: 3,000
updates are not 3,000 epochs. LoRA shuffles the selected rows and visits them once
per epoch. TinyGPT's logged training loss is a moving average, validation loss
uses the full validation set, and validation EM uses at most its first 200 items.
LoRA's before/after validation loss also uses at most the first 200 items.

Generation is greedy for comparison. TinyGPT groups equal-length prompts;
SmolLM2 uses left padding for generation and right padding for training.
LoRA freezes base weights and learns small low-rank updates on attention
projections. Chat templates and the DD/MM system instruction are part of the
model-facing specification, not incidental formatting.
Greedy decoding chooses the highest-scoring next token rather than sampling.
It removes decoding randomness, not all possible variation across hardware and
software environments.

## Evaluation and uncertainty

See [harness](../evals.py) and [CLI](../step5_eval.py).

| Measurement | What it tells you |
|---|---|
| Exact match | Entire answer correct after trimming surrounding whitespace |
| Wilson 95% interval | Approximate item-sampling uncertainty under binomial assumptions |
| Year/month/day accuracy | Which output field failed |
| Valid-date rate | Whether an ISO-shaped answer is a possible calendar date |
| Per-format slices | Failures hidden by an aggregate |
| Error taxonomy | Malformed, impossible date, swapped day/month, wrong field, multiple fields |
| Prediction seconds/item | Batch-averaged inference cost, excluding model loading/training |

The harness scores the strings returned by each system. Model helpers keep only
the first generated line; scoring trims surrounding whitespace. A correct scored
answer therefore does not prove that the unprocessed model output obeyed a
strict "date only, no extra lines" contract.

Slices and component accuracies overlap: they are diagnostic views, not
independent experiments. Wilson intervals do not capture training-seed variation,
benchmark bias, population drift, or synthetic-data limitations. For an
intervention, report results across at least three fixed training seeds.
A confidence interval describes uncertainty in an estimated success rate; it is
not the model's confidence in an individual answer.

McNemar's paired exact test uses only discordant items: A correct/B wrong versus
B correct/A wrong. The comparison CLI requires identical suites, IDs, inputs,
labels, and formats, and matching decoding policies where recorded. It rejects
partial overlap, duplicates, empty sets, and mismatched counts.
`--gate` exits nonzero when B significantly regresses at p < 0.05 on any suite.
This exploratory gate has no multiple-comparison correction; do not interpret
repeated searches for p < 0.05 as confirmatory evidence.

High validation accuracy answers a narrow, distribution-specific question.
Validation is not lying when a new distribution exposes a failure. Similarly,
dateutil's `dayfirst=True`/`fuzzy=True` configuration can conflict with some input
formats; that is evidence about this configured system, not automatically a
defect in the library.

## Development and final protocol

Iterate on `dev*`, not `test*`. Choose data, prompts, hyperparameters, model
revision, and checkpoint using training, validation, and development evidence.
Freeze the choice in a nonempty decision document.

```bash
python step5_eval.py --run runs/baseline --final --decision submissions/decision.md
```

Final evaluation requires full suites; `--limit` is rejected. Final results use
separate `_final` report, prediction, and manifest names. The manifest snapshots
the decision's contents and hash. This guard encourages discipline; it cannot
prevent reading local files, editing scripts, changing the decision, or rerunning.
A teacher needing truly blind evaluation should keep final data outside learner
access. After inspecting final errors, treat those items as development data for
any subsequent iteration and obtain a fresh holdout before another final claim.

## Reproducibility and cost

`training_manifest.json`, `training_manifest_lora.json`, and
`manifest_<system>.json` record dataset SHA-256 hashes, CLI configuration,
source commit and dirty state, Python-source hashes, package versions,
hardware/device, and elapsed time. Evaluation records checkpoint or adapter
hashes; SmolLM2 records model/tokenizer revision, actual chat template, system
instruction, and few-shot examples. Keep manifests with their artifacts.

TinyGPT training also retains `config.json` and `log.csv`. LoRA training writes
its configuration in the training manifest and prints losses to the terminal;
it does not create that TinyGPT CSV log. Dataset hashes cover full files; configuration
records a training subset or evaluation limit, and reports record actual scored
counts. Source hashes describe code at record time; do not edit code or data
during a run. A manifest records provenance, not a guarantee of bitwise
determinism or an automatically restorable environment.

Record device, batching, item count, and warmup conditions when comparing cost.
TinyGPT evaluation groups prompts by length with up to 256 per batch; SmolLM2's
`--batch_size` controls its generation batches; the parser processes inputs on
CPU. Their default timings compare whole configured systems, not neural models
under identical batching. Match settings where possible and state differences.
`prediction_seconds` includes prompt/tokenization work but not loading or
training; `seconds_per_item` is throughput-derived, not interactive p95 latency.
CUDA/MPS synchronization and warmup can affect measurements; repeat inference
measurements before making latency claims. Overall manifest elapsed time includes
loading and, for train-and-evaluate mode, training.

## Reading your artifacts

JSON is a structured document; JSONL stores one JSON object per line. Report
fractions such as `0.95` mean 95%; console tables display percentages.

| Artifact | Where / what to inspect |
|---|---|
| Data card | `data/data_card.json`: sizes, format mix, and leakage checks |
| Training curve | `runs/<tinygpt-name>/log.csv`: step, train/val loss, validation EM, elapsed time |
| Summary report | `runs/<output>/report_<tag>.json`: one entry per scored suite, including `n`, errors, and timing |
| Predictions | `preds_<tag>_<suite>.jsonl` beside the report: input, gold answer, prediction, and error type |
| Run identity | `manifest_<tag>.json` beside the report; training manifests describe training separately |
| Adapter | `runs/smollm/lora_adapter/`: small learned updates, **not** a standalone base model |

Tags are `dateutil`, `tinygpt`, `zeroshot`, `fewshot`, or `lora`; final tags append
`_final`. `--name` selects a TinyGPT training directory. `--run` selects an
existing TinyGPT run to evaluate. `--out` sets the parser or SmolLM2 output
directory; it does not redirect TinyGPT evaluation.

## Troubleshooting

- On Apple Silicon, tiny models may be faster on `--device cpu`; SmolLM2 auto
  selects MPS. Lower its `--batch_size` if memory is tight.
- For fast exploration, use the same `--limit 100` for all systems. Never mix
  reports from old and new split protocols.
- No development files? Back up old artifacts, then regenerate data with step 1.
- Keep unique run names. Evaluation with the same tag overwrites its prior report.
- Run `make check` in the activated environment. Tests do not download models.
