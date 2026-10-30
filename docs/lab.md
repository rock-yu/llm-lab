# Guided lab

[Entry point](../README.md) | [Reference](reference.md) |
[Worked solutions (spoilers)](solutions.md) | [Capstone](capstone.md)

Run commands from the repository root with your virtual environment activated.
Use separate run names; reusing a name overwrites its artifacts. Keep a notebook
or a local file under `submissions/` with each prediction and its eventual evidence.
Do not open final test files during these exercises.

Challenges 1-4 are the core path. Challenges 5-7 are extensions; the six training
runs in Challenge 6 take longer than a single baseline run. Each challenge follows
**predict -> act -> collect evidence -> explain**. Keep the prediction even when
it is wrong: the difference is part of the lesson.

New terms such as *token*, *loss*, *batch*, *ID/OOD*, and *teacher forcing* are
defined in the [reference](reference.md#essential-vocabulary).

## 1. Establish the boring baseline

**Goal:** decide whether ML is necessary and make the labeling policy explicit.

**Predict:** which input formats will a standard date parser find ambiguous?
Write down what `03/04/2024` must mean under this lab's specification.

**Act:**

```bash
python step1_make_data.py
python step5_eval.py --dateutil
```

Skip data generation if you already completed the README quickstart with the
current split protocol. Regenerating with different options can invalidate your
earlier comparisons.

**Evidence:** confirm every leakage count is zero. Read the per-format slices,
largest error bucket, and at least three predictions from `runs/dateutil/`.
State which suite each observation came from. Never use final files for diagnosis.

**Explain:** is a failure a library defect, a configuration mismatch, or an
ambiguous specification? Would the non-ML system already satisfy your use case?

**Checkpoint:** identify `train`, `val`, `dev*`, and `test*`, and explain which
files may influence training, checkpoint selection, experimentation, or final
assessment. These names describe this lab's roles; other projects may call both
validation and development data a "validation set."

## 2. Trace the learning objective

**Goal:** connect text, tokens, shifted targets, and cross-entropy.

**Predict:** for `3 March 2024=2024-03-03\n`, which target first predicts an
answer character? Does masking remove prompt tokens from the model's input?

**Act:**

```bash
python step2_tokenizer.py
python step3_model.py
python - <<'PY'
from step2_tokenizer import CharTokenizer
from step4_train import build_tensors
tok = CharTokenizer()
row = {"input": "3 March 2024", "output": "2024-03-03"}
for masked in (True, False):
    x, y = build_tensors([row], tok, 48, masked)
    print("masked:", masked)
    for i, (a, b) in enumerate(zip(x[0].tolist(), y[0].tolist())):
        if a == tok.pad_id:
            print(i, "<pad>", "-> ignored (remaining positions are padding)")
            break
        print(i, repr(tok.decode([a])), "->",
              "ignored" if b == -100 else repr(tok.decode([b])))
PY
```

**Evidence:** annotate the separator, first answer target, EOS, and padding.
Calculate the uniform-prediction loss `ln(vocab_size)`.

**Explain:** why is a low unmasked training loss not directly comparable to a
masked training loss? Why is validation always scored on answer tokens?

## 3. Prove the loop works, then break it

**Goal:** debug with a small experiment before scaling.

**Predict:** what should happen to training loss, validation loss, and generated
answers when training repeatedly on only one batch? Separately, what might happen
if the model can see future answer tokens during training?

**Act:**

```bash
python step4_train.py --name overfit_check --overfit_one_batch --max_steps 300 --eval_interval 100
python step4_train.py --name causal_control --max_steps 300 --eval_interval 100
python step4_train.py --name broken_causality --diagnostic_no_causal_mask --max_steps 300 --eval_interval 100
```

The first run tests whether the loop can memorize one fixed batch. The last two
form a controlled pair: same full dataset, seed, architecture, and step budget;
only causality changes. Do not compare the first and third runs as if causality
were their only difference.

The third command is **intentionally invalid as an autoregressive model**.
It switches off the causal mask and saves that setting in the checkpoint.
Do not carry this model into later quality comparisons.

**Evidence:** plot or tabulate train/validation loss and validation generated
exact match from the `log.csv` files. Analyze the one-batch run separately, then
compare the controlled pair. Compare teacher-forced loss to generation, not just
the two losses. If 300 steps are inconclusive, extend both controlled runs equally.
At step 0, `train_loss` is `nan` because no update has happened yet; that entry
alone is not evidence of numerical failure.

**Explain:** identify where future answer tokens can leak into an earlier
prediction. Why can't generation use the same shortcut? Describe the expected
symptom rather than assuming a specific numerical result.

## 4. Predict generalization before measuring

**Goal:** distinguish familiar-distribution performance from transferable skills.

**Predict:** write expected exact match and likely failure types for development
ID, unseen-format, and unseen-year sets before running this model.

**ID** means new examples from the same kinds of dates/formats used for training.
**OOD** means out of distribution: here, either a format or year range excluded
from training. The [split table](reference.md#task-and-split-contract) defines each.

**Act:**

```bash
python step4_train.py --name baseline
python step5_eval.py --run runs/baseline
python step5_eval.py --compare runs/dateutil/report_dateutil.json runs/baseline/report_tinygpt.json
```

**Evidence:** compare predictions to observations; report exact match, confidence
intervals, field accuracies, validity, and the largest error slice.

Results live in `runs/baseline/report_tinygpt.json`; individual predictions live
in `preds_tinygpt_dev*.jsonl`. The
[artifact guide](reference.md#reading-your-artifacts) explains these files.

**Explain:** does the validation distribution cover the failure you found?
What evidence supports memorization, and what would be needed to establish a
stronger explanation? High familiar-distribution accuracy alone is insufficient.

## 5. Repair a failure under a fixed data budget

**Goal:** separate data diversity from data quantity.

**Predict:** will adding training formats help a format that remains held out?
Keep training-row count, seed, architecture, and optimizer fixed.

**Act:**

```bash
python step1_make_data.py --extra_formats --out data_aug
python step4_train.py --name more_formats --data data_aug
python step5_eval.py --run runs/more_formats --data data
python step5_eval.py --compare runs/baseline/report_tinygpt.json runs/more_formats/report_tinygpt.json --gate
```

**Evidence:** verify evaluation-file hashes agree between dataset directories.
You can check validation and development files without opening final examples:

```bash
python - <<'PY'
from evals import DEV_SETS
from provenance import file_identity
for name in ("val", *DEV_SETS):
    a = file_identity(f"data/{name}.jsonl")["sha256"]
    b = file_identity(f"data_aug/{name}.jsonl")["sha256"]
    assert a == b, f"{name} changed"
    print(name, "identical")
PY
```

Record whether the targeted error bucket shrank and whether other slices regressed.
The comparison command exits nonzero if it detects a significant regression;
that is evidence to investigate, not an instruction to hide a failing result.

**Explain:** what changed besides the number of formats? Can this single run
establish that diversity is always better? The regenerated training pairs are not
row-for-row identical: this intervention changes the sampled pairs as well as
the format mixture.

Optional extensions: use `--train_size 1000`, or change `--n_layer` and `--n_embd`.
Change one factor at a time; model width must remain divisible by `--n_head`.

## 6. Separate an intervention from initialization luck

**Goal:** measure training-seed variability separately from item uncertainty.

**Predict:** will unmasked training improve unseen-year development accuracy
consistently, or only for one seed?

**Act:**

```bash
for seed in 0 1 2; do
  python step4_train.py --name masked_s${seed} --seed "$seed"
  python step5_eval.py --run runs/masked_s${seed}
  python step4_train.py --name unmasked_s${seed} --seed "$seed" --no_mask_prompt
  python step5_eval.py --run runs/unmasked_s${seed}
  python step5_eval.py --compare runs/masked_s${seed}/report_tinygpt.json runs/unmasked_s${seed}/report_tinygpt.json
done
```

**Evidence:** build a three-row seed table, report mean and range for each
condition on each suite, and identify any inconsistent direction of improvement.
Reuse exactly the same dataset. Here `step4_train.py --seed` changes training
randomness; `step1_make_data.py --seed` would change the dataset, which must stay
fixed for this exercise.

**Explain:** why doesn't a per-run Wilson interval describe seed variation?
Three seeds are a teaching minimum, not proof of a universal effect. Repeated
exploratory tests are not independent confirmatory discoveries.

## 7. Choose between prompting and fine-tuning

**Goal:** justify fine-tuning against realistic alternatives.

**Zero-shot** supplies instructions but no worked examples in the prompt.
**Few-shot** adds seven examples without changing weights. **LoRA** updates small
adapter matrices using labeled training pairs while freezing the base weights.
Zero-shot does not mean the pretrained model has never encountered dates.

**Predict:** which will dominate on accuracy, inference cost, training cost, and
maintenance: parser, TinyGPT, zero-shot, few-shot, or LoRA?

**Act:** start with the same small development limit for all compared systems.

```bash
python step6_finetune_lora.py --mode zeroshot --limit 100
python step6_finetune_lora.py --mode fewshot --limit 100
python step6_finetune_lora.py --mode train --limit 100
python step5_eval.py --compare runs/smollm/report_fewshot.json runs/smollm/report_lora.json
```

These commands download SmolLM2 on first use. For comparisons to the parser or
TinyGPT, rerun their evaluators with `--limit 100` too. Reports must contain
exactly the same items; a partial intersection is rejected.
`--limit 100` uses the first 100 items of each suite, not a new random sample.
It is a quick development check, not the full-suite release result.

**Warning:** rerunning an evaluator with the same output tag replaces its previous
report and predictions, including an earlier full-suite result. Rerun without
`--limit` before the capstone's full-development comparison.

**Evidence:** report item counts, field/error breakdowns, prediction seconds per
item, model size, trainable-parameter count, and elapsed training time. Inspect
the printed chat prompt and saved manifests. Few-shot context affects inference
cost even though it does not change model weights.

Timing fields are saved in the JSON report, not shown in the console table:

```bash
python - <<'PY'
import json
from pathlib import Path
report = json.loads(Path("runs/smollm/report_lora.json").read_text())
for suite, result in report.items():
    print(suite, "n =", result["n"],
          "seconds/item =", round(result["seconds_per_item"], 4))
PY
```

**Explain:** is the extra training justified? TinyGPT/SmolLM2 differences cannot
be attributed solely to pretraining. For a careful rerun, pin `--revision` to the
resolved base-model commit in the manifest.

Rerun the frozen adapter without retraining:

```bash
python step6_finetune_lora.py --mode lora --adapter runs/smollm/lora_adapter --limit 100
```

The reload example assumes the same base model and revision used for training.
For reproducible assessment, follow the capstone's
[revision-pinning instructions](capstone.md#pin-the-base-model-for-adapter-evaluation).

Proceed to the [capstone](capstone.md). Keep final holdouts closed until your
system and decision are frozen.
