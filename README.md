# LLM Lab — train, break, and evaluate a language model on your laptop

You'll train a small GPT **from random weights** to do one job, then spend most of
your time doing what real LLM teams spend most of *their* time doing: finding out
what the model actually learned. Part 2 repeats the experiment with a real
pretrained model and LoRA, scored by the same harness, so you can see what
pretraining buys you.

**The job:** turn a human-written date into ISO 8601.

```
"3rd of March, 2024"    -> "2024-03-03"
"Sunday, 3 March 2024"  -> "2024-03-03"
"03/03/2024"            -> "2024-03-03"     (Australian DD/MM/YYYY)
```

Small enough to train on a CPU in minutes; every answer has an exact ground
truth; and it has sub-skills and distribution shifts you can test on purpose.

```
 Part 1 (from scratch)
 ┌─────────────┐  ┌──────────────┐  ┌───────────┐  ┌────────────┐  ┌──────────────┐
 │ step1 data  │─►│ step2 tokens │─►│ step3     │─►│ step4      │─►│ step5 eval   │
 │ splits, OOD │  │ text → ids   │  │ tiny GPT  │  │ train loop │  │ harness      │
 └─────────────┘  └──────────────┘  └───────────┘  └────────────┘  └──────┬───────┘
                                                                          │ same sets,
 Part 2 (pretrained)                                                      │ same metrics
 ┌──────────────────────────────────────────────────────────────┐         │
 │ step6  SmolLM2-135M: zero-shot → few-shot → LoRA fine-tune    │─────────┘
 └──────────────────────────────────────────────────────────────┘
```

Time: about 1–2 hours including experiments. No GPU needed.

## Setup

```bash
cd llm-lab
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

All results quoted below are real numbers from running this lab on a 2-core
cloud CPU (each training run took ~6 minutes). Yours will be close but not
identical — and *how* not identical is itself one of the lessons (Experiment 6).

---

# Part 1 — Train a GPT from scratch

## Step 1 · Build the data before you build the model

```bash
python step1_make_data.py
```

You get five files plus a data card:

| file | rows | purpose |
|---|---|---|
| `train.jsonl` | 20,000 | what the model learns from (7 formats, years 1950–2039) |
| `val.jsonl` | 500 | what *you* look at during training to pick a checkpoint |
| `test.jsonl` | 1,000 | read once at the end — same distribution as train |
| `test_ood_format.jsonl` | 500 | a format never seen in training: `03-Mar-2024` |
| `test_ood_year.jsonl` | 500 | years never seen in training: 2040–2099 |

**Concepts**

- **The label is a decision.** Is `03/04/2024` the 3rd of April or March 4th? The
  dataset says DD/MM. Your labelling guideline is part of the dataset.
- **Split by entity, not by row.** If *3 March 2024* is in train and *03/03/2024*
  is in test, the model has already seen the answer string. The script splits by
  date first, then renders formats. The leakage checks it prints should all be 0.
- **Build out-of-distribution (OOD) sets on purpose.** The held-out format reuses
  pieces the model has seen (zero-padded day, `Mar`, 4-digit year) in a new
  combination. That's a test of *composition* vs *memorisation*.

> **Think first:** before you train anything, write down what score you expect on
> each of the three test sets. You'll compare later.

## Step 2 · Tokenizer — the only bridge between text and numbers

```bash
python step2_tokenizer.py
```

A model never sees text, only integer IDs, and it outputs a probability over the
vocabulary for the next ID. We use one token per character (vocab = 72) so
nothing is hidden. Real LLMs use sub-word BPE vocabularies of 100k+ tokens.

Two symbols turn "predict the next character" into "answer a question":
`=` separates prompt from answer, and `\n` means *stop*. Generation is just
"keep predicting until you emit `\n`".

Note the printed number: **ln(72) ≈ 4.28**. That's the loss of a model that knows
nothing. It's your first sanity check in Step 4.

## Step 3 · The model — a tiny decoder-only Transformer

```bash
python step3_model.py      # prints 808,704 parameters
```

Read `step3_model.py` top to bottom — it's ~150 lines and it is the same
architecture family as GPT, Llama and Claude:

```
ids → token embedding + position embedding
    → 4 × [ x + Attention(LayerNorm(x)) ;  x + MLP(LayerNorm(x)) ]
    → LayerNorm → Linear → a score for every possible next token
```

The **causal mask** (position *t* may only look at positions ≤ *t*) is what makes
it a language model: every position is trained to predict the next token from
the past only, so one sequence of length *T* gives *T* training signals at once.

## Step 4 · Train it

### 4a. First, prove the training loop can learn *anything*

```bash
python step4_train.py --name overfit_check --overfit_one_batch --max_steps 300 --eval_interval 100
```

This trains on one fixed batch of 64 examples. If loss doesn't go to ~0 here,
you have a bug, and no amount of data will fix it. What I saw:

| step | train loss | val loss | val exact match |
|---:|---:|---:|---:|
| 0 | — | 4.315 (≈ ln 72 ✓) | 0% |
| 100 | 0.501 | **1.455** | 0% |
| 200 | 0.022 | 2.311 | 0% |
| 300 | 0.004 | **2.665** | 0.5% |

Train loss → 0 while val loss *rises*: textbook **memorisation**. The model
learned 64 answers, not a skill. Keep this picture in your head.

### 4b. The real run

```bash
python step4_train.py          # → runs/baseline/
```

The loop is the whole of LLM training: batch → forward → cross-entropy loss →
backward → AdamW step, with learning-rate warmup + cosine decay and gradient
clipping. Every 250 steps it measures val loss and val exact match and saves the
checkpoint with the best val loss. What I saw:

| step | val loss | val exact match | sample: `'6 May 1981'` → |
|---:|---:|---:|---|
| 0 | 4.315 | 0% | `'        jjj'` |
| 250 | 0.317 | 28% | `'1983-05-06'` (format right, year wrong) |
| 500 | 0.088 | 74.5% | `'1981-05-06'` ✓ |
| 1000 | 0.022 | 94.5% | ✓ |
| 1750 | 0.016 | **95.5%** ← a wobble | ✓ |
| 3000 | 0.0000 | 100% | ✓ |

Notice it learns the **shape** of the answer before the **content** — and that
progress isn't monotonic (step 1750).

**Concept — loss masking.** Only the answer characters are scored; prompt
positions get target `-100` and are ignored. That's how supervised fine-tuning
(SFT) usually works. Pretraining is the *same loop* with no masking, on raw text.

## Step 5 · Evaluate — where the real work is

```bash
python step5_eval.py --run runs/baseline
python step5_eval.py --dateutil              # a non-ML baseline
```

My results:

| system | test (in-dist) | OOD format | OOD year |
|---|---:|---:|---:|
| tiny GPT (baseline) | **100.0%** (99.6–100) | **0.0%** (0.0–0.8) | **72.4%** (68.3–76.1) |
| python-dateutil, no ML | 94.5% (92.9–95.8) | 100.0% | 95.6% |

*(exact match, 95% Wilson confidence interval in brackets)*

**What this is telling you**

1. **Validation said 100%. Validation was lying by omission.** Val is drawn from
   the training distribution, so it cannot warn you about anything outside it.
   And a saturated metric (100%) tells you nothing more — you've learned your
   eval is too easy, not that the model is done.
2. **OOD format: 0%, and every answer is malformed** (`'195195'`, `'1967-19'`).
   The model learned seven templates, not "how to read a date". Each piece of
   `03-Mar-2024` was familiar; the combination wasn't.
3. **OOD year: month and day are 98.6% right, the year is wrong** —
   `'11 October 2099' → 1999-10-11`, `'10 April 2047' → 2038-04-10`. The model
   snaps to the nearest answer it has seen. This is what hallucination looks like
   at small scale: fluent, well-formed, confidently wrong.
4. **The partial-credit columns told you *which* skill broke.** Exact match alone
   would just say "72%".
5. **The boring baseline wins on OOD — and has its own bug.** dateutil's 5.5% of
   in-distribution errors are *all one thing*: `1953.12.01` read as YYYY.DD.MM
   (`day_month_swap = 55`, all in the `ymd_dots` slice). Averages hide this; the
   per-format slice and error taxonomy found it in one glance.

**Comparing two systems properly:** they were scored on the same items, so use a
*paired* test. `--gate` makes it a CI-style regression gate (non-zero exit code):

```bash
python step5_eval.py --compare runs/dateutil/report_dateutil.json runs/baseline/report_tinygpt.json --gate
```
```
eval set                A       B     B−A   B wins A wins   p (McNemar)  verdict
test                94.5%  100.0%   +5.5       55      0        0.0000  B better
test_ood_format    100.0%    0.0% -100.0        0    500        0.0000  B WORSE
test_ood_year       95.6%   72.4%  -23.2       16    132        0.0000  B WORSE
GATE FAILED: significant regression on ['test_ood_format', 'test_ood_year']
```

McNemar only looks at items where the two systems *disagree* — that's where
all the information is.

---

## Experiments — change one thing, measure everything

Each experiment is one command plus `python step5_eval.py --run runs/<name>`,
then `--compare` against `runs/baseline/report_tinygpt.json`.

### E1 · Is loss masking actually better?

```bash
python step4_train.py --name no_mask --no_mask_prompt
```

The textbook answer is "mask the prompt". What I measured:

| | test | OOD format | OOD year | val EM @ step 500 |
|---|---:|---:|---:|---:|
| masked (baseline) | 100% | 0% | 72.4% | 74.5% |
| **not masked** | 100% | 0% | **99.0%** | **93.0%** |

Unmasked learned faster *and* generalised to unseen years far better. One
plausible reason: being graded on the prompt forces the model to model the
input's digits and month names, not just map them. Its *train* loss looks worse
(0.40 vs 0.00) because predicting a random date in the prompt is impossible —
**train losses aren't comparable across objectives**, which is why val loss here
is always measured on answer tokens only.

Don't trust one run, though — E6 checks whether this survives a different seed.

### E2 · Less data

```bash
python step4_train.py --name small_data --train_size 1000
```

| | test | OOD year |
|---|---:|---:|
| 20,000 rows | 100.0% | 72.4% |
| 1,000 rows | 98.7% | **54.2%** |

In-distribution barely moved (−1.3 points, still significant by McNemar: 13
items lost, 0 gained). Generalisation dropped 18 points. **In-distribution
accuracy saturates early; robustness keeps paying for more data.**

### E3 · Overfitting on purpose

You already did it in Step 4a. Try `--train_size 200 --max_steps 3000` and watch
`runs/<name>/log.csv`: train loss → 0, val loss bottoms out and climbs. That
turning point is why we save the *best-val* checkpoint, not the last one
(`--ckpt last` in step5 lets you compare).

### E4 · Model size

```bash
python step4_train.py --name tiny --n_layer 2 --n_embd 64      # ~0.1M params
python step4_train.py --name bigger --n_layer 6 --n_embd 192   # ~2.7M params
```

Not run here — predict the outcome first, then check.

### E5 · More *diverse* data

```bash
python step1_make_data.py --extra_formats --out data_aug
python step4_train.py --name more_formats --data data_aug
python step5_eval.py --run runs/more_formats      # evaluates on the ORIGINAL test sets
```

Same 20,000 rows, but spread across 11 formats instead of 7 (e.g. `03-03-2024`,
`03 Mar 2024`). The held-out `03-Mar-2024` is still never seen.

| | test | OOD format | OOD year |
|---|---:|---:|---:|
| 7 formats | 100% | 0.0% | 72.4% |
| 11 formats | 100% | **35.6%** | **99.8%** |

Same amount of data, more variety → the model started *composing*. It's still
far from solved (it now writes things like `1962-23-21` — `impossible_date` in
the taxonomy), but this is the core idea behind why pretraining on diverse text
works.

### E6 · Seed variance — how much of this is luck?

```bash
python step4_train.py --name baseline_s1 --seed 1
python step4_train.py --name baseline_s2 --seed 2
python step4_train.py --name no_mask_s1 --seed 1 --no_mask_prompt
python step4_train.py --name no_mask_s2 --seed 2 --no_mask_prompt
```

I ran seed 1 for both (seed 0 is the default runs above):

| | seed 0 | seed 1 | test (in-dist), both seeds |
|---|---:|---:|---:|
| masked | 72.4% | **28.4%** | 100% |
| not masked | 99.0% | 100.0% | 100% |

*(OOD-year exact match)*

Read that again. Two masked models, **identical** 100% on the test set,
identical training data — one gets 72% of future years right, the other 28%.
Nothing in training loss, val loss or test accuracy distinguishes them. Only
the OOD set does.

Two lessons:

- **Confidence intervals and seed variance measure different things.** The
  Wilson CI covers "which 500 test items did we happen to pick"; it says nothing
  about "which random initialisation did we happen to get". A single run's OOD
  number is an anecdote. When a decision rests on it, report mean and spread
  over 3+ seeds.
- **The E1 result survives:** unmasked training generalised to unseen years on
  both seeds, while masked training is a coin-flip. Run seed 2 to make it three.

---

# Part 2 — Start from a model that already knows language

```bash
python step6_finetune_lora.py --mode zeroshot    # just ask it
python step6_finetune_lora.py --mode fewshot     # ask with 7 worked examples
python step6_finetune_lora.py --mode train       # LoRA fine-tune on 2,000 rows, then evaluate
python step5_eval.py --compare runs/smollm/report_fewshot.json runs/smollm/report_lora.json
```

This uses **SmolLM2-135M-Instruct** (Apache-2.0, ~270 MB download on first run)
and the same `evals.py` harness and test sets as Part 1. It answers the question
teams actually face: *do we need to fine-tune at all, and what does it buy us?*

What's new compared with Part 1:

- **Chat template.** The model was trained on a specific conversation format
  (`<|im_start|>user …`). The script prints the exact prompt so you can see it.
  The system prompt states the DD/MM rule — the prompt is part of the spec.
- **LoRA.** All 135M base weights are frozen; small low-rank adapter matrices
  are learned on the attention projections (well under 1% of parameters). The
  saved adapter is a few MB; the base model is untouched.
- **Same loss masking**, now on sub-word tokens.
- **Left padding for generation, right padding for training** — a classic source
  of silent bugs.

**Before you run it, predict:** will few-shot beat zero-shot? Will the LoRA model
beat your tiny GPT on the OOD sets? (Pretraining has seen countless date formats
and years.) Will it beat dateutil?

> I tested this script's mechanics (chat template, loss masking, LoRA wrapping,
> training, batched generation, scoring) against a tiny randomly-initialised
> Llama, because my sandbox can't reach huggingface.co. The numbers from the real
> SmolLM2 run will be yours to discover. Use `--limit 100` for a quicker first
> pass; `--model_id HuggingFaceTB/SmolLM2-360M-Instruct` for a stronger model.

---

# How this maps to real LLM training

| this lab | production LLMs |
|---|---|
| 72-character vocabulary | 100k–200k sub-word tokens |
| 0.8M parameters | billions to trillions |
| 20k examples, one task | pretraining on trillions of tokens, then SFT on many tasks |
| one objective (next-token cross-entropy) | pretraining → SFT → preference tuning / RL |
| exact match on 2,000 items | benchmark suites, rubric graders, human evals, red-teaming |
| 6 minutes on a CPU | weeks on thousands of GPUs |

The loop, the loss, the masking and the evaluation discipline are the same.

# Eval engineering checklist (the part that transfers to your day job)

1. Write the labelling spec before collecting data.
2. Split by the underlying entity, then check for leakage.
3. Build OOD sets deliberately — they're the only thing that tells you *what* was learned.
4. Always score a non-ML baseline and a prompting baseline before fine-tuning.
5. Report confidence intervals. 500 items can't distinguish 97% from 98%.
6. Slice results by the things you control (format, source, customer segment).
7. Keep an error taxonomy; fix the biggest bucket.
8. Compare systems with paired tests on identical items; wire it into CI as a gate.
9. Use deterministic decoding for evals and measure seed variance.
10. Select checkpoints on val. Touch test once.

# Files

| file | what it is |
|---|---|
| `step1_make_data.py` | dataset, splits, OOD sets, leakage checks |
| `step2_tokenizer.py` | character tokenizer |
| `step3_model.py` | the Transformer + greedy `predict()` |
| `step4_train.py` | training loop |
| `step5_eval.py` | evaluate a run, the dateutil baseline, paired compare / gate |
| `step6_finetune_lora.py` | Part 2: zero-shot / few-shot / LoRA on SmolLM2 |
| `evals.py` | the model-agnostic harness: metrics, CIs, taxonomy, McNemar |

**Troubleshooting.** On Apple Silicon, Part 1's tiny model is often *faster* on
`--device cpu` than `mps` (too small to benefit); Part 2 picks `mps` automatically.
If Part 2 runs out of memory, lower `--batch_size`.
