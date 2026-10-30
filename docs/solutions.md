# Worked solutions and interpretation

**Spoilers:** make predictions and collect evidence in the
[guided lab](lab.md) before reading this page.

## 1. Baseline

The specification says `03/04/2024` is April 3, not March 4. A parser can be strong
on unfamiliar formats without any training. Inspect `ymd_dots`: a global
day-first policy may disagree with year-month-day dot notation. Diagnose this
configured policy, rather than declaring the library broken.

## 2. Objective trace

The input prompt includes `=`. If its length is P, `y[P-1]` is the first
answer token. Earlier targets are ignored when masking; prompt inputs remain
visible. EOS is scored and padding is not. Uniform loss is `ln(72)`, about 4.28
for the current vocabulary.

Unmasked training includes unpredictable date content in the prompt, so its
training loss need not approach the answer-only objective's loss. Compare
generated accuracy and common answer-only validation loss instead.

## 3. Broken causality and overfitting

One-batch loss can fall while validation performance stalls or worsens. That
shows capacity to memorize the batch, not a broadly learned skill.
Use the full-data `causal_control` versus `broken_causality` pair to investigate
the causal mask. Comparing the one-batch run to the full-data broken run would
confound training-data exposure with causality.

Without a causal mask, position t can attend to the token at t+1 that it is
supposed to predict. Validation teacher forcing contains future ground-truth
tokens too, so both train and validation loss can look reassuring. Generation
does not contain those future answers. A gap between low teacher-forced loss
and weak generation is a useful diagnostic. Exact timings and magnitudes vary.

## 4-6. Historical observations, not expected answers

The original walkthrough reported these **illustrative runs on the old split
protocol**, before separate development/final populations were introduced:

| Intervention | Familiar ID | Unseen format | Unseen year |
|---|---:|---:|---:|
| TinyGPT, masked, seed 0 | 100% | 0% | 72.4% |
| TinyGPT, unmasked, seed 0 | 100% | 0% | 99.0% |
| 1,000 training rows | 98.7% | not quoted here | 54.2% |
| More training formats, same row count | 100% | 35.6% | 99.8% |

For masked training, seed 1 reportedly reached 28.4% on unseen years despite
100% familiar-distribution accuracy. These figures are retained as teaching
examples, **not newly reproduced results, baselines for the new splits, or
promises about your machine**.

Interpretation: familiar-distribution accuracy may saturate before robustness;
data diversity can help composition; initialization can change generalization.
None of these observations proves a universal rule or uniquely identifies the
mechanism. The new exercises ask you to test them rather than reproduce a table.

For three seeds, show every value, mean, and range per suite. A Wilson interval
uses the items for one trained model; a seed table varies the training process.
Neither establishes representativeness of a hand-designed date benchmark.

## 7. Prompting versus fine-tuning

There is no prescribed winner. Real SmolLM2 quality must be measured, not inferred
from mechanics tests or historical TinyGPT scores.

Few-shot prompting can increase context length and inference cost. LoRA adds
training and adapter maintenance. It is a method of fine-tuning, not a separate
alignment phase after SFT. A parser may be preferable when its policy can
be made consistent with a constrained specification. Do not select solely on
aggregate exact match if the important failure slice regresses.

For the capstone, "no candidate meets the release requirements" is a valid
decision when supported by evidence. Do not tune against the final holdout to
manufacture a pass.
