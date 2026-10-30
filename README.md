# LLM Lab: train, investigate, and explain

A laptop-sized lab for learning **how language models learn, how they fail, and
how to evaluate them fairly**. The task is deliberately small:

```text
"3rd of March, 2024" -> "2024-03-03"
"03/04/2024"        -> "2024-04-03"  (DD/MM/YYYY)
```

Predict -> implement -> measure -> explain -> improve. A high score is not the
finish line: evidence that a system meets its requirements is.

You will train a small **GPT-style language model**, not a general-purpose chatbot.
It learns to generate an answer one token at a time. The small task makes training
mechanics visible without needing large-scale compute.

## Start here

| Material | Purpose |
|---|---|
| [Guided lab](docs/lab.md) | Seven progressive challenges with predictions and deliverables |
| [Reference](docs/reference.md) | Concepts, architecture, metrics, and reproducibility |
| [Worked solutions](docs/solutions.md) | Hints, interpretations, and illustrative historical results; spoilers |
| [Capstone](docs/capstone.md) | Choose a date normalizer under quality and resource constraints |
| [Decision template](docs/decision-template.md) | Record and freeze your choice before final evaluation |
| [Specs and tasks](specs/README.md) | Small contributor workflow and acceptance criteria |

**Prerequisites:** basic Python (functions, loops, lists) and terminal commands.
Prior PyTorch experience is not required; start with the
[essential vocabulary](docs/reference.md#essential-vocabulary) when a term is new.
Use a Python version supported by the installed PyTorch release.

| Route | What to do |
|---|---|
| Core basics | Challenges 1-4: data, tokens, training, and evaluation |
| Experiment extensions | Challenges 5-6: data diversity and repeated training seeds |
| Pretrained-model extension | Challenge 7: prompting and LoRA fine-tuning |
| Decision exercise | Capstone, using the systems you have actually evaluated |

Budget roughly 1-2 hours for the core basics, depending on hardware and reading
time. Six seed-comparison runs and pretrained-model experiments can take
substantially longer. CPU is sufficient; no GPU or API key is required.
Challenge 7 downloads a pretrained model on first use and needs internet access.

## Setup and first experiment

Run from the repository root. These commands and the guided lab use Bash/zsh
syntax on macOS/Linux; other shells need equivalent activation and loop commands.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python step1_make_data.py
python step5_eval.py --dateutil
```

Inspect the baseline's development errors before training a model.
Continue with [Challenge 1](docs/lab.md#1-establish-the-boring-baseline); if you
already ran these commands, reuse their output rather than regenerating data.

**Existing checkout?** Back up old `data/` and `runs/` before regenerating data.
The new split protocol adds development populations and changes training/validation
membership. Old checkpoints are not clean baselines for the new protocol.
Missing development files cause an explicit error, never a fallback to test data.

## Where this fits in the LLM lifecycle

| Phase | This lab |
|---|---|
| Data preparation | Synthetic labels, entity-based splits, leakage checks, deliberate distribution shifts |
| Tokenization and architecture | Inspectable character tokenizer and ~0.8M-parameter decoder-only GPT |
| Broad-corpus pretraining | Explained, **not implemented** |
| Supervised task training | TinyGPT trained from random weights on date pairs |
| Fine-tuning | SmolLM2 prompting baselines and LoRA |
| Preference alignment / RLHF / DPO | Out of scope |
| Evaluation | Development, final holdout, error diagnosis, paired comparisons, seed variation |
| Production deployment | Decision exercise, **not a production service** |

Training from scratch is not synonymous with pretraining. Part 1 uses a supervised
task objective from random initialization. Part 2 compares whole systems: model
size, tokenizer, architecture, pretraining, and instruction tuning all differ.
It does not isolate the causal effect of pretraining.

## Evaluation contract

- `val` selects TinyGPT checkpoints; it is not the experiment scoreboard.
- `dev`, `dev_ood_format`, and `dev_ood_year` support repeated investigation.
- `test`, `test_ood_format`, and `test_ood_year` are final holdouts. Freeze the
  selected system in a decision document first, then use `--final --decision PATH`.
- Final evaluation always uses full suites and writes separately named
  `report_*_final.json` files. Files are visible locally: this is a procedural
  safeguard, not access control or a guarantee against reruns.

## Source map and checks

| File | Role |
|---|---|
| `step1_make_data.py` | Data, splits, OOD sets, data card, leakage checks |
| `step2_tokenizer.py` | Character encoding |
| `step3_model.py` | Decoder-only Transformer and greedy generation |
| `step4_train.py` | Training, checkpoint selection, diagnostic experiments |
| `step5_eval.py` | TinyGPT / parser evaluation and paired regression gate |
| `step6_finetune_lora.py` | Zero-shot, few-shot, LoRA training and adapter evaluation |
| `evals.py` | Shared suite protocol, scoring, error taxonomy, statistics |
| `provenance.py` | Dataset, source, environment, and run identity |

```bash
make check                   # activated environment; no model downloads
# or: make check PYTHON=.venv/bin/python
```

Generated datasets, runs, environments, and `submissions/` are ignored by Git.
Specifications and learning materials remain tracked source.
