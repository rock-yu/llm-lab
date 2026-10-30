# Frozen system decision

Copy this template to `submissions/decision.md`. Replace every placeholder before
final evaluation. Keep post-evaluation commentary in a separate note.

## Specification and constraints

- Date ambiguity policy and supported input population:
- Quality targets, resource budget, hardware, and batch size:
- Out-of-scope inputs and intended failure behavior:

## Selected system and frozen identity

- System choice and exact evaluation command:
- Code commit and any uncommitted changes:
- Dataset identities / manifest paths:
- Model ID, resolved revision, tokenizer, prompt, few-shot examples:
- Checkpoint or adapter path and SHA-256:
- Training seed(s), configuration, local task-training/fine-tuning time:

For non-ML systems, mark model/training fields not applicable and record parser
version and configuration instead.

## Development evidence

| System | Suite / n | Exact match / CI | Validity | Seconds/item | Seed |
|---|---|---|---|---|---|
| | | | | | |

- Paired comparison to baseline and important error slices:
- Confirmation that reports use full development suites, not a `--limit` subset:
- Seed variation and changes tried during selection:
- Total experimentation cost versus selected-system training cost:

## Rationale and limitations

- Why this system, rather than the simpler alternative:
- Known weaknesses and unsupported claims:
- Ship/no-ship rule to apply to final results:
- Whether output requirements apply to processed predictions or raw generation:
- Confirmation that final items/results have not informed this choice:
