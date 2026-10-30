# Spec-driven development

This harness is deliberately small: Markdown specifications, task checklists,
and executable checks. No external spec framework is required.

1. Before changing behavior, add or update a numbered specification directory.
2. In `spec.md`, define requirements, observable acceptance criteria, decisions,
   and non-goals.
3. In `tasks.md`, link implementation tasks to requirements and dependencies.
4. Implement, run `make check` and feature-specific smoke checks, then record
   evidence and check off completed tasks.
5. Keep specifications, CLI behavior, exercises, and solutions consistent.

## Specifications

- [001: Guided learning lab](001-guided-lab/spec.md)
- [Implementation tasks and evidence](001-guided-lab/tasks.md)

Generated datasets, checkpoints, reports, and learner submissions are not source
files and must remain ignored by Git.
