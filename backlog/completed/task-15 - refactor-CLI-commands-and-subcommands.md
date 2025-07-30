---
id: task-15
title: refactor CLI commands and subcommands
status: Done
assignee: []
created_date: "2025-07-22"
labels: []
dependencies: []
---

## Description

The CLI interface defined in @src/panic_tda/main.py (you can see the commands
with `uv run panic-tda --help`) is just "one level" and a bit inconsistent. I'd
like to:

- move to a `panic-tda COMMAND SUBCOMMAND` approach, e.g.
  `panic-tda experiment list` instead of `panic-tda list-experiments`
- the top-level commands will be `experiment`, `run`, `cluster`, `model` and
  `export`
- `doctor` should be a subcommand of `experiment`
- `script` can stay as a top-level command
- `paper-charts` should become a `charts` subcommand of `export`

If possible, for the experiment, cluster and run commands it'd be good if the
plural version was acceptable as an alias as well.

Ensure that any tests are updated accordingly, and also any references to these
commands in the documentation (e.g. @README.md, @DESIGN.md).

## Progress

Completed:

- Created hierarchical command groups (experiment, run, cluster, model, export)
- Migrated all commands to appropriate subcommands:
  - `perform-experiment` → `experiment perform`
  - `resume-experiment` → `experiment resume`
  - `list-experiments` → `experiment list`
  - `experiment-status` → `experiment show`
  - `delete-experiment` → `experiment delete`
  - `doctor` → `experiment doctor`
  - `list-runs` → `run list`
  - `list-models` → `model list`
  - `export-video` → `export video`
  - `paper-charts` → `export charts`
  - `export-db` → `export db`
  - `cluster-embeddings` → `cluster embeddings`
  - `list-clusters` → `cluster list`
  - `delete-cluster` → `cluster delete`
  - `cluster-status` → `cluster show`
- Added plural aliases (experiments, runs, clusters) that work as aliases
- Updated documentation in README.md and CLAUDE.md
- Updated shell scripts (perform-experiment.sh, perform-clustering.sh,
  run-doctor.sh)
