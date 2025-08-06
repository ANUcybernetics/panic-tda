---
id: task-42
title: update doctor command
status: To Do
assignee: []
created_date: "2025-08-06 07:07"
labels: []
dependencies: []
---

## Description

Currently the `uv run panic-tda experiment doctor` command only (in main.py)
works on a single experiment.

Refactor it so that it checks all data in the db for:

- invocations with missing outputs
- embeddings with missing vectors
- runs have a full set of outputs (i.e. one invocation only for sequence_number
  == 0 to max_sequence_number)

Keep the current report/fix behavior.

In addition, promote `doctor` to a top-level command, not an "experiment"
subcommand.

Ensure that all tests still pass, and add/update any tests of the
`engine.experiment_doctor` function in accordance with any changes made.
