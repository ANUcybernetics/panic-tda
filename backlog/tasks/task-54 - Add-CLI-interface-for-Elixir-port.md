---
id: TASK-54
title: Add CLI interface for Elixir port
status: To Do
assignee: []
created_date: '2026-02-06 10:20'
labels:
  - elixir
  - cli
dependencies:
  - TASK-53
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add a Mix task or escript CLI for running PANIC-TDA experiments from the command line, analogous to the Python CLI (uv run panic-tda). Should support creating experiments with specified models, seeds, prompts, and max_length, then executing the full three-stage pipeline and reporting results. Consider using OptionParser or a library like Owl for richer terminal output.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A mix task (e.g. mix panic_tda.run) creates and executes an experiment from CLI arguments
- [ ] #2 Supports specifying networks, seeds, prompts, embedding models, and max_length
- [ ] #3 Prints progress updates as each stage completes
- [ ] #4 Prints a summary of results (invocation count, embedding count, PD info) on completion
<!-- AC:END -->
