---
id: TASK-54
title: Add Mix task for running experiments from the CLI
status: Done
assignee: []
created_date: '2026-02-06 10:20'
updated_date: '2026-02-09 06:35'
labels:
  - elixir
  - cli
dependencies:
  - TASK-53
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
## Context

The full experiment pipeline works end-to-end (runs → embeddings → TDA → clustering) via PanicTda.Engine.perform_experiment/1, but there's no CLI entry point. Users must currently use iex to create and run experiments programmatically.

## Goal

A user should be able to run a full experiment from the command line with a single command, e.g.:

```
mise exec -- mix experiment.run config.json
```

## Implementation plan

### 1. Create Mix.Tasks.Experiment.Run

Create lib/mix/tasks/experiment.run.ex that:

- accepts a path to a JSON config file as the first argument
- parses the JSON into the map shape expected by PanicTda.create_experiment!/1:
  ```elixir
  %{
    networks: [["FluxSchnell", "BLIP2"], ...],
    seeds: [42, 123, ...],
    prompts: ["a banana", ...],
    embedding_models: ["STSBMpnet", ...],
    max_length: 1000
  }
  ```
- creates the experiment via the domain code interface
- prints the experiment ID
- calls PanicTda.Engine.perform_experiment/1
- logs progress at each stage (the engine already uses Logger)
- prints a summary on completion (run count, embedding count, PD count, cluster count)
- exits with a non-zero status on failure

### 2. Add a sample config file

Create config/experiment.example.json with a working dummy-model config so users can test without a GPU:

```json
{
  "networks": [["DummyT2I", "DummyI2T"]],
  "seeds": [42, 123],
  "prompts": ["A beautiful sunset", "A mountain landscape"],
  "embedding_models": ["DummyText"],
  "max_length": 10
}
```

### 3. Wire up database setup

The Mix task should ensure the database is created and migrated before running (either via setup alias or by calling ecto.create and ecto.migrate inline, similar to how the test alias works).

### 4. Stretch: mix experiment.list and mix experiment.status

Optional follow-up tasks:
- mix experiment.list — list all experiments with their status (pending/running/completed) and timestamps
- mix experiment.status <id> — show details for a specific experiment (run count, stage progress, completion time)
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 mise exec -- mix experiment.run config/experiment.example.json runs a full pipeline with dummy models and exits cleanly
- [ ] #2 Experiment results are persisted in the SQLite database
- [ ] #3 Progress is visible in the terminal via Logger output
- [ ] #4 A final summary is printed showing what was created (run count, embedding count, PD count, cluster count)
- [ ] #5 Non-zero exit status on failure
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Already implemented: mix experiment.run, experiment.list, experiment.status all exist
<!-- SECTION:NOTES:END -->
