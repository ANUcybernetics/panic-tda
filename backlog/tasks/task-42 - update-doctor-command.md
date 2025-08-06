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

Currently the `uv run panic-tda experiment doctor` command only (in
@src/panic_tda/main.py) works on a single experiment.

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

Keep the implementation simple and readable.

## Enhanced Requirements

### Performance & Scalability

- Add progress indicators when checking all experiments (progress bar/counter)
- Process experiments in batches to avoid memory issues with large databases
- **Memory optimization**: Process one experiment/embedding model at a time -
  load all data for a single experiment, perform all necessary work, then
  release that data before loading the next. This avoids loading the entire
  database into RAM at once. Exception: when using Ray parallelism where larger
  batches would benefit throughput

### Enhanced Reporting

- Show summary statistics: total issues found/fixed per category, processing
  time
- Add `--format json` option for structured output (useful for CI/CD
  integration)
- Return appropriate exit codes (non-zero on issues found) for
  monitoring/automation

### Additional Integrity Checks

- Check for orphaned records (embeddings without invocations, PDs without runs)
- Verify sequence_numbers are contiguous (0 to max with no gaps)
- Ensure all foreign key relationships are valid across tables

### Safety Improvements

- Wrap all fixes in database transactions with rollback on error
- Add `--yes` flag to skip confirmation prompts for automation

### Testing Requirements

- Test with empty database
- Test with corrupted/inconsistent data scenarios
- Test transaction rollback on partial failures
- Update tests in @tests/ for the refactored `engine.experiment_doctor` function
