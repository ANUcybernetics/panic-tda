---
id: TASK-109
title: >-
  experiment.export_data leaves the text-to-image seed out of
  invocations.parquet
status: To Do
assignee: []
created_date: '2026-10-06 05:02'
labels:
  - export
dependencies: []
references:
  - lib/panic_tda/data_export.ex
  - test/data_export_test.exs
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Every text-to-image invocation stores the seed it was generated from (Invocation.seed, added for TASK-90), but the invocations table written by mix experiment.export_data has no seed column: the column list in lib/panic_tda/data_export.ex predates the attribute. Anyone working from a parquet export cannot attribute within-condition variation to the seed or name the seed needed to regenerate an image.

Found on 6 October 2026 while preparing the long-horizon panel (experiment 01a09e21) for Sungyeon. Her copy, long_horizon_panel_01a09e21_parquet/, had the column joined on by hand from the database: 96,000 image rows with a seed, 95,999 of them distinct, null on text rows. The original export in 01a09e21_parquet/ still lacks it.

The fix is in lib/, so it waits until the long run of experiment 01a10613 has finished (about 25 October 2026). The synthetic prompt row that --embed-prompts adds (sequence_number -1) needs a null seed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 invocations.parquet has an integer seed column, set on image rows and null on text rows and on the synthetic prompt row
- [ ] #2 test/data_export_test.exs checks that an exported seed equals the stored one
- [ ] #3 The change lands after experiment 01a10613 has completed
<!-- AC:END -->
