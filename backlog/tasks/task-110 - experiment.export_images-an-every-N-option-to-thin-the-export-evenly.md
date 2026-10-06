---
id: TASK-110
title: 'experiment.export_images: an --every N option to thin the export evenly'
status: To Do
assignee: []
created_date: '2026-10-06 05:19'
labels:
  - export
dependencies: []
references:
  - lib/mix/tasks/experiment.export_images.ex
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
mix experiment.export_images writes every image of an experiment, and its only size control is --limit N, which keeps the first N images overall and so covers a few runs in full and the rest not at all. There is no way to say 'every nth image of every run', which is what a collaborator needs to see what each run looks like without taking the whole set. The long-horizon panel (experiment 01a09e21) is 96,000 images and 5.5 GB; every tenth is 9,600 images and 557 MB.

On 6 October 2026 the every-tenth set for Sungyeon was written by hand, straight from the database into the task's folder layout (sequence_number divisible by 20, so 15 images per run at steps 0 to 280), under long_horizon_panel_01a09e21_parquet/images/. Those files lack the EXIF/XMP metadata the task embeds.

The option should count images, not sequence numbers, so --every 10 keeps a run's 1st, 11th, 21st image and so on. Whether experiment.export_data should grow a matching option that bundles the thinned images beside the parquet files is open. The change is in lib/, so it waits until the long run of experiment 01a10613 has finished (about 25 October 2026).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 mix experiment.export_images <id> --every N writes every Nth image of each run, starting from the run's first, with the usual metadata
- [ ] #2 --every and --limit work together, and the task's moduledoc and CLAUDE.md's task list describe the option
- [ ] #3 A test checks which sequence numbers are written for a run
- [ ] #4 The change lands after experiment 01a10613 has completed
<!-- AC:END -->
