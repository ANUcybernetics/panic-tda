---
id: TASK-106
title: bin/long-run records only eight characters of the experiment id
status: To Do
assignee: []
created_date: '2026-10-04 09:36'
labels:
  - instrument
  - bug
dependencies:
  - TASK-104
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
experiment.run prints the first eight characters of the new experiment's id, and bin/long-run records what it prints. Those eight characters are the top 32 bits of a UUIDv7's millisecond clock, so two experiments created within about 65 seconds of each other share them. bin/long-run then asks experiment.resume for a prefix that matches two experiments, which is an error, and its loop retries every minute without getting anywhere, with the GPU idle. Seen on 2026-10-04, when two dummy launches thirteen seconds apart were used to exercise the script (TASK-104). A real launch meets it only if a first attempt is abandoned and relaunched inside a minute without being deleted. The fix is in lib/ (print the whole id) or in bin/long-run, and neither should change while experiment 01a10613 is running.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 bin/long-run records an id that stays unique however close together two experiments are created, and resumes by it
- [ ] #2 A test or scripted check launches two experiments within a minute of each other and both resume
<!-- AC:END -->
