---
id: TASK-56
title: simplify long run exclusion code
status: Done
assignee: []
created_date: '2025-08-14 04:47'
updated_date: '2026-02-09 06:35'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Experiment 067efc98-c179-7da1-9e25-07bf296960e1 (see @src/panic_tda/schemas.py)
has a longer `max_length` property is longer than all the other experiments in
the db.

As a result, it's excluded from a lot of the analysis (e.g. in @data_prep.py)
and also the @doctor.py functionality. However, the way that this exclusion is
handled is a bit inconsistent (either doing "max_length == 5000" checks or
checking that initial_prompt is in ["yeah", "nah"])

Simplify all this "long run exclusion code" so that it just checks if the
associated experiment ID is equal to 067efc98-c179-7da1-9e25-07bf296960e1. So
the behaviour will be the same, but the code can be simplified.
<!-- SECTION:DESCRIPTION:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Closed: irrelevant since porting from Python to Elixir
<!-- SECTION:NOTES:END -->
