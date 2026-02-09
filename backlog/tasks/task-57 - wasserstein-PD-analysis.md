---
id: TASK-57
title: wasserstein PD analysis
status: Done
assignee: []
created_date: '2025-08-15 05:48'
updated_date: '2026-02-09 06:35'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
We have the infrastructure in place to calculate Wasserstein distances between
text persistence diagrams (PDs) from different runs (grep for
`calculate_wasserstein_distances`).

However, we need to do some data analysis (and visualisation) to determine some
sensible bounds for us to assess the result.

In the @src/panic_tda/local_modules/cybernetics_26.py file:

- print out the full (uniqued) list of initial prompts across all runs in the db
- select 2 of them (TBC, wait till we see the printed list from the previous
  step) and load all the persistence diagrams associated with those initial
  prompts
- compute the pairwise wasserstein distances between all these PDs
- plot the distribution (add a new plotting function to the datavis.py module
  for this) which maps fill to whether the two initial prompts were the same or
  different (so we can see whether the initial prompt matters in the
  distribution of these pairwise distances)
<!-- SECTION:DESCRIPTION:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Closed: wontfix for now
<!-- SECTION:NOTES:END -->
