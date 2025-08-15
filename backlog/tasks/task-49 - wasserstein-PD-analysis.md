---
id: task-49
title: wasserstein PD analysis
status: In Progress
assignee: []
created_date: '2025-08-15 05:48'
updated_date: '2025-08-15 06:39'
labels: []
dependencies: []
---

## Description

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

## Implementation Notes

Completed analysis with the following:
- Selected "a cat" and "a dog" as the two initial prompts
- Found 64 runs total (32 for each prompt)
- Found 256 persistence diagrams (128 for each prompt, 64 for each of 4 embedding models)
- Computed 8064 pairwise Wasserstein distances
- Added `plot_wasserstein_distribution()` function to datavis.py
- Visualization saved to output/vis/wasserstein_distribution.pdf

Key findings:
- Mean distance for same prompt pairs: 72.64 (std: 63.27)
- Mean distance for different prompt pairs: 71.30 (std: 62.62)
- The distributions are very similar, suggesting initial prompt may not strongly affect the Wasserstein distances

Note: Had to filter out infinite values from persistence diagrams before computing distances.
