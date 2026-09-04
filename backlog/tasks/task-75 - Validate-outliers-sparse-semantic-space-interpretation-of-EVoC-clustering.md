---
id: TASK-75
title: Validate 'outliers = sparse semantic space' interpretation of EVoC clustering
status: To Do
assignee: []
created_date: '2026-07-10 00:50'
updated_date: '2026-09-04 01:01'
labels:
  - analysis
  - paper
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The ~62% outlier rate in global EVoC clustering is currently interpreted as genuine sparsity of the visited semantic space (only dense regions are clusters, per the Hartigan level-set view). Make this a measurement rather than an assumption, since the planned symbolic-dynamics analysis and any paper claims about time-in-transit depend on it. Post-hoc analysis over existing experiment data; blocked on the currently running experiment finishing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Distribution of local density (kth-NN distance) over all embeddings is computed and characterised (bimodal dense-cores-plus-diffuse-sea vs unimodal continuum)
- [ ] #2 Outlier fraction is reported as a function of min_cluster_size (and noise_level) across a reasonable hyperparameter range, demonstrating whether ~62% is threshold-stable
- [ ] #3 Outliers are checked for caption length/verbosity artefacts (outlier status vs caption length and per-I2T-model verbosity)
- [ ] #4 Written summary states whether the sparsity interpretation survives, with figures
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Gate for the paper's 'sparse-space interpretation of outliers' section (backlog/docs/research-programme.md): decides whether 'time in transit' is a real observable or an estimator artefact, which the headline kinetic claim in Results I depends on. Runs after TASK-89, before TASK-76.
<!-- SECTION:NOTES:END -->
