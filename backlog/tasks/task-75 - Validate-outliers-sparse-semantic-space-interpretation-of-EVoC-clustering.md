---
id: TASK-75
title: Validate 'outliers = sparse semantic space' interpretation of EVoC clustering
status: Done
assignee:
  - '@claude'
created_date: '2026-07-10 00:50'
updated_date: '2026-09-05 05:36'
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
- [x] #1 Distribution of local density (kth-NN distance) over all embeddings is computed and characterised (bimodal dense-cores-plus-diffuse-sea vs unimodal continuum)
- [x] #2 Outlier fraction is reported as a function of min_cluster_size (and noise_level) across a reasonable hyperparameter range, demonstrating whether ~62% is threshold-stable
- [x] #3 Outliers are checked for caption length/verbosity artefacts (outlier status vs caption length and per-I2T-model verbosity)
- [x] #4 Written summary states whether the sparsity interpretation survives, with figures
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
RESOLVED 2026-09-05, negatively. Full write-up in backlog/docs/outlier-sparsity.md; script analysis/outlier_sparsity.py, numbers in analysis/outlier_sparsity.json, figures in analysis/outlier_sparsity/. Measured over all 147,193 Qwen3Embed embeddings (3,896 runs, 12 experiments), the pool the global clustering uses. The 62% in the description was the old mean-pooled scale (TASK-96); the corrected clustering leaves 34-41% unlabelled by layer.

AC#1 density: k-th-NN cosine distance (k=5, 15) is one skewed mode, not dense-cores-plus-sea; the only second mode is the exact-repeat spike at zero. Outliers are no sparser than clustered points: P(outlier sparser than clustered) = 0.51 at k=5, 0.56 at k=15; half of outliers are denser than the median clustered point.

AC#2 sweep: outlier share is 26-45% across base_min_cluster_size 5-1500 and noise_level 0.1-0.9, so the SHARE is threshold-stable, but the SET is not: at min size 5, 75% of stored outliers get labelled and a different third of the data loses its label (ARI 0 vs stored). A second EVoC pass on the outliers alone leaves 39% of them unlabelled again. The share is the procedure's, not the data's. (noise_level is a layout repulsion term in EVoC, not an outlier threshold; outliers are HDBSCAN noise in a 4-d node embedding.)

AC#3 length: outlier rate 0.38-0.48 per captioner with no ordering by verbosity, flat within each captioner across its length terciles; logistic fit on density + length + captioner predicts no better than the majority class.

AC#4 summary: the sparsity interpretation does not survive. The decisive part is temporal: 46% of outlier steps are run tails that never re-enter a cluster (median 19 steps) and 29% are runs never labelled at all; genuine transits between clusters are 10% and short. Outlier rate RISES along the trajectory (0.33 to 0.45). The outlier set is one connected region of the 15-NN graph (94% of outlier neighbours are outliers; 93% of outlier points in components visited by 5+ runs) of ordinary density that runs settle into. 'Time in transit' is not an observable EVoC labels can supply, and TASK-76 cannot milestone outliers as transit; it needs a state definition that assigns every point (note added there).
<!-- SECTION:NOTES:END -->
