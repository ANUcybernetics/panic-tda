---
id: task-16
title: why is cluster epsilon being set to 0.6
status: Done
assignee: []
created_date: "2025-07-23"
labels: []
dependencies: []
---

## Description

There's a hardcoded value of 0.4 in @src/panic_tda/clustering.py, so why do the
clusters (as shown by the `cluster list` command) show 0.6? Here's an example
output:

```
[16:53] weddle:panic_tda $ uv run panic-tda clusters list
2025-07-23 16:57:50,045 - INFO - Connecting to database at db/trajectory_data.sqlite
Found 3 clustering result(s):
0687fc51-6166-7ed2-9e11-50c97d8cc93b - model: STSBRoberta, epsilon: 0.6, clusters: 72, assignments: 141,600, outliers: 67.7%, created: 2025-07-22 17:06
0687fac7-fae6-7cfc-be09-9cdcaac8a3af - model: STSBMpnet, epsilon: 0.6, clusters: 74, assignments: 141,600, outliers: 70.9%, created: 2025-07-22 15:21
0687f938-e8e0-7447-b2f7-fbb0a4398e0b - model: Nomic, epsilon: 0.6, clusters: 77, assignments: 141,600, outliers: 69.4%, created: 2025-07-22 13:35
```

It's good for the `clusters create` command to choose a sensible value (and 0.6
is fine) but the code should be refactored so that the CLI command can set the
epsilon value as part of "running" the clustering operation.

## Investigation Notes

Found the root cause of the discrepancy:

1. The clustering algorithm in `src/panic_tda/clustering.py:35` uses
   `cluster_selection_epsilon = 0.4`
2. However, when storing the results in the database,
   `src/panic_tda/clustering_manager.py:312` hardcodes the epsilon value as 0.6
   in the ClusteringResult parameters

This is a bug - the stored value doesn't match the actual value used in
clustering.

The CLI command structure:

- `panic-tda clusters embeddings` is the command that runs clustering
- It's defined in `src/panic_tda/main.py` as `cluster_embeddings_command`
- Currently accepts: embedding_model_id, db_path, and downsample parameters
- Does NOT accept epsilon parameter
- Calls `cluster_all_data` from clustering_manager which uses the hardcoded
  values

## Proposed Solution

1. Fix the immediate bug: Make the stored epsilon value match the actual value
   used (0.4)
2. Add epsilon as a CLI parameter to allow users to control the clustering
   sensitivity
3. Pass epsilon through the call chain: CLI → cluster_all_data → hdbscan
   function

## Implementation Completed

Fixed the discrepancy and added epsilon as a configurable parameter:

1. **Fixed the bug**: Changed `clustering_manager.py:312` from hardcoded 0.6 to
   use the actual epsilon value
2. **Updated hdbscan function**: Added epsilon parameter with default 0.4
   (`clustering.py:7`)
3. **Updated cluster_all_data**: Added epsilon parameter to function signature
   (`clustering_manager.py:210`)
4. **Updated CLI command**: Added `--epsilon/-e` option to
   `panic-tda clusters embeddings` command (`main.py:666-671`)
5. **Updated tests**: Fixed test expectation to use 0.4 instead of 0.6

The command now supports custom epsilon values:

```bash
# Use default epsilon (0.4)
uv run panic-tda clusters embeddings

# Use custom epsilon (e.g., 0.6 for less strict clustering)
uv run panic-tda clusters embeddings --epsilon 0.6
```

All tests pass with the changes.
