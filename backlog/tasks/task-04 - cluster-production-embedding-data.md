---
id: task-04
title: cluster production embedding data
status: In Progress
assignee: []
created_date: "2025-07-15"
labels: []
dependencies: []
---

## Description

The production embedding data is in `db/trajectory_data.sqlite` (currently about
130GB). I'd like to cluster it using the @src/panic_tda/main.py
`cluster_embeddings_command`, with no downsampling (i.e. downsampling_factor=1).

I know this will take a long time, so I want to run it with nohup and logs
directed to a file using the @perform-clustering.sh script.

Before doing this, ensure that the prod db contains no clustering data (I've
never successfully done this before, so any data in there is erroneous and comes
from previous attempts which have not worked).

## Progress Notes

- Removed existing clustering data (7 ClusteringResult rows, 2353 EmbeddingCluster rows)
- Updated perform-clustering.sh to avoid hanging on tail -f
- Started clustering process with nohup using: ./perform-clustering.sh
- The clustering is running but will take a very long time due to 4.2M embeddings with no downsampling
- Process appears to be working correctly, just slow due to the large data volume
