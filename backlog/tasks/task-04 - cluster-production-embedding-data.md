---
id: task-04
title: cluster production embedding data
status: Done
assignee: []
created_date: "2025-07-15"
labels: []
dependencies: []
---

Task abandoned: a better version is now in task-06

notes:

- I don't want this to have anything to do with the dataframes
- downsampling should be based on embedding.invocation.sequence_number (== 0 mod
  downsampling factor)
- current implementation still seems to be using the dataframe

## Description

The production embedding data is in `db/trajectory_data.sqlite` (currently about
130GB). I'd like to cluster it using the @src/panic*tda/main.py
`cluster_embeddings_command`, \_eventually* with no downsampling (i.e.
downsampling_factor=1) if that's possible on the hardware I have (although I'm
not sure it is).

I know this will take a long time, so I want to run it with nohup and logs
directed to a file using the @perform-clustering.sh script.

## Current status

I can start the clustering run (using that script), but it seems to hang. The
best debugging approach is: use the clustering script with large downsampling
factor values (e.g. start at 10k) and watch the script to see how it completes.
Then, decrease it in order-of-magnitude steps until we hit the practical limit
on this hardware.

### Testing Results (2025-07-16)

- **Downsample factor 10000**: Successfully completed

  - Clustered 329/3,288,025 embeddings into 44 clusters

- **Downsample factor 1000**: Failed with error

  - Error: "n_samples=1 while HDBSCAN requires more than one sample"
  - Appears to be a bug where too few samples per model are selected

- **Downsample factor 100**: Failed with same error as 1000

- **Downsample factor 10**: Process appears to hang after starting
  - Log shows it found embeddings and started clustering but no further progress
  - This may be the practical hardware limit

### Key Issues

1. There appears to be a bug in the downsampling logic where factors between
   100-1000 result in too few samples per embedding model for HDBSCAN to work
2. Downsampling factor of 10 may be too computationally intensive for the
   hardware
3. Need to avoid using direct output commands in claude code as they cause
   crashes with std::bad_alloc errors
