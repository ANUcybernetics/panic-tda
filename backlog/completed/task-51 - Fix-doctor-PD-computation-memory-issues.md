---
id: task-51
title: Fix doctor PD computation memory issues
status: Done
assignee: []
created_date: '2025-08-13 00:19'
labels: []
dependencies: []
---

## Description

Remove return_generators from ripser_parallel to fix OOM crashes

## Problem
The `panic-tda doctor --fix` command was crashing with `WorkerCrashedError` when computing persistence diagrams on the production database. Investigation revealed:
- Database has runs with up to 2500 embeddings per model (for max_length=5000 runs)
- Each embedding is 768-dimensional
- The memory issue was caused by `return_generators=True` in ripser_parallel
- Computing homology generators for 2500 points in 768 dimensions requires gigabytes of RAM
- Generators were only used for optional display purposes, not core functionality

## Solution Implemented

### 1. Remove generators computation (src/panic_tda/tda.py)
- Changed `return_generators=True` to `return_generators=False`
- This dramatically reduces memory usage during PD computation
- Generators were only used in display functions and are handled gracefully when absent

### 2. Optimize parallelization settings
- Set `num_cpus=8` per Ray task (src/panic_tda/engine.py:327)
- Set `max_concurrent=4` for selective PD computation (src/panic_tda/engine.py:721, doctor.py:848)
- This allows 4 concurrent jobs × 8 CPUs = 32 CPUs total utilization
- Kept `n_threads=4` in ripser_parallel for good performance

## Result
With these changes:
- Memory usage is dramatically reduced (no generator computation)
- System can handle 4 concurrent PD computations without OOM
- Each computation gets 8 CPUs for good performance
- The doctor --fix command should now work on the production database
