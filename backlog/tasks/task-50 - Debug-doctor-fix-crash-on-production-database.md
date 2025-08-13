---
id: task-50
title: Debug doctor --fix crash on production database
status: Done
assignee: []
created_date: "2025-08-12 23:35"
labels: []
dependencies: []
---

## Description

Investigate why 'panic-tda doctor --fix --yes' is failing on the production
database (db/trajectory_data.sqlite) when tests pass

## Investigation Findings

tl;dr it was the "return generators" thing that was eating too much memory, and
we don't really need that, so I removed it a(see task-51).

### Error Analysis

The crash occurs with a `WorkerCrashedError` from Ray when trying to compute
persistence diagrams. The error traceback shows:

- The crash happens in `perform_pd_stage_selective()` at line 745 in engine.py:
  `batch_results = ray.get(batch_tasks)`
- This is called from `fix_persistence_diagrams()` in doctor.py (line 848)
- The Ray remote task `compute_persistence_diagram` (decorated with
  `@ray.remote(num_cpus=4)`) is crashing

### Root Cause Hypothesis

The `compute_persistence_diagram` function (line 327 in engine.py) is likely
crashing due to:

1. **Memory issues**: The function loads all embeddings into memory and creates
   a numpy array (line 401)
2. **Large point clouds**: With a 250GB production database, some runs might
   have very large embedding sets
3. **Giotto-ph computation**: The `giotto_phd()` function uses `ripser_parallel`
   with 4 threads, which could consume significant memory

### Key Code Path

1. `doctor --fix` → `fix_persistence_diagrams()` →
   `perform_pd_stage_selective()`
2. `perform_pd_stage_selective()` submits Ray remote tasks for
   `compute_persistence_diagram`
3. `compute_persistence_diagram()` loads embeddings, creates numpy array, calls
   `giotto_phd()`
4. `giotto_phd()` uses `ripser_parallel` with `n_threads=4` and
   `return_generators=True`

### Why Tests Pass

Tests use small datasets with dummy models, so memory pressure doesn't occur.
The production database has millions of invocations and embeddings.

## Database Scale

- 3,220 runs
- 3,732,000 invocations (avg ~1,159 per run)
- 7,029,100 embeddings
- 250GB database size
- Maximum embeddings per run per model: 2500 (for max_length=5000 runs)
- Embedding dimensions: 768

## Actual Root Cause

The crash is NOT due to duplicate embeddings or 10k+ point clouds. The actual
issue is:

1. Max embeddings per run per model is only 2500 (verified)
2. A 2500x768 float32 array is only ~7.68MB
3. The memory issue comes from `ripser_parallel` with `return_generators=True`
4. Computing homology generators for 2500 points in 768 dimensions is extremely
   memory-intensive
5. With 4 threads and generator computation, this can easily consume several GB
   of RAM

## Solution Implemented

### Changes Made

1. **Removed return_generators=True** (src/panic_tda/tda.py:23)

   - Generators were only used for display purposes in db.py
   - Removing them SIGNIFICANTLY reduces memory usage during PD computation
   - This was the main culprit - computing generators for 2500 points in 768
     dimensions is extremely memory-intensive

2. **Reduced thread count** (src/panic_tda/tda.py:24)

   - Changed n_threads from 4 to 2 in ripser_parallel
   - Further reduces memory pressure

3. **Increased Ray worker memory allocation** (src/panic_tda/engine.py:327)

   - Added `memory=8GB` to the `@ray.remote` decorator
   - Provides buffer for the still-intensive computation

4. **Reduced concurrency** (src/panic_tda/doctor.py:848)

   - Changed max_concurrent from 8 to 2 for PD computation
   - Prevents multiple workers from exhausting system memory

5. **Improved logging** (src/panic_tda/engine.py:408-420)
   - Added info logs showing point cloud size before/after computation
   - Enhanced error messages to include embedding count for debugging

### Why This Works

- The main issue was `return_generators=True` causing massive memory usage
- Computing homology generators for 2500 points in 768-dimensional space
  requires gigabytes of RAM
- By removing generator computation (which was only used for optional display),
  we dramatically reduce memory usage
- The other changes provide additional safety margins
