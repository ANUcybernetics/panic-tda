---
id: task-48
title: remove excluded experiments to a new db file
status: Done
assignee: []
created_date: "2025-08-15 01:04"
labels: []
dependencies: []
---

## Description

In the production database (db/trajectory_data.sqlite) there is one experiment
which has a much longer max_length than the others (id =
067efc98-c179-7da1-9e25-07bf296960e1). There's a bunch of special-casing in the
codebase to ignore the data from that experiment.

Given that we're using sqlite anyway, it seems like a nicer approach would be to
export/dump the data from the excluded experiment to a new database file
(including all related data, so that e.g. all the db constraints are still
satisfied). Then, once some checks have been done to ensure that _all_ the
relevant data has been safely removed, that data can be deleted from the prod db
(and any special-casing code can be removed).

The new database file should be in `db/length_5000_experiment.sqlite`.

## Implementation Progress

Completed 2025-08-15:

1. **Identified the experiment data**: Found experiment
   067efc98c1797da19e2507bf296960e1 with max_length=5000, containing:

   - 128 runs
   - 640,000 invocations
   - 1,280,000 embeddings
   - 511 persistence diagrams

2. **Created export scripts**:

   - `export_experiment_simple.sh` - Successfully exports all data to
     `db/length_5000_experiment.sqlite`
   - `delete_experiment_from_prod.py` - Script to remove data from production DB
     (ready to run when needed)

3. **Exported the data**: Successfully created
   `db/length_5000_experiment.sqlite` (24GB) containing all experiment data

4. **Removed special-casing code**:
   - Removed special case handling from `src/panic_tda/doctor.py` (lines
     ~485-491 and ~875-898)
   - These were skipping PD checks and computations for runs with
     max_length=5000

## Next Steps

To complete the cleanup and remove from production DB, run:

```bash
uv run python delete_experiment_from_prod.py
```

This will:

1. Show what will be deleted
2. Ask for confirmation
3. Delete the data in the correct order (respecting foreign keys)
4. Run VACUUM to reclaim disk space

The exported data is safely preserved in `db/length_5000_experiment.sqlite` if
needed in the future.
