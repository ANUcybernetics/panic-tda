---
id: task-26
title: refactor local.py module into separate ones for each different paper/deck
status: Done
assignee: []
created_date: "2025-07-27"
labels: []
dependencies: []
---

## Description

The @local.py module now contains code for producing visuals and analysis for
two separate documents. There's

- ieee_smc_charts
- artificial_futures_slides_charts

and all the code they depend on

Finally, there's the paper_charts function at the bottom.

Refactor (potentially adding one more level of module nesting via a `local/`
subfolder) into separate modules for each of those (and potentially more to come
in the future). Ensure that `export charts` (in @main.py) still works (and I'm
fine having the specific module-du-jour hardcoded in paper_charts, because that
doesn't need to change often).

This will help with keeping the claude context relevant. Also, it'd be nice if
the "warn on unused imports" linter thing was turned off for these files (since
they change so much).

## Implementation Notes

The refactoring has been completed:

1. Created a `local_modules/` subdirectory (instead of `local/` to avoid import
   conflicts)
2. Extracted modules:
   - `local_modules/ieee_smc.py` - Contains ieee_smc_charts function
   - `local_modules/artificial_futures.py` - Contains
     artificial_futures_slides_charts function
   - `local_modules/shared.py` - Contains ALL shared helper functions:
     - example_run_ids
     - export_cluster_examples
     - run_counts
     - cluster_counts
     - render_hallway_videos
     - droplet_and_leaf_invocations
     - create_top_class_image_grids
     - list_completed_run_ids
     - TOP4_RUN_IDS constant
3. Updated `local.py` to only contain the paper_charts function and necessary
   imports
4. The `paper_charts` function now imports and calls
   `artificial_futures_slides_charts`
5. Configured ruff linter to ignore unused imports (F401) in these
   frequently-changing files
6. Verified that all imports work correctly
7. Ran ruff linter and formatter on all refactored modules

To switch between different papers/presentations, simply update the import in
`local.py` line 8 to import the desired module (e.g., change from
`artificial_futures` to `ieee_smc`).
