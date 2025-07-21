---
id: task-10
title: change the way persistence diagram data is stored in runs_df
status: Done
assignee: []
created_date: "2025-07-20"
labels: []
dependencies: []
---

## Description

The `add_persistence_entropy` function in @src/panic_tda/data_prep.py
unmarshalls the persistence diagram data from the runs_df DataFrame (both the
entropy scores, which are one-per-homology-dimension, and the persistence
diagram data itself, which is one-per-birth-death-pair).

I'd like to refactor the code (and associated tests) so that:

- the `add_persistence_entropy` function only pulls out the entropy scores
  (still with the homology dimension) but adds no birth/death/persistence
  information

- there is a new `load_pd_df` function which follows the same pattern as
  `load_runs_df`:
  - use a SQL select statement (on the `persistence_diagrams` table) to extract
    the persistence diagram data
  - each row has the relevant ID information, including run_id,
    homology_dimension, and persistence_diagram_id
  - there are birth, death and persistence columns (persistence can just be
    calculated from death - birth)
  - use a join (on runs) to get the initial_prompt and network information

Add tests for this new df, and cache it (using parquet) in the same way as the
other polars DataFrames (runs, invocations, embeddings).

This will also mean that several of the functions in @src/panic_tda/datavis.py
will need to be tweaked, becuase it's this new `pd_df` that contains the
relevant birth/death/persisntace data they need (e.g.
create_persistence_diagram_chart). All of the persistence entropy vis functions
can stay the same, because they only need the information that's (still) in the
`runs_df`.

Ensure all tests are updated accordingly.

## Completion Notes

Completed the refactoring as requested:

1. Created new `load_pd_df()` function in @src/panic_tda/data_prep.py:945 that loads persistence diagram data into a dedicated DataFrame with birth/death/persistence columns
2. Refactored `add_persistence_entropy()` in @src/panic_tda/data_prep.py:674 to only add entropy scores (per homology dimension) without birth/death data
3. Added caching support for persistence diagrams via `load_pd_from_cache()` and updated `cache_dfs()`
4. Updated visualization functions in datavis.py to accept pd_df parameter instead of runs_df
5. Added comprehensive test for `load_pd_df()` in @tests/test_data_prep.py:512
6. Updated all affected tests in test_datavis.py to use the new pd_df approach
7. Extended `load_pd_df()` to include text_model and image_model columns needed for faceted visualizations

The refactoring successfully separates concerns:
- `runs_df` contains run-level data with entropy scores only
- `pd_df` contains the detailed persistence diagram data (one row per birth/death pair)
- Both DataFrames are cached using parquet format for performance
