---
id: task-10
title: change the way persistence diagram data is stored in runs_df
status: To Do
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
