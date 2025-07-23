---
id: task-18
title: debug semantic drift (cosine) calculation
status: To Do
assignee: []
created_date: "2025-07-23"
labels: []
dependencies: []
---

## Description

I want to:

- remove the cosine AND euclidean "semantic drift" calculations (in
  data_prep.py, but referenced in other places); replace it with a single one
  (that's cosine, but that doesn't need to be in the name anymore)
- comprehensively test the "add semantic drift" calculation, which takes an
  embeddings_df and
  - splits it by run + embedding_model
  - re-computes the embedding for the initial prompt
  - calculates the cosine distance from the initial prompt's embedding to each
    successive embedding in the run (well, the text ones anyway)
- for that calculation, it probably makes sense to use the "normalise, then use
  euclidean" trick that we use in clustering.py

The most important thing is that this is tested (ideally with known values).

In principle this functionality is already there, but I want to refactor it and
convince myself that it's correct.
