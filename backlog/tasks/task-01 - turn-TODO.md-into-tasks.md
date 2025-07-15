---
id: task-01
title: turn TODO.md into tasks
status: To Do
assignee: []
created_date: "2025-07-15"
labels: []
dependencies: []
---

## Description

This is the (old) TODO.md file. Now we're using backlog, it makes sense to turn
some of them (the ones that are actually tasks, and are still relevant - which
isn't all of them) into proper tasks. At some stage I'll do that.

# TODO.md

- figure out exactly why the clustering log is causing claude code to crash
  (maybe submit bug report)... one workaround might be to give the "details" as
  json (which seems less likely to cause an escaping issue... if that's what it
  is)

- cluster the existing data

- check that the clustering manager CLI can delete un-needed clustering runs

- clustering

  - transition matrices (removing self-transitions)
  - order embeddings by "longest stable runs" (with `.rle()`), and calculate
    longest runs for each network (both "from start" and "anytime")
  - histogram of top_n bigrams
  - facet cluster distributions by "stage" (perhaps beginning/middle/end)
  - how often does it return to the initial cluster? how often in a cluster vs
    outlier
  - how many of the labels are similar across different embeddings (or some
    metric on which embeddings get clustered together across different embedding
    models)... double-check that the clustering stuff is actually being faceted
    correctly

- glm with autoregressive parameter (time column)

- add some subsections to the design doc about the GenAIModel and EmbeddingModel
  superclasses (and why we chose the models and params we've currently chosen)

- (maybe) add the vector embedding of the initial prompt to the Run object (to
  save having to re-calculate it later)

- add clustering results to the schemas & db (perhaps a new Cluster SQLModel
  with label/medoid/centroid/embedding_model fields, and then each Embedding has
  many of those)

- populate the estimated_time function (for genai and embedding models) with
  real values

in output video, add visual indicator for which cluster the trajectory is in
(maybe in combination with tSNE)

- for export module, get the `prompt_order` parameter actually working

- add more genai models

- experiment with actor pools for the run stage (because e.g. SDXLTurbo can
  certainly fit a few copies at once)

- visualise the time taken for the various different invocations

- run the tests in GitHub actions

- use the most modern typing syntax (`list` instead of `List`) consistently
  throughout

- check DB indexes
