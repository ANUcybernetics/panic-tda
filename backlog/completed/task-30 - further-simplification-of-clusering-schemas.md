---
id: task-30
title: further simplification of clusering schemas
status: Done
assignee: []
created_date: "2025-07-28"
labels: []
dependencies: []
---

## Description

Commit `eef002c40cc9bbfbb05fcf51e794954e368720c8` refactored the way that the
clustering information was stored in the db. Could it have gone even further
(from a simplification perspective)?

We DO NOT need to store any metadata associated with a given cluster (e.g. the
`Cluster` size property could just be a `count` in a DB statement).

In light of this, could the db implementation be further simplified to just a
join table between `ClusteringResult`, `Embedding` (the actual embedding) and
`Embedding` (the medoid embedding, which could be used as the "cluster ID").

Drop any existing clustering data in the DB - we can easily re-create it. And DO
NOT include any "backwards compatibility" code if the schema does change -
perform the migrations, and we'll re-do the clustering.
