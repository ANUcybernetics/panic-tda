---
id: task-12
title: investigate embedding normalisation
status: To Do
assignee: []
created_date: "2025-07-21"
labels: []
dependencies: []
---

## Description

The embedding models support (I think) auto-normalisation, which would make some
of the "normalise so that euclidean == cosine" login in
@src/panic_tda/clustering.py redundant (esp. the "un-normlise medoid to get the
label back" part).

I need to investigate whether this is a better option that the current approach.
