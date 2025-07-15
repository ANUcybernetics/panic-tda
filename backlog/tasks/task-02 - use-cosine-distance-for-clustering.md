---
id: task-02
title: use cosine distance for clustering
status: To Do
assignee: []
created_date: "2025-07-15"
labels: []
dependencies: []
---

## Description

The clustering in @src/panic_tda/clustering.py (hdbscan is the one I'm currently
using, although there's an optics function in there as well) is done by sklearn.
Currently it uses the euclidean distance, but we should switch to cosine
distance. Sklearn provides this functionality somehow, but we may need to look
at the docs to see the best way to do it.

It's crucial that any tests are updated as well (e.g.
@tests/test_clustering.py).
