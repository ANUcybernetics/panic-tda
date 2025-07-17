---
id: task-08
title: improve clustering command interface
status: To Do
assignee: []
created_date: "2025-07-17"
labels: []
dependencies: []
---

## Description

Currently, the @src/panic_tda/main.py clustering command clusters for all
embedding models. I'm fine with this for the default, but it would be nice to be
able to specify the clustering to only happen for embeddings with a specific
embedding model.

In light of this, how should the --force flag work? Should it delete all
embeddings? Or only those which will be "replaced" by the new clustering run? I
think I'd prefer to remove the --force option from that command, and just have
another top-level command for deleting cluster data (again, either for all
embedding models or just for one).

In addition to this, the cluster-details command still takes an `experiment_id`
argument. This should be changed to `embedding_model_id` to better reflect what
the argument is used for (again, with "all" being the default).
