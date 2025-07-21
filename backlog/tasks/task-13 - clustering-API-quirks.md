---
id: task-13
title: clustering API quirks
status: To Do
assignee: []
created_date: "2025-07-21"
labels: []
dependencies: []
---

## Description

The "clustering" API (in @src/panic_tda/main.py) has some quirks compared to
e.g. the "experiments" stuff.

- there's no `list-clusters` command (analogous to `list-experiments`)
- there's no `delete-cluster` command (analogous to `delete-experiment`); there
  is a `delete-clusters` which drops all of them, but I think it should be
  `delete-cluster` which drops a single cluster (and perhaps an `all` command to
  drop all clusters---with confirmation, of course)
- cluster-details should be `cluster-status` and take a ClusteringResult ID

Does this sound like an improvement to the overall ergonomics of the CLI? What
changes would be required to the codebase and/or test suite?
