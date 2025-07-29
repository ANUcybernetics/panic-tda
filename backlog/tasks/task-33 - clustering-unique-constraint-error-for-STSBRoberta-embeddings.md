---
id: task-33
title: clustering unique constraint error for STSBRoberta embeddings
status: To Do
assignee: []
created_date: "2025-07-29"
labels: []
dependencies: []
---

## Description

There's a persistent error that happens when clustering the STSBRoberta
embeddings. Here's an example log file for one of the failed runs
@logs/clustering_2025-07-29_10-36-36.log (there are many like it, and it's
always the STSBRoberta model that triggers the failure).

What's the specific error that's being caused? Is it because the clustering is
returning the same embedding ID as medoid for multiple different clusters - and
if so why is this happening? Or if not, what's the issue and how do we fix it?
