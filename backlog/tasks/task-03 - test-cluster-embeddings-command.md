---
id: task-03
title: test cluster embeddings command
status: To Do
assignee: []
created_date: "2025-07-15"
labels: []
dependencies: []
---

## Description

The main cli (@src/panic_tda/main.py) has a cluster embeddings command. It's
tricky to test the cli directly, but I'd like to add a test which tests the full
clustering process (including the creation/storage/retrieval of the embeddings
from the db as per @src/panic_tda/schemas.py).

I'd like to test this for different values of the downsample parameter, both 1
(i.e. no downsampling) and 2 (i.e. downsampling by a factor of 2).

First, check if this functionality is already tested in the current test suite.
If not, create new test(s) to test it.
