---
id: task-36
title: investigate whether yeah/nah run filtering is used consistently
status: To Do
assignee: []
created_date: "2025-07-30"
labels: []
dependencies: []
---

## Description

There are a few runs in the prod database with the initial prompt "yeah" or
"nah". These runs (uniquely) had longer max length - 5000 compared to 1000 for
the other runs in the db. Rather than have to account for this difference in run
lengths, the current analysis code _mostly_ filter out those runs (see, for
example, the SQL statements in the @src/panic_tda/data_prep.py module).

I want to double check that any other parts of the codebase also filter out
these runs. In particular the `cluster embeddings` command, although potentially
others as well.

If necessary, extra filters should be added to the sqlmodel expressions to
filter out any invocations or embeddings from these runs.
