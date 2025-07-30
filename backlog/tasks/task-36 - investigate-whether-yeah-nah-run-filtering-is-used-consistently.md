---
id: task-36
title: investigate whether yeah/nah run filtering is used consistently
status: In Progress
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

## Investigation Results

### Files Checked

1. **data_prep.py** ✅ - All SQL queries properly filter out yeah/nah runs:
   - Line 992: `WHERE run.initial_prompt NOT IN ('yeah', 'nah')`
   - Line 1036: `WHERE invocation.type = 'TEXT' AND run.initial_prompt NOT IN ('yeah', 'nah')`
   - Line 1088: `WHERE run.initial_prompt NOT IN ('yeah', 'nah')`
   - Line 1274: `AND run.initial_prompt NOT IN ('yeah', 'nah')`

2. **clustering_manager.py** ❌ - Missing yeah/nah filtering in cluster embeddings command:
   - `_get_embeddings_query()` - Did not filter yeah/nah runs
   - `cluster_all_data()` count query - Did not filter yeah/nah runs
   - `cluster_all_data()` total query - Did not filter yeah/nah runs

### Fixes Applied

Updated `clustering_manager.py` to add yeah/nah filtering:

1. Added `Run` to imports
2. Modified `_get_embeddings_query()` to:
   - Join with Run table
   - Add filter: `.where(Run.initial_prompt.notin_(["yeah", "nah"]))`
3. Modified count query in `cluster_all_data()` to include Run join and filter
4. Modified total query in `cluster_all_data()` to include Run join and filter

### Other Files Reviewed

- **db.py** - Contains some Embedding queries but they are for specific lookups or exports, not for analysis
- **main.py** - Uses clustering_manager functions, so will benefit from the fixes
- **engine.py** - No relevant queries found
