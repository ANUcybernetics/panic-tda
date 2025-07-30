---
id: task-07
title: Investigate claude code crashes during database queries
status: Done
assignee: []
created_date: "2025-07-16"
labels: []
dependencies: []
---

GH Issue created: <https://github.com/anthropics/claude-code/issues/3660>

## Description

Claude Code is crashing with std::bad_alloc errors when running certain database
queries. This appears to happen with: 1) Long-running Python scripts that query
the database, 2) Commands that may produce large outputs, 3) Potentially sqlite3
CLI commands. Hypotheses: memory exhaustion from large result sets, buffer
overflow from too much output, or issue with claude code's subprocess handling.
Plan: Use smaller queries, avoid sqlite3 CLI where possible, add output
limiting.

## Investigation Results

**CONFIRMED**: Claude Code crashes with `std::bad_alloc` when running even
simple `sqlite3` CLI commands. A basic query to list 5 table names caused an
immediate crash with error code 134.

## Root Cause

The issue appears to be with Claude Code's handling of the `sqlite3` CLI tool
output, not with the query complexity or result size. Even trivial queries
crash.

## Workarounds

### 1. Use Python SQLite Instead of CLI

Instead of:

```bash
sqlite3 database.db "SELECT * FROM table;"
```

Use:

```python
import sqlite3
conn = sqlite3.connect('database.db')
cursor = conn.execute("SELECT * FROM table")
for row in cursor.fetchmany(10):  # Limit results
    print(row)
conn.close()
```

### 2. Redirect sqlite3 Output to File

Instead of direct output, redirect to a file:

```bash
sqlite3 database.db "SELECT * FROM table;" > query_results.txt
head -n 20 query_results.txt  # View first 20 lines
```

### 3. Use Database Tools via Python Scripts

Create small Python scripts for database operations:

```python
# db_query.py
import sqlite3
import sys

def query_db(db_path, query, limit=100):
    conn = sqlite3.connect(db_path)
    cursor = conn.execute(query)
    results = cursor.fetchmany(limit)
    for row in results:
        print(row)
    conn.close()

if __name__ == "__main__":
    query_db(sys.argv[1], sys.argv[2])
```

### 4. Use SQLModel/Polars for Database Access

This codebase already uses SQLModel and Polars, which handle database
connections safely:

```python
from panic_tda.schemas import Run
from panic_tda.engine import open_session

with open_session("./db/trajectory_data.sqlite") as session:
    runs = session.query(Run).limit(10).all()
```

## Recommendations

1. **Never use `sqlite3` CLI directly** in Claude Code on this machine
2. **Always use Python-based database access** (sqlite3 module, SQLModel,
   Polars)
3. **If CLI output is needed**, redirect to file first, then read the file
4. **For ad-hoc queries**, create small Python scripts or use the existing
   panic-tda infrastructure
5. **Document this limitation** in CLAUDE.md for future reference
