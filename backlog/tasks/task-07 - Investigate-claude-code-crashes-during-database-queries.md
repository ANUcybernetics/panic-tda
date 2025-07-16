---
id: task-07
title: Investigate claude code crashes during database queries
status: To Do
assignee: []
created_date: '2025-07-16'
labels: []
dependencies: []
---

## Description

Claude Code is crashing with std::bad_alloc errors when running certain database queries. This appears to happen with: 1) Long-running Python scripts that query the database, 2) Commands that may produce large outputs, 3) Potentially sqlite3 CLI commands. Hypotheses: memory exhaustion from large result sets, buffer overflow from too much output, or issue with claude code's subprocess handling. Plan: Use smaller queries, avoid sqlite3 CLI where possible, add output limiting.
