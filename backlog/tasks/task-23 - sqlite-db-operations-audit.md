---
id: task-23
title: sqlite db operations audit
status: To Do
assignee: []
created_date: "2025-07-26"
labels: []
dependencies: []
---

## Description

Perform an audit of all the databse operations in this application. Pay
attention to:

- use sqlmodel (rather than sqlalchemy) functionality
- use sensible, modern defaults (e.g. sqlite WAL mode) but keep the code
  readable and maintainable---no need to over-engineer for performance reasons
- ensure the use of db session objects are properly and efficiently managed and
  closed
- push filtering/sorting/matching to the db (rather than doing it in application
  code) whenever possible
- ensure the main database code paths are tested using the pytest infrastructure
  in this project
