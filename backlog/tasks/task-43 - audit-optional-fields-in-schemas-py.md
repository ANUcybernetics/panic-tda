---
id: task-43
title: audit optional fields in schemas.py
status: To Do
assignee: []
created_date: "2025-08-06 10:35"
labels: []
dependencies: []
---

## Description

Several of the `Optional` fields in `schemas.py` aren't really optional from a
"domain model" perspective. A couple of examples:

- for the numpy type decorator classes (at the top of the file) the
  inputs/outputs aren't really optional; if they're none, they shouldn't be
  stored to the db (something else has gone wrong and should be handled in
  application code)
- a run's experiment shouldn't be optional
- a persistence diagram's `diagram_data` isn't optional; if it's `None` then you
  don't have a persistence diagram

Perform a full audit of the `schemas.py` file to ensure that all `Optional`
fields are used appropriately.

Making some of these field required may require a migration and a change to the
engine module (e.g. so that the objects aren't committed until the computation
is done, rather than the create object, run the computation then update and
commit pattern used currently in some places).
