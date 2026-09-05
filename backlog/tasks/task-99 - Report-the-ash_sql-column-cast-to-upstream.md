---
id: TASK-99
title: Report the ash_sql column cast to upstream
status: To Do
assignee: []
created_date: '2026-09-05 11:38'
updated_date: '2026-09-05 11:44'
labels:
  - ash
  - instrument
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
ash_sql 0.5.5 casts both sides of every comparison on a typed column, so on
SQLite every filter on a uuid primary key or foreign key (and every atomic
update by primary key) renders as CAST(col AS TEXT) = CAST(? AS TEXT) and
full-scans the table. TASK-98 measured about 175 ms per query on the 7.9 GB
dev database, validated a one-function fix in AshSql.Expr.maybe_type_expr
(skip the cast when the Ref already has the target type), and wrote it up
with the reproduction and measurements in backlog/docs/ash-sql-cast-issue.md.
The project works around it with expression indexes on CAST(col AS TEXT)
declared through custom_indexes; those can go once the fix ships.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Issue or pull request opened on ash-project/ash_sql from the draft in backlog/docs/ash-sql-cast-issue.md, with the ash_sqlite reproduction
- [ ] #2 Once a release contains the fix: deps bumped, the CAST expression indexes dropped in a migration, and test/query_plan_test.exs still green
- [ ] #3 The :map atomic-update binding bug reported to ash_sqlite with the reproduction from the doc
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Second AshSqlite bug found on the same path, worth its own upstream issue: an atomic update of a :map attribute passes the map to exqlite unencoded ("unsupported type: %{...}"); the non-atomic path JSON-encodes it. PanicTda.ClusteringResult.update keeps require_atomic?(false) for that reason alone. Reproduction and both findings are in backlog/docs/ash-sql-cast-issue.md.

When the cast fix ships: drop the 13 CAST(col AS TEXT) custom_indexes (migration add_cast_indexes), and Recompute.keyset_order can go back to a plain sort on id.
<!-- SECTION:NOTES:END -->
