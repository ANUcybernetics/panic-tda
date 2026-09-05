---
id: TASK-98
title: Audit the Ash and AshSqlite layer against current best practice
status: To Do
assignee: []
created_date: '2026-09-05 00:39'
labels:
  - instrument
  - ash
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Ash was updated to 3.23.1 (ash_sqlite 0.2.16) without a review of what the
newer versions make possible, and the resource layer has accumulated
workarounds that may no longer be necessary --- or that were never the
idiomatic way to do the thing.

The prompt for this is TASK-92's session, where every resource's update action
turned out to full-scan its table because Ash's atomic update builder writes
the primary key as CAST(id AS TEXT) = CAST(? AS TEXT), which SQLite cannot
answer from the index. The fix was require_atomic?(false) on five resources,
which works and is guarded by test/query_plan_test.exs, but it trades away
atomicity to dodge a query-planner problem rather than addressing it. If
AshSqlite has since gained a way to keep atomic updates and an indexed
predicate, that is the better answer, and it may be worth reporting upstream
either way.

Read the Ash and AshSqlite changelogs since the version this project started
on, and the usage_rules that ship with the deps, then go through the resource
layer looking for things done the hard way.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Ash and AshSqlite changelogs reviewed since the version the project started on, with the features that would apply to this codebase listed and the rest explicitly dismissed
- [ ] #2 Every require_atomic?(false) re-examined: kept with a reason, or replaced by an approach that keeps atomicity and still uses the index, with test/query_plan_test.exs still passing
- [ ] #3 The CAST-on-primary-key behaviour either fixed at the AshSqlite level or reported upstream with a minimal reproduction
- [ ] #4 Raw Ecto and raw SQL call sites audited --- PanicTda.Repo.transaction in the recompute, the sqlite3 reads in analysis/ --- and either moved to an Ash equivalent or recorded as deliberate
- [ ] #5 Custom code that Ash now covers (manual paging, custom calculations, bulk actions, identities) identified and replaced where the Ash version is simpler
- [ ] #6 No behaviour change: the full non-GPU suite passes, and any resource change is covered by a test
<!-- AC:END -->
