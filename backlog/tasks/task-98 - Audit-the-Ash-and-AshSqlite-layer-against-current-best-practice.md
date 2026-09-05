---
id: TASK-98
title: Audit the Ash and AshSqlite layer against current best practice
status: Done
assignee:
  - '@claude'
created_date: '2026-09-05 00:39'
updated_date: '2026-09-05 11:47'
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
- [x] #1 Ash and AshSqlite changelogs reviewed since the version the project started on, with the features that would apply to this codebase listed and the rest explicitly dismissed
- [x] #2 Every require_atomic?(false) re-examined: kept with a reason, or replaced by an approach that keeps atomicity and still uses the index, with test/query_plan_test.exs still passing
- [x] #3 Raw Ecto and raw SQL call sites audited --- PanicTda.Repo.transaction in the recompute, the sqlite3 reads in analysis/ --- and either moved to an Ash equivalent or recorded as deliberate
- [x] #4 Custom code that Ash now covers (manual paging, custom calculations, bulk actions, identities) identified and replaced where the Ash version is simpler
- [x] #5 No behaviour change: the full non-GPU suite passes, and any resource change is covered by a test
- [x] #6 The CAST-on-uuid behaviour is worked around in this repo (expression indexes via custom_indexes, plans asserted for reads and updates) and the upstream report is handed to TASK-99 with a minimal reproduction
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read the Ash 3.14.0->3.23.1, ash_sql 0.4.2->0.5.5 and ash_sqlite 0.2.15->0.2.16 changelogs and the usage-rules; list what applies.
2. Reproduce the CAST-on-primary-key SQL with a probe resource and EXPLAIN QUERY PLAN; establish which DSL switches avoid it and whether reads are affected.
3. Measure the cost on the real dev database, read-only.
4. Validate a candidate ash_sql fix against the probes and the full suite on a local copy of the dep, then revert; write it up as an upstream report (backlog/docs/ash-sql-cast-issue.md).
5. Re-examine every require_atomic?(false); audit the raw Ecto sites and the analysis sqlite3 reads; replace custom code Ash covers where AshSqlite actually supports it.
6. Full non-GPU suite green; commit reflow and semantic changes separately.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
CHANGELOG REVIEW (AC#1). Project started on ash 3.14.0 / ash_sql 0.4.2 / ash_sqlite 0.2.15 (commit 178b798); now 3.23.1 / 0.5.5 / 0.2.16. Applies here: ash_sql 0.5.0 "don't cast integer/string equality check unnecessarily" (the partial fix the cast bug below extends); ash 3.23.0 atomic update_timestamp fix (explains the CASE expression in atomic SQL); ash 3.17.0 not_found_error? on gets (used implicitly by the new find_experiment). Dismissed as not applicable: pipelines DSL, batch_validate, paginate_by_default?, :auto calculation type, attribute_always_select?, required! shorthand, gettext, internal actions, offset on has_many, partial-success bulk_create, touch_update_defaults?, nils_distinct? identities, multitenancy bypass, manual relationships, notifier loads, ash_sqlite migration-generator fixes.

THE CAST BUG (AC#2, AC#3). Reproduced with a probe resource: every comparison on a uuid column, reads and atomic updates alike, renders as CAST(col AS TEXT) = CAST(? AS TEXT) and plans as a full SCAN; measured 171-178 ms per query on the 7.5 GB dev DB (308k invocations) for exactly the per-run queries the embeddings, PD and resume stages issue. A one-function ash_sql fix was validated on a local copy of the dep and reverted; patch, reproduction and measurements are in backlog/docs/ash-sql-cast-issue.md, and reporting it is TASK-99 (Ben's call not to post from here).

The in-repo fix is the canonical DSL one: every filtered uuid column carries an expression index on CAST(col AS TEXT) declared through custom_indexes (13 indexes; migration add_cast_indexes; 6.6 s and 138 MB on the dev DB, applied 2026-09-05). SQLite matches them against ash_sql's predicate, so reads and atomic updates seek: the same queries now run in 0-11 ms on the real DB. That let four of the five require_atomic?(false) guards go. ClusteringResult.update keeps its own for a different AshSqlite bug: the atomic path binds the :map attribute unencoded and exqlite rejects it (also recorded on TASK-99). Experiment's four flags were dead (AshSqlite has no expr_error, so validated actions are silently non-atomic and Ash never raises) and are gone; expr(now()) for its timestamps was tried and reverted because set_attribute does not evaluate expressions on the non-atomic path. The recompute's keyset page needed one more step: with ORDER BY id the planner still walked the primary key from the first row each page, so it now sorts by expr_sort(fragment("?", id)), which renders as the indexed cast expression and turns the page into a seek.

RAW ECTO / SQL (AC#4). Both lib sites are deliberate and say so in comments: AshSqlite reports can?(:transact) false, so Ash.transaction is a no-op and the recompute keeps PanicTda.Repo.transaction; can?(:distinct) false, so ClusteringStage.all_embedding_models keeps its Ecto distinct query. The analysis/*.py sqlite3 reads are read-only SELECTs over the dev DB from Python; the Ash-side equivalent is experiment.export_data to parquet. Deliberate, no change.

CUSTOM CODE (AC#5). Run.invocation_count and ClusteringResult.cluster_count were count/2 calculations AshSqlite cannot evaluate (aggregates unsupported) and nothing used them: removed. Six copies of find_experiment are one read action by_id_prefix plus a code-interface get; an ambiguous prefix now errors instead of picking the first match. EmbeddingsStage creates a run's embeddings with one Ash.bulk_create; stored rows identical. Considered and rejected with the reason in a comment: cascade_destroy for experiment.delete (needs keyset pagination on every primary read action and cannot honour the newest-first order the self-referencing input_invocation_id key forces). Kept as-is: the julianday duration calculations (Ash has no datetime subtraction), identities, RunExecutor's per-item invocation creates (the resume logic reasons about their ordering).

IMPACT ON TASK-90. Nothing here changes what is stored; the run path is create-dominated and the per-run reads it does issue now seek instead of scanning a growing table.

VERIFICATION (AC#6). mix test: 115 tests, 0 failures, 49 GPU excluded, no warnings. New test/resource_query_test.exs (find_experiment, the calculations the data layer can evaluate); test/query_plan_test.exs extended to the engine's read paths. Commits e21889d 7193f00 86e40a1 94f1294 0f11407 fdfc644 29442e0 38c51b3.
<!-- SECTION:NOTES:END -->
