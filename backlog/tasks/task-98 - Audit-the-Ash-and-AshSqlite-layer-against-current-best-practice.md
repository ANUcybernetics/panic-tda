---
id: TASK-98
title: Audit the Ash and AshSqlite layer against current best practice
status: In Progress
assignee:
  - '@claude'
created_date: '2026-09-05 00:39'
updated_date: '2026-09-05 11:05'
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
- [ ] #3 The CAST-on-primary-key behaviour either fixed at the AshSqlite level or reported upstream with a minimal reproduction
- [x] #4 Raw Ecto and raw SQL call sites audited --- PanicTda.Repo.transaction in the recompute, the sqlite3 reads in analysis/ --- and either moved to an Ash equivalent or recorded as deliberate
- [x] #5 Custom code that Ash now covers (manual paging, custom calculations, bulk actions, identities) identified and replaced where the Ash version is simpler
- [x] #6 No behaviour change: the full non-GPU suite passes, and any resource change is covered by a test
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

THE CAST BUG (AC#2, AC#3). Reproduced with a probe resource: the atomic update path emits WHERE (CAST(id AS TEXT) = CAST(? AS TEXT)) and plans as SCAN; the non-atomic path emits WHERE (id = ?) and plans as SEARCH. require_atomic?(false) is the only DSL switch that avoids it (atomic_upgrade?(false) is ignored when require_atomic? is true). Experiment and Run never reach the atomic path at all: AshSqlite has no expr_error, so any action with a validation is silently non-atomic, which is why Experiment's four flags were dead and are gone. The five guards on Invocation, Embedding, PersistenceDiagram, ClusteringResult and EmbeddingCluster stay, with a comment pointing at the report.

The bigger finding: reads have the same bug. Every Ash.Query.filter on a uuid column (run_id, experiment_id, id) renders both sides cast and full-scans; measured 171-178 ms per query on the 7.9 GB dev DB (308k invocations) against about 1 ms indexed, for the exact queries the embeddings, PD and resume stages issue per run. A one-function ash_sql fix (skip the cast when a Ref already has the target type) was validated on a local copy of the dep: every probed query plans SEARCH and the full suite passes. The dep was reverted; the patch, the reproduction and the measurements are in backlog/docs/ash-sql-cast-issue.md, ready to post upstream. AC#3 is left unchecked because posting it is Ben's call.

Impact on TASK-90: none that gates the launch. The run path is create-dominated; the per-run scans cost about 0.2 s per run (tens of minutes over a 33-67 GPU-day panel) and grow with the table. The slow paths are embeddings.recompute (its keyset page 'id > after' is an index-ordered scan, so page cost still grows with position), experiment.export and any analysis through Ash.

RAW ECTO / SQL (AC#4). Both lib sites are deliberate and now say so in comments: AshSqlite reports can?(:transact) false, so Ash.transaction is a no-op and the recompute keeps PanicTda.Repo.transaction; can?(:distinct) false, so ClusteringStage.all_embedding_models keeps its Ecto distinct query. The analysis/*.py sqlite3 reads are read-only SELECTs over the dev DB from Python (polars/numpy pipeline); the Ash-side equivalent is experiment.export_data to parquet. Deliberate, no change.

CUSTOM CODE (AC#5). Run.invocation_count and ClusteringResult.cluster_count were count/2 calculations that AshSqlite cannot evaluate at all (aggregates unsupported) and nothing used them: removed. The six copies of find_experiment (list every experiment, filter in Elixir) are one read action by_id_prefix plus a code-interface get; an ambiguous prefix now errors instead of picking the first match. EmbeddingsStage creates a run's embeddings with one Ash.bulk_create instead of a transaction per row; stored rows identical. Kept as-is with reasons: the recompute's manual keyset paging (Ash.stream! needs keyset pagination on the action and cannot resume from a printed id), the julianday duration calculations (Ash has no datetime subtraction), identities. Not touched before the launch: RunExecutor's per-item invocation creates, whose ordering the resume logic reasons about.

VERIFICATION (AC#6). mix test: 107 tests, 0 failures, 49 GPU excluded, no warnings. New test/resource_query_test.exs covers find_experiment (found, missing, ambiguous) and the three calculations the data layer can evaluate. Commits e21889d (reflow) 7193f00 86e40a1 94f1294 0f11407.
<!-- SECTION:NOTES:END -->
