# ash_sql casts typed columns and defeats SQLite indexes

Draft of an upstream report for `ash-project/ash_sql`, with the reproduction
and measurements it needs. Found while auditing this project's Ash layer
(TASK-98); the local symptom was first seen in TASK-96's recompute.

## Summary

When a filter compares a column to a value, `AshSql.Expr.maybe_type_expr/6`
wraps _both_ operands in `type/2`. On Postgres a cast of a `uuid` column to
`uuid` is optimised away. On SQLite it is rendered as
`CAST("id" AS TEXT)` and the planner never sees through a `CAST`, so every
filter on a typed column (every uuid primary key and foreign key) becomes a
full table scan.

```sql
-- Ash.Query.filter(PanicTda.Invocation, run_id == ^run_id)
SELECT ... FROM "invocations" AS i0
WHERE (CAST(i0."run_id" AS TEXT) = CAST(? AS TEXT))
-- EXPLAIN QUERY PLAN: SCAN i0

-- the same predicate without the column-side cast
WHERE (i0."run_id" = CAST(? AS TEXT))
-- EXPLAIN QUERY PLAN: SEARCH i0 USING INDEX invocations_unique_run_sequence_index (run_id=?)
```

The same predicate is generated for atomic updates (`Ash.update` on an
action that can be made atomic, `Ash.bulk_update` with `strategy: :atomic`),
because the primary key filter goes through the same expression builder.
The non-atomic update path uses `Ecto.Query.where(query, ^pkey)` and is
indexed.

ash_sql 0.5.0 added "don't cast integer/string equality check unnecessarily"
(`no_cast_for_native_value?/2`), which exempts a _value_ of native string or
integer type. A uuid value is a string but its Ash type is `Ash.Type.UUIDv7`,
so the exemption does not apply, and nothing exempts the column side.

## Versions

ash 3.23.1, ash_sql 0.5.5, ash_sqlite 0.2.16, ecto_sqlite3 0.22.0, exqlite
0.36.0, Elixir 1.20.0-rc.1.

## Reproduction

Any AshSqlite resource with a `uuid_v7_primary_key` and a `belongs_to`:

```elixir
require Ash.Query

PanicTda.Invocation
|> Ash.Query.filter(run_id == ^run.id)
|> Ash.read!()
```

Attach to `[:my_app, :repo, :query]` telemetry (or set `log: :debug` on the
repo) and run `EXPLAIN QUERY PLAN` on the emitted SQL. Every uuid predicate
in this project's read paths plans as `SCAN`; `test/query_plan_test.exs`
shows the telemetry capture and the plan assertion.

## Measurements

On this project's development database (7.9 GB, 308,592 invocations,
147,793 embeddings), read-only, warm cache:

| query                                         | with cast          | without cast    |
| --------------------------------------------- | ------------------ | --------------- |
| last invocation of a run (resume)             | 178 ms, SCAN       | 0 ms, SEARCH    |
| all invocations of a run (`Ash.load!`)        | 176 ms, SCAN       | 1 ms, SEARCH    |
| embeddings of a run via `invocation.run_id`   | 171 ms, SCAN       | 1 ms, SEARCH    |
| keyset page `id > ^after` on embeddings       | index-ordered SCAN | SEARCH `(id>?)` |
| atomic update by primary key (TASK-96 report) | 72 ms / row        | 2 ms / row      |

The keyset row matters for anyone paging with `id > ^last`: the plan says
`SCAN ... USING INDEX`, which walks the index from the start evaluating the
`CAST` on each entry, so page cost still grows with position.

## Candidate fix

Skip the cast when the expression is a `Ref` to an attribute whose type
already is the target type. Casting a column to its own type is a no-op in
every dialect, so this is safe for Postgres too; it only removes a cast the
Postgres planner was already eliding. The parameter side keeps its cast,
which is what makes the comparison well typed.

```elixir
# lib/expr.ex
defp maybe_type_expr(query, expr, bindings, embedded?, acc, type) do
  skip? =
    no_cast_for_native_value?(expr, type) ||
      (is_struct(expr, Ash.Query.Ref) && bindings[:skip_cast_for_ref?]) ||
      ref_already_of_type?(expr, type)
  ...

# Casting a column to the type it already has is a no-op, but some planners
# (SQLite) cannot see through it and turn an indexed lookup into a scan.
defp ref_already_of_type?(
       %Ash.Query.Ref{attribute: %Ash.Resource.Attribute{type: ref_type}},
       type
     ) do
  resolve_type(type) == resolve_type(ref_type)
end

defp ref_already_of_type?(_, _), do: false

defp resolve_type({type, _constraints}) when is_atom(type), do: resolve_type(type)
defp resolve_type({:array, type}), do: {:array, resolve_type(type)}
defp resolve_type(type) when is_atom(type), do: Ash.Type.get_type(type)
defp resolve_type(other), do: other
```

Note the change from `or`/`and` to `||`/`&&`: `bindings[:skip_cast_for_ref?]`
is usually `nil`, which `or` rejects once it is no longer the last operand.

With this patch applied to the local dep, every query in the table above
plans as `SEARCH`, the atomic update by primary key plans as `SEARCH`, and
this project's full test suite passes. The patch does not touch `LIKE`:
`like(experiment_id, ^prefix)` still casts, because the ref is a uuid and
the target type is string, which is the intended behaviour.

## A second AshSqlite bug found on the same path

An atomic update of a `:map` attribute hands the map to exqlite unencoded:

```
** (Exqlite.Error) unsupported type: %{"k" => 3}
UPDATE "clustering_results" AS c0 SET ..., "parameters" = ? WHERE ...
```

The non-atomic path encodes it as JSON. `PanicTda.ClusteringResult.update`
keeps `require_atomic?(false)` for this reason alone; worth a separate
ash_sqlite issue when the cast report goes up.

## What this project does meanwhile

Every uuid column that a query filters on carries an expression index on
`CAST(col AS TEXT)`, declared through `custom_indexes` on the resource and
generated into `priv/repo/migrations/*_add_cast_indexes.exs` (13 indexes,
138 MB and 6.6 s on the 7.5 GB dev database). SQLite matches the index
against the predicate ash_sql emits, so reads and atomic updates plan as
`SEARCH`; `test/query_plan_test.exs` asserts this for the engine's real
queries, and on the dev database they run in 0-11 ms where they took about
175 ms.

The one query that needs more is the recompute's keyset page: with
`ORDER BY id` the planner prefers an ordered walk of the primary key and
re-evaluates the cast from the first row every page, so
`PanicTda.Embeddings.Recompute` sorts by `expr_sort(fragment("?", id))`,
which renders as the same `CAST(id AS TEXT)` the index is built on and turns
the page into a seek.

The indexes encode an ash_sql quirk in the schema, so TASK-99 drops them
once a release contains the fix.
