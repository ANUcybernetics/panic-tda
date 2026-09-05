defmodule PanicTda.PersistenceDiagram do
  use Ash.Resource,
    domain: PanicTda,
    data_layer: AshSqlite.DataLayer

  sqlite do
    table("persistence_diagrams")
    repo(PanicTda.Repo)
  end

  attributes do
    uuid_v7_primary_key(:id)

    attribute :embedding_model, :string do
      allow_nil?(false)
      public?(true)
    end

    attribute :diagram_data, PanicTda.Types.PersistenceDiagramData do
      public?(true)
    end

    attribute :started_at, :utc_datetime_usec do
      allow_nil?(false)
      public?(true)
    end

    attribute :completed_at, :utc_datetime_usec do
      public?(true)
    end

    create_timestamp(:inserted_at)
    update_timestamp(:updated_at)
  end

  relationships do
    belongs_to :run, PanicTda.Run do
      allow_nil?(false)
      attribute_type(:uuid_v7)
    end
  end

  identities do
    identity(:unique_run_model, [:run_id, :embedding_model])
  end

  actions do
    defaults([:read, :destroy])

    create :create do
      accept([:embedding_model, :diagram_data, :started_at, :completed_at, :run_id])
    end

    update :update do
      # Non-atomic until the ash_sql cast fix ships
      # (backlog/docs/ash-sql-cast-issue.md): the atomic path's CAST on the
      # primary key defeats the index (test/query_plan_test.exs).
      require_atomic?(false)
      accept([:diagram_data, :completed_at])
    end
  end

  validations do
    validate {PanicTda.Validations.TimestampOrder, []} do
      on([:create])
    end
  end

  calculations do
    calculate(
      :duration,
      :float,
      expr(fragment("(julianday(?) - julianday(?)) * 86400", completed_at, started_at))
    )
  end
end
