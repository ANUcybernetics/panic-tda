defmodule PanicTda.ClusteringResult do
  use Ash.Resource,
    domain: PanicTda,
    data_layer: AshSqlite.DataLayer

  sqlite do
    table("clustering_results")
    repo(PanicTda.Repo)
  end

  attributes do
    uuid_v7_primary_key(:id)

    attribute :embedding_model, :string do
      allow_nil?(false)
      public?(true)
    end

    attribute :algorithm, :string do
      allow_nil?(false)
      public?(true)
    end

    attribute :parameters, :map do
      default(%{})
      public?(true)
    end

    attribute :layer, :integer do
      allow_nil?(false)
      default(0)
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
    has_many :embedding_clusters, PanicTda.EmbeddingCluster do
      destination_attribute(:clustering_result_id)
    end
  end

  actions do
    defaults([:read, :destroy])

    create :create do
      accept([
        :embedding_model,
        :algorithm,
        :parameters,
        :layer,
        :started_at,
        :completed_at
      ])
    end

    update :update do
      # Non-atomic until the ash_sql cast fix ships
      # (backlog/docs/ash-sql-cast-issue.md): the atomic path's CAST on the
      # primary key defeats the index (test/query_plan_test.exs).
      require_atomic?(false)
      accept([:parameters, :completed_at])
    end
  end

  validations do
    validate {PanicTda.Validations.TimestampOrder, []} do
      on([:create])
    end
  end

  identities do
    identity(:unique_layer, [:embedding_model, :algorithm, :layer])
  end

  calculations do
    calculate(
      :duration,
      :float,
      expr(fragment("(julianday(?) - julianday(?)) * 86400", completed_at, started_at))
    )
  end
end
