defmodule PanicTda.Embedding do
  use Ash.Resource,
    domain: PanicTda,
    data_layer: AshSqlite.DataLayer

  sqlite do
    table("embeddings")
    repo(PanicTda.Repo)
  end

  attributes do
    uuid_v7_primary_key(:id)

    attribute :embedding_model, :string do
      allow_nil?(false)
      public?(true)
    end

    attribute :vector, PanicTda.Types.Vector do
      allow_nil?(false)
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
    belongs_to :invocation, PanicTda.Invocation do
      allow_nil?(false)
      attribute_type(:uuid_v7)
    end

    has_many :cluster_assignments, PanicTda.EmbeddingCluster do
      destination_attribute(:embedding_id)
    end
  end

  identities do
    identity(:unique_invocation_model, [:invocation_id, :embedding_model])
  end

  actions do
    defaults([:read, :destroy])

    create :create do
      accept([:embedding_model, :vector, :started_at, :completed_at, :invocation_id])
    end

    update :update do
      accept([:vector, :completed_at])
    end
  end

  validations do
    validate {PanicTda.Validations.TimestampOrder, []} do
      on([:create])
    end
  end

  calculations do
    calculate(:dimension, :integer, expr(fragment("length(?) / 4", vector)))

    calculate(
      :duration,
      :float,
      expr(fragment("(julianday(?) - julianday(?)) * 86400", completed_at, started_at))
    )
  end

end
