defmodule PanicTda.Experiment do
  use Ash.Resource,
    domain: PanicTda,
    data_layer: AshSqlite.DataLayer

  sqlite do
    table("experiments")
    repo(PanicTda.Repo)
  end

  attributes do
    uuid_v7_primary_key(:id)

    attribute :networks, {:array, {:array, :string}} do
      allow_nil?(false)
      public?(true)
      constraints(min_length: 1)
    end

    attribute :num_runs, :integer do
      allow_nil?(false)
      public?(true)
      default(1)
    end

    attribute :prompts, {:array, :string} do
      allow_nil?(false)
      public?(true)
      constraints(min_length: 1)
    end

    attribute :embedding_models, {:array, :string} do
      allow_nil?(false)
      public?(true)
      constraints(min_length: 1)
    end

    attribute :max_length, :integer do
      allow_nil?(false)
      public?(true)
    end

    attribute :started_at, :utc_datetime_usec do
      public?(true)
    end

    attribute :completed_at, :utc_datetime_usec do
      public?(true)
    end

    create_timestamp(:inserted_at)
    update_timestamp(:updated_at)
  end

  relationships do
    has_many :runs, PanicTda.Run do
      destination_attribute(:experiment_id)
    end

    has_many :clustering_results, PanicTda.ClusteringResult do
      destination_attribute(:experiment_id)
    end
  end

  actions do
    defaults([:read, :destroy])

    create :create do
      accept([:networks, :num_runs, :prompts, :embedding_models, :max_length])
    end

    update :update do
      accept([:networks, :num_runs, :prompts, :embedding_models, :max_length])
      require_atomic?(false)
    end

    update :start do
      require_atomic?(false)
      change(set_attribute(:started_at, &DateTime.utc_now/0))
    end

    update :complete do
      require_atomic?(false)
      change(set_attribute(:completed_at, &DateTime.utc_now/0))
    end
  end

  validations do
    validate compare(:max_length, greater_than: 0) do
      message("must be greater than 0")
      on([:create, :update])
    end

    validate compare(:num_runs, greater_than: 0) do
      message("must be greater than 0")
      on([:create, :update])
    end

    validate {PanicTda.Validations.NonEmptyNestedLists, attribute: :networks} do
      on([:create])
    end

    validate {PanicTda.Validations.NonEmptyList, attribute: :prompts} do
      on([:create])
    end

    validate {PanicTda.Validations.NonEmptyList, attribute: :embedding_models} do
      on([:create])
    end

    validate {PanicTda.Validations.TimestampOrder, []} do
      on([:create])
    end
  end
end
