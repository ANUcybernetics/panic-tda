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
    end

    attribute :seeds, {:array, :integer} do
      allow_nil?(false)
      public?(true)
    end

    attribute :prompts, {:array, :string} do
      allow_nil?(false)
      public?(true)
    end

    attribute :embedding_models, {:array, :string} do
      allow_nil?(false)
      public?(true)
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
  end

  actions do
    defaults([:read, :destroy])

    create :create do
      accept([:networks, :seeds, :prompts, :embedding_models, :max_length])
    end

    update :update do
      require_atomic?(false)
      accept([:networks, :seeds, :prompts, :embedding_models, :max_length])
    end

    update :start do
      change(set_attribute(:started_at, &DateTime.utc_now/0))
    end

    update :complete do
      change(set_attribute(:completed_at, &DateTime.utc_now/0))
    end
  end

  validations do
    validate(present([:networks, :seeds, :prompts, :embedding_models, :max_length]))

    validate compare(:max_length, greater_than: 0) do
      message("must be greater than 0")
    end
  end
end
