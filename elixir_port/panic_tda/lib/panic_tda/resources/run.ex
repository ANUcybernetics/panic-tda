defmodule PanicTda.Run do
  use Ash.Resource,
    domain: PanicTda,
    data_layer: AshSqlite.DataLayer

  sqlite do
    table("runs")
    repo(PanicTda.Repo)
  end

  attributes do
    uuid_v7_primary_key(:id)

    attribute :network, {:array, :string} do
      allow_nil?(false)
      public?(true)
    end

    attribute :seed, :integer do
      allow_nil?(false)
      public?(true)
    end

    attribute :max_length, :integer do
      allow_nil?(false)
      public?(true)
    end

    attribute :initial_prompt, :string do
      allow_nil?(false)
      public?(true)
    end

    create_timestamp(:inserted_at)
    update_timestamp(:updated_at)
  end

  relationships do
    belongs_to :experiment, PanicTda.Experiment do
      allow_nil?(false)
      attribute_type(:uuid_v7)
    end

    has_many :invocations, PanicTda.Invocation do
      destination_attribute(:run_id)
      sort(sequence_number: :asc)
    end

    has_many :persistence_diagrams, PanicTda.PersistenceDiagram do
      destination_attribute(:run_id)
    end
  end

  actions do
    defaults([:read, :destroy])

    create :create do
      accept([:network, :seed, :max_length, :initial_prompt, :experiment_id])
    end

    update :update do
      accept([:network, :seed, :max_length, :initial_prompt])
    end
  end

  calculations do
    calculate(:invocation_count, :integer, expr(count(invocations)))
  end

  validations do
    validate(present([:network, :seed, :max_length, :initial_prompt]))

    validate compare(:max_length, greater_than: 0) do
      message("must be greater than 0")
    end
  end
end
