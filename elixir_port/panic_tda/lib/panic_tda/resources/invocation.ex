defmodule PanicTda.Invocation do
  use Ash.Resource,
    domain: PanicTda,
    data_layer: AshSqlite.DataLayer

  sqlite do
    table("invocations")
    repo(PanicTda.Repo)
  end

  attributes do
    uuid_v7_primary_key(:id)

    attribute :model, :string do
      allow_nil?(false)
      public?(true)
    end

    attribute :type, PanicTda.Types.InvocationType do
      allow_nil?(false)
      public?(true)
    end

    attribute :seed, :integer do
      allow_nil?(false)
      public?(true)
    end

    attribute :sequence_number, :integer do
      allow_nil?(false)
      default(0)
      public?(true)
    end

    attribute :output_text, :string do
      public?(true)
    end

    attribute :output_image, PanicTda.Types.Image do
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

    belongs_to :input_invocation, PanicTda.Invocation do
      allow_nil?(true)
      attribute_type(:uuid_v7)
    end

    has_many :embeddings, PanicTda.Embedding do
      destination_attribute(:invocation_id)
    end
  end

  actions do
    defaults([:read, :destroy])

    create :create do
      accept([
        :model,
        :type,
        :seed,
        :sequence_number,
        :output_text,
        :output_image,
        :started_at,
        :completed_at,
        :run_id,
        :input_invocation_id
      ])
    end

    update :update do
      accept([:output_text, :output_image, :completed_at])
    end
  end

  calculations do
    calculate(
      :duration,
      :float,
      expr(fragment("(julianday(?) - julianday(?)) * 86400", completed_at, started_at))
    )
  end

  validations do
    validate(present([:model, :type, :seed, :started_at]))
  end
end
