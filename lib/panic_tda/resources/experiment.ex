defmodule PanicTda.Experiment do
  use Ash.Resource,
    domain: PanicTda,
    data_layer: AshSqlite.DataLayer

  sqlite do
    table("experiments")
    repo(PanicTda.Repo)

    # Expression indexes for the CAST predicates ash_sql emits on uuid columns
    # (backlog/docs/ash-sql-cast-issue.md); drop them with TASK-99.
    custom_indexes do
      index(["CAST(id AS TEXT)"], name: "experiments_id_text_index")
    end
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

    # Uniform generation ceiling applied to every image-to-text model in the
    # experiment; nil keeps each model's own default from panic_models.py.
    attribute :i2t_max_new_tokens, :integer do
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

    # The Mix tasks take an id prefix; `get?: true` on the code interface
    # makes an ambiguous prefix an error rather than a silent first match.
    read :by_id_prefix do
      argument(:prefix, :string, allow_nil?: false)
      filter(expr(like(id, ^arg(:prefix) <> "%")))
    end

    create :create do
      accept([
        :networks,
        :num_runs,
        :prompts,
        :embedding_models,
        :max_length,
        :i2t_max_new_tokens
      ])
    end

    update :update do
      accept([
        :networks,
        :num_runs,
        :prompts,
        :embedding_models,
        :max_length,
        :i2t_max_new_tokens
      ])
    end

    # Function captures, not `expr(now())`: AshSqlite never runs these
    # atomically (no expr_error support), and on the non-atomic path
    # set_attribute does not evaluate expressions.
    update :start do
      change(set_attribute(:started_at, &DateTime.utc_now/0))
    end

    update :complete do
      change(set_attribute(:completed_at, &DateTime.utc_now/0))
    end

    update :reopen do
      change(set_attribute(:completed_at, nil))
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

    validate compare(:i2t_max_new_tokens, greater_than: 0) do
      message("must be greater than 0")
      where(present(:i2t_max_new_tokens))
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
