defmodule PanicTda.Repo.Migrations.SimplifyExperimentModel do
  @moduledoc """
  Simplify experiment model: single network, remove seeds, add num_runs/run_number.
  """

  use Ecto.Migration

  def up do
    drop_if_exists unique_index(:runs, [:experiment_id, :initial_prompt, :seed],
                     name: "runs_unique_experiment_run_index"
                   )

    alter table(:runs) do
      remove :seed
      add :run_number, :bigint, null: false
    end

    create unique_index(:runs, [:experiment_id, :initial_prompt, :run_number],
             name: "runs_unique_experiment_run_index"
           )

    alter table(:invocations) do
      remove :seed
    end

    alter table(:experiments) do
      remove :seeds
      remove :networks
      add :network, {:array, :text}, null: false
      add :num_runs, :bigint, null: false
    end
  end

  def down do
    alter table(:experiments) do
      remove :num_runs
      remove :network
      add :networks, {:array, {:array, :text}}, null: false
      add :seeds, {:array, :bigint}, null: false
    end

    alter table(:invocations) do
      add :seed, :bigint, null: false
    end

    drop_if_exists unique_index(:runs, [:experiment_id, :initial_prompt, :run_number],
                     name: "runs_unique_experiment_run_index"
                   )

    alter table(:runs) do
      remove :run_number
      add :seed, :bigint, null: false
    end

    create unique_index(:runs, [:experiment_id, :initial_prompt, :seed],
             name: "runs_unique_experiment_run_index"
           )
  end
end
