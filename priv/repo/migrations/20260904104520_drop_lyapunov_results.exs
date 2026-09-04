defmodule PanicTda.Repo.Migrations.DropLyapunovResults do
  @moduledoc """
  Drops the Lyapunov stage's table.

  FTLE was removed from the analysis (see TASK-73): the embeddings are
  L2-normalised onto the unit sphere, so pairwise distances are bounded and
  cannot grow exponentially, which is why the fits were consistently weak and
  distant prompts gave smaller exponents than identical ones. Hand-written
  rather than generated, since the resource no longer exists for Ash to diff
  against.
  """

  use Ecto.Migration

  def up do
    drop table(:lyapunov_results)
  end

  def down do
    raise Ecto.MigrationError,
      message:
        "irreversible: the LyapunovResult resource has been removed, so the " <>
          "table cannot be recreated from a snapshot. Restore from git history " <>
          "if this is ever needed."
  end
end
