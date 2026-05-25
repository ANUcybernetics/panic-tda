defmodule Mix.Tasks.Cleanup.Phi4visionRuns do
  @shortdoc "Delete every run whose network includes Phi4Vision"

  @moduledoc """
  Deletes every Run (and its invocations, embeddings, embedding-cluster
  assignments, persistence diagrams, and matching Lyapunov rows) whose
  network includes the broken `Phi4Vision` I2T model. Sibling runs in
  the same experiments that use other I2T models are left untouched.

  Does not touch global `ClusteringResult` rows — run
  `mix cluster.recompute` afterwards to refresh the global clustering
  on the cleaned data.

      $ mix cleanup.phi4vision_runs
      $ mix cleanup.phi4vision_runs --force
  """

  use Mix.Task

  import Ecto.Query
  require Ash.Query

  @impl Mix.Task
  def run(args) do
    {opts, _, _} = OptionParser.parse(args, strict: [force: :boolean])

    Mix.Task.run("ecto.create", ["--quiet"])
    Mix.Task.run("ecto.migrate", ["--quiet"])
    Mix.Task.run("app.start")

    bad_run_ids =
      PanicTda.Run
      |> Ash.read!()
      |> Enum.filter(&("Phi4Vision" in &1.network))
      |> Enum.map(& &1.id)

    bad_lyapunov =
      PanicTda.LyapunovResult
      |> Ash.read!()
      |> Enum.filter(&("Phi4Vision" in (&1.network || [])))

    Mix.shell().info("Phi4Vision cleanup scope:")
    Mix.shell().info("  runs:           #{length(bad_run_ids)}")
    Mix.shell().info("  lyapunov rows:  #{length(bad_lyapunov)}")

    if length(bad_run_ids) == 0 do
      Mix.shell().info("Nothing to delete.")
    else
      proceed? = opts[:force] || Mix.shell().yes?("Delete?")

      if proceed? do
        delete_in_order(bad_run_ids, bad_lyapunov)
        Mix.shell().info("Done. Run `mix cluster.recompute` to refresh global clustering.")
      else
        Mix.shell().info("Aborted.")
      end
    end
  end

  defp delete_in_order(bad_run_ids, bad_lyapunov) do
    Mix.shell().info("Deleting embedding_clusters...")

    PanicTda.EmbeddingCluster
    |> Ash.Query.filter(
      embedding.invocation.run_id in ^bad_run_ids or
        medoid_embedding.invocation.run_id in ^bad_run_ids
    )
    |> Ash.bulk_destroy!(:destroy, %{}, return_errors?: true)

    Mix.shell().info("Deleting embeddings...")

    PanicTda.Embedding
    |> Ash.Query.filter(invocation.run_id in ^bad_run_ids)
    |> Ash.bulk_destroy!(:destroy, %{}, return_errors?: true)

    Mix.shell().info("Deleting persistence_diagrams...")

    PanicTda.PersistenceDiagram
    |> Ash.Query.filter(run_id in ^bad_run_ids)
    |> Ash.bulk_destroy!(:destroy, %{}, return_errors?: true)

    Mix.shell().info("Deleting lyapunov_results...")
    Enum.each(bad_lyapunov, &Ash.destroy!/1)

    Mix.shell().info("Clearing input_invocation_id self-references on bad invocations...")

    from(i in "invocations", where: i.run_id in ^bad_run_ids)
    |> PanicTda.Repo.update_all(set: [input_invocation_id: nil])

    Mix.shell().info("Deleting invocations...")

    PanicTda.Invocation
    |> Ash.Query.filter(run_id in ^bad_run_ids)
    |> Ash.bulk_destroy!(:destroy, %{}, return_errors?: true)

    Mix.shell().info("Deleting runs...")

    PanicTda.Run
    |> Ash.Query.filter(id in ^bad_run_ids)
    |> Ash.bulk_destroy!(:destroy, %{}, return_errors?: true)
  end
end
