defmodule PanicTda.Engine.EmbeddingsStage do
  @moduledoc """
  Computes embeddings for all invocations in a run.
  """

  require Ash.Query

  alias PanicTda.Models.Embeddings

  def compute(env, run, embedding_models) do
    invocations = load_invocations(run)

    Enum.each(embedding_models, fn embedding_model ->
      :ok = compute_for_invocations(env, invocations, embedding_model)
    end)

    :ok
  end

  def resume(env, run, embedding_models) do
    invocations = load_invocations(run)

    Enum.each(embedding_models, fn embedding_model ->
      embedded_invocation_ids =
        PanicTda.Embedding
        |> Ash.Query.filter(
          invocation.run_id == ^run.id and embedding_model == ^embedding_model
        )
        |> Ash.Query.load(:invocation)
        |> Ash.read!()
        |> MapSet.new(& &1.invocation.id)

      missing = Enum.reject(invocations, &MapSet.member?(embedded_invocation_ids, &1.id))
      :ok = compute_for_invocations(env, missing, embedding_model)
    end)

    :ok
  end

  defp load_invocations(run) do
    run
    |> Ash.load!(:invocations)
    |> Map.get(:invocations, [])
  end

  def compute_for_invocations(_env, [], _embedding_model), do: :ok

  def compute_for_invocations(env, invocations, embedding_model) do
    model_type = Embeddings.model_type(embedding_model)

    relevant_invocations =
      Enum.filter(invocations, fn inv ->
        case model_type do
          :text -> inv.type == :text
          :image -> inv.type == :image
        end
      end)

    if relevant_invocations == [] do
      :ok
    else
      contents =
        Enum.map(relevant_invocations, fn inv ->
          case model_type do
            :text -> inv.output_text
            :image -> inv.output_image
          end
        end)

      started_at = DateTime.utc_now()
      {:ok, vectors} = Embeddings.embed(env, embedding_model, contents)
      completed_at = DateTime.utc_now()

      Enum.zip(relevant_invocations, vectors)
      |> Enum.each(fn {inv, vector_binary} ->
        PanicTda.create_embedding!(%{
          embedding_model: embedding_model,
          vector: vector_binary,
          started_at: started_at,
          completed_at: completed_at,
          invocation_id: inv.id
        })
      end)

      :ok
    end
  end
end
