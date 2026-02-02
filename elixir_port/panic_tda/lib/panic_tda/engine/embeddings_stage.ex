defmodule PanicTda.Engine.EmbeddingsStage do
  @moduledoc """
  Computes embeddings for all invocations in a run.
  """

  alias PanicTda.Models.Embeddings

  def compute(env, run, embedding_models) do
    invocations =
      run
      |> Ash.load!(:invocations)
      |> Map.get(:invocations, [])

    Enum.each(embedding_models, fn embedding_model ->
      compute_for_model(env, invocations, embedding_model)
    end)

    :ok
  end

  defp compute_for_model(env, invocations, embedding_model) do
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

      case Embeddings.embed(env, embedding_model, contents) do
        {:ok, vectors} ->
          completed_at = DateTime.utc_now()

          Enum.zip(relevant_invocations, vectors)
          |> Enum.each(fn {inv, vector_binary} ->
            PanicTda.Embedding
            |> Ash.Changeset.for_create(:create, %{
              embedding_model: embedding_model,
              vector: vector_binary,
              started_at: started_at,
              completed_at: completed_at,
              invocation_id: inv.id
            })
            |> Ash.create!()
          end)

          :ok

        {:error, reason} ->
          {:error, reason}
      end
    end
  end
end
