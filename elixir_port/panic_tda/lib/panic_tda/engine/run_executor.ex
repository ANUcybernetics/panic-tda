defmodule PanicTda.Engine.RunExecutor do
  @moduledoc """
  Executes a single run through its model network.
  Creates invocations sequentially, feeding output to input.
  """

  alias PanicTda.Models.GenAI

  def execute(env, run) do
    execute_loop(env, run, run.initial_prompt, 0, [])
  end

  defp execute_loop(_env, run, _input, seq, invocation_ids) when seq >= run.max_length do
    {:ok, invocation_ids}
  end

  defp execute_loop(env, run, input, seq, invocation_ids) do
    model_name = Enum.at(run.network, rem(seq, length(run.network)))
    output_type = GenAI.output_type(model_name)

    started_at = DateTime.utc_now()

    case GenAI.invoke(env, model_name, input, run.seed) do
      {:ok, output} ->
        completed_at = DateTime.utc_now()

        attrs = %{
          model: model_name,
          type: output_type,
          seed: run.seed,
          sequence_number: seq,
          started_at: started_at,
          completed_at: completed_at,
          run_id: run.id,
          input_invocation_id: List.last(invocation_ids)
        }

        attrs =
          case output_type do
            :text -> Map.put(attrs, :output_text, output)
            :image -> Map.put(attrs, :output_image, output)
          end

        {:ok, invocation} =
          PanicTda.Invocation
          |> Ash.Changeset.for_create(:create, attrs)
          |> Ash.create()

        next_input =
          case output_type do
            :text -> output
            :image -> output
          end

        execute_loop(env, run, next_input, seq + 1, invocation_ids ++ [invocation.id])

      {:error, reason} ->
        {:error, reason}
    end
  end
end
