defmodule PanicTda.Engine.RunExecutor do
  @moduledoc """
  Executes a single run through its model network.
  Creates invocations sequentially, feeding output to input.
  """

  alias PanicTda.Models.GenAI

  def execute(env, run) do
    execute_loop(env, run, run.initial_prompt, 0, nil)
  end

  defp execute_loop(_env, run, _input, seq, _prev_id) when seq >= run.max_length do
    :ok
  end

  defp execute_loop(env, run, input, seq, prev_invocation_id) do
    model_name = Enum.at(run.network, rem(seq, length(run.network)))
    output_type = GenAI.output_type(model_name)

    started_at = DateTime.utc_now()
    {:ok, output} = GenAI.invoke(env, model_name, input, run.seed)
    completed_at = DateTime.utc_now()

    attrs = %{
      model: model_name,
      type: output_type,
      seed: run.seed,
      sequence_number: seq,
      started_at: started_at,
      completed_at: completed_at,
      run_id: run.id,
      input_invocation_id: prev_invocation_id
    }

    attrs =
      case output_type do
        :text -> Map.put(attrs, :output_text, output)
        :image -> Map.put(attrs, :output_image, output)
      end

    invocation = PanicTda.create_invocation!(attrs)

    execute_loop(env, run, output, seq + 1, invocation.id)
  end
end
