defmodule PanicTda.Engine.RunExecutor do
  @moduledoc """
  Executes runs through their model network.
  Supports both single-run and batch execution.

  Every function here takes a `PanicTda.Models.PythonSession` rather than a
  raw env, so that a step whose Python process died can be retried against a
  fresh one (see `PanicTda.Engine.Retry`).
  """

  require Ash.Query

  alias PanicTda.Engine.Retry
  alias PanicTda.Models.{GenAI, PythonBridge, PythonSession}

  def execute(session, run) do
    execute_loop(session, run, run.initial_prompt, 0, nil)
  end

  def resume(session, run) do
    invocations =
      PanicTda.Invocation
      |> Ash.Query.filter(run_id == ^run.id)
      |> Ash.Query.sort(sequence_number: :desc)
      |> Ash.Query.limit(1)
      |> Ash.read!()

    case invocations do
      [] ->
        execute(session, run)

      [last_invocation] ->
        next_seq = last_invocation.sequence_number + 1

        if next_seq >= run.max_length do
          :ok
        else
          output =
            case last_invocation.type do
              :text -> last_invocation.output_text
              :image -> last_invocation.output_image
            end

          execute_loop(session, run, output, next_seq, last_invocation.id)
        end
    end
  end

  def execute_batch(session, runs) do
    network = hd(runs).network
    max_length = hd(runs).max_length

    states =
      Enum.map(runs, fn run ->
        %{run: run, input: run.initial_prompt, prev_invocation_id: nil}
      end)

    execute_batch_loop(session, network, max_length, states, 0)
  end

  def resume_batch(session, runs) do
    states =
      Enum.map(runs, fn run ->
        invocations =
          PanicTda.Invocation
          |> Ash.Query.filter(run_id == ^run.id)
          |> Ash.Query.sort(sequence_number: :desc)
          |> Ash.Query.limit(1)
          |> Ash.read!()

        case invocations do
          [] ->
            %{run: run, input: run.initial_prompt, prev_invocation_id: nil, completed_seq: -1}

          [last] ->
            output =
              case last.type do
                :text -> last.output_text
                :image -> last.output_image
              end

            %{
              run: run,
              input: output,
              prev_invocation_id: last.id,
              completed_seq: last.sequence_number
            }
        end
      end)

    min_completed = states |> Enum.map(& &1.completed_seq) |> Enum.min()
    start_seq = min_completed + 1

    network = hd(runs).network
    max_length = hd(runs).max_length

    if start_seq >= max_length do
      :ok
    else
      resume_states =
        Enum.map(states, fn state ->
          Map.drop(state, [:completed_seq])
        end)

      execute_batch_loop(session, network, max_length, resume_states, start_seq)
    end
  end

  defp execute_batch_loop(_session, _network, max_length, _states, seq) when seq >= max_length do
    :ok
  end

  defp execute_batch_loop(session, network, max_length, states, seq) do
    model_name = Enum.at(network, rem(seq, length(network)))
    output_type = GenAI.output_type(model_name)
    inputs = Enum.map(states, & &1.input)

    seeds = if GenAI.seeded?(model_name), do: Enum.map(inputs, fn _ -> GenAI.draw_seed() end)

    started_at = DateTime.utc_now()

    {:ok, outputs} =
      Retry.with_retry(
        "#{model_name} step #{seq} (batch of #{length(inputs)})",
        session,
        fn env ->
          GenAI.invoke_batch(env, model_name, inputs, seeds)
        end
      )

    completed_at = DateTime.utc_now()

    next_seq = seq + 1
    next_model = Enum.at(network, rem(next_seq, length(network)))

    if next_seq < max_length and next_model != model_name do
      :ok = PythonBridge.swap_model_to_cpu(PythonSession.env(session), model_name)
    end

    new_states =
      Enum.zip([states, outputs, seeds || Enum.map(states, fn _ -> nil end)])
      |> Enum.map(fn {state, output, seed} ->
        attrs = %{
          model: model_name,
          type: output_type,
          sequence_number: seq,
          seed: seed,
          started_at: started_at,
          completed_at: completed_at,
          run_id: state.run.id,
          input_invocation_id: state.prev_invocation_id
        }

        attrs =
          case output_type do
            :text -> Map.put(attrs, :output_text, output)
            :image -> Map.put(attrs, :output_image, output)
          end

        invocation = PanicTda.create_invocation!(attrs)

        %{state | input: output, prev_invocation_id: invocation.id}
      end)

    execute_batch_loop(session, network, max_length, new_states, next_seq)
  end

  defp execute_loop(_session, run, _input, seq, _prev_id) when seq >= run.max_length do
    :ok
  end

  defp execute_loop(session, run, input, seq, prev_invocation_id) do
    model_name = Enum.at(run.network, rem(seq, length(run.network)))
    output_type = GenAI.output_type(model_name)

    seed = if GenAI.seeded?(model_name), do: GenAI.draw_seed()

    started_at = DateTime.utc_now()

    {:ok, output} =
      Retry.with_retry("#{model_name} step #{seq}", session, fn env ->
        GenAI.invoke(env, model_name, input, seed)
      end)

    completed_at = DateTime.utc_now()

    next_seq = seq + 1
    next_model = Enum.at(run.network, rem(next_seq, length(run.network)))

    if next_seq < run.max_length and next_model != model_name do
      :ok = PythonBridge.swap_model_to_cpu(PythonSession.env(session), model_name)
    end

    attrs = %{
      model: model_name,
      type: output_type,
      sequence_number: seq,
      seed: seed,
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

    execute_loop(session, run, output, next_seq, invocation.id)
  end
end
